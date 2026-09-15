"""Does FFN redundancy in a forecasting foundation model grow with scale?

chronos_ablation.py established, on the 9M `tiny` checkpoint, that no single FFN
feature carries as much as 1% of forecasting accuracy and that half the features
can be removed for ~0% WQL change. The obvious objection is that this is an
artifact of a tiny, over-parameterised model.

This script runs the identical protocol across the whole Chronos-Bolt ladder
(9M -> 205M, 8192 -> 73728 FFN features) and on Chronos-2, the current
state of the art. Inference only -- no training -- so an A100 handles the whole
sweep in well under an hour.

The prediction worth stating in advance: if redundancy is a scale artifact, the
removable fraction should SHRINK with model size. If it grows, redundancy is a
property of how these models store forecasting behaviour, not of being small.

    pip install chronos-forecasting "transformers>=4.40,<4.48"
    python experiments/chronos_scaling.py --device cuda
"""
from __future__ import annotations

import argparse, json, time

import numpy as np
import torch

from chronos_ablation import (LEVELS, load_ett, make_windows, wql, naive_wql)

LADDER = [
    "amazon/chronos-bolt-tiny",    # 9M   -  8192 features
    "amazon/chronos-bolt-mini",    # 21M  - 12288
    "amazon/chronos-bolt-small",   # 48M  - 24576
    "amazon/chronos-bolt-base",    # 205M - 73728
]


def build(model_id, device):
    """Return (pipe, ffn_modules, names, n_feat). Handles differing depths and
    encoder-only models rather than assuming the `tiny` layout."""
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(model_id, device_map=device,
                                               torch_dtype=torch.float32)
    model = pipe.model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ffn, names = [], []
    for mod_name, mod in model.named_modules():
        if mod_name.endswith("DenseReluDense") or mod_name.endswith("DenseGatedActDense"):
            ffn.append(mod)
            names.append(mod_name.replace(".layer", "L").replace(".block", "B"))
    if not ffn:
        raise RuntimeError(f"no FFN modules found in {model_id}; layout differs")
    n_feat = ffn[0].wi.weight.shape[0] if hasattr(ffn[0], "wi") else ffn[0].wi_0.weight.shape[0]
    return pipe, ffn, names, n_feat


def scorer(pipe, ins, tgt, horizon):
    @torch.no_grad()
    def s():
        q, _ = pipe.predict_quantiles(inputs=ins, prediction_length=horizon,
                                      quantile_levels=list(LEVELS))
        return wql(q.detach().cpu().float().numpy(), tgt)
    return s


@torch.no_grad()
def ablate(ffn, n_feat, picks, score):
    saved = []
    for p in picks:
        L, j = int(p) // n_feat, int(p) % n_feat
        mod = ffn[L]
        saved.append((mod, j, mod.wi.weight[j].clone(), mod.wo.weight[:, j].clone()))
        mod.wi.weight[j] = 0
        mod.wo.weight[:, j] = 0
    out = score()
    for mod, j, row, col in saved:
        mod.wi.weight[j] = row
        mod.wo.weight[:, j] = col
    return out


@torch.no_grad()
def ablate_all(ffn, score):
    saved = [(m, m.wi.weight.clone(), m.wo.weight.clone()) for m in ffn]
    for m in ffn:
        m.wi.weight.zero_(); m.wo.weight.zero_()
    out = score()
    for m, wi, wo in saved:
        m.wi.weight.copy_(wi); m.wo.weight.copy_(wo)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dataset", default="ETTh1")
    ap.add_argument("--n-single", type=int, default=2000,
                    help="features sampled for the single-feature sweep (0 = exhaustive)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--models", nargs="*", default=LADDER)
    args = ap.parse_args()

    arr = load_ett(args.dataset)
    ins, tgt = make_windows(arr, 512, 64)
    print(f"{args.dataset}: {len(ins)} windows | naive WQL {naive_wql(ins, tgt, 64):.4f}\n")

    results = []
    for mid in args.models:
        t0 = time.time()
        try:
            pipe, ffn, names, n_feat = build(mid, args.device)
        except Exception as e:
            print(f"{mid}: SKIPPED ({type(e).__name__}: {e})\n"); continue
        ins_d = [x.to(args.device) for x in ins]
        score = scorer(pipe, ins_d, tgt, 64)
        total = len(ffn) * n_feat
        base = score()
        allz = ablate_all(ffn, score)
        print(f"=== {mid}  ({len(ffn)} FFN layers x {n_feat} = {total} features) ===")
        print(f"  baseline WQL {base:.4f} | CONTROL all-zeroed {allz:.4f} ({(allz-base)/base*100:+.0f}%)")
        if allz < base * 1.5:
            print("  !! control failed, skipping\n"); continue

        row = {"model": mid, "n_features": total, "baseline": base,
               "control_all_zeroed": allz}

        for frac in (0.25, 0.5, 0.75):
            k = int(total * frac)
            s = float(np.mean([ablate(ffn, n_feat,
                       np.random.default_rng(sd).choice(total, k, replace=False), score)
                       for sd in range(args.seeds)]))
            row[f"drop_{int(frac*100)}"] = (s - base) / base * 100
            print(f"  {frac:>4.0%} removed: WQL {s:.4f}  ({(s-base)/base*100:+.1f}%)")

        n = total if args.n_single == 0 else min(args.n_single, total)
        picks = np.random.default_rng(0).choice(total, n, replace=False)
        d = np.array([ablate(ffn, n_feat, [p], score) - base for p in picks])
        row.update(single_n=n, single_max_pct=float(d.max() / base * 100),
                   single_helps_pct=float((d < 0).mean() * 100))
        print(f"  single-feature ({n} sampled): max {d.max()/base*100:+.2f}% | "
              f"{(d<0).mean()*100:.0f}% improve on removal | {time.time()-t0:.0f}s\n")
        results.append(row)
        del pipe, ffn
        if args.device == "cuda":
            torch.cuda.empty_cache()

    json.dump(results, open("chronos_scaling.json", "w"), indent=2)
    print(f"{'model':<28}{'features':>10}{'-25%':>9}{'-50%':>9}{'-75%':>9}{'max single':>12}")
    for r in results:
        print(f"{r['model'].split('/')[-1]:<28}{r['n_features']:>10}"
              f"{r['drop_25']:>+8.1f}%{r['drop_50']:>+8.1f}%{r['drop_75']:>+8.1f}%"
              f"{r['single_max_pct']:>+11.2f}%")


if __name__ == "__main__":
    main()
