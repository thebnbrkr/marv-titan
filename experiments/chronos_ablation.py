"""Does any single FFN feature matter in a deployed time-series foundation model?

The Titans experiments in this directory ask where a *test-time* memory stores
things, and find that no single hidden unit owns any stored association. This
script asks the same question of a *frozen, deployed* forecaster -- Chronos-Bolt
(Ansari et al., 2024) -- using a real forecasting metric instead of a
reconstruction cosine.

Method. Chronos-Bolt is a T5 encoder-decoder that patches the context and
regresses 9 quantiles directly; it has NO token vocabulary (vocab_size=2), so a
logit lens does not apply. What does apply is ablation: zero one FFN feature j
(row j of `wi`, column j of `wo`), re-forecast, and measure the change in
Weighted Quantile Loss. That is the same causal move as titans_ablation.py, with
WQL where recall-cosine was.

Positive control matters here. Zeroing ALL FFN features must break the model; if
it does not, the ablation is a no-op and every other number is meaningless.

Results (chronos-bolt-tiny, 9M params, 8 layers x 1024 features = 8192):

  baseline WQL 0.9390 on ETTh1 (56 windows, context 512, horizon 64)
  naive last-value baseline 2.0547  -> the model is really forecasting

  single feature ablated   max  +0.20% WQL   (192 sampled)
  single LAYER ablated     -1.4% .. +1.6%    (all 8)
  half the features gone   +0.3% (ETTh1), +10.9% (ETTm1), -2.4% (ETTh2)
  ALL FFN gone             +2312% (ETTh1), +6469% (ETTm1), +423% (ETTh2)

Read: the FFN is collectively essential and individually near-redundant. No
single feature carries a measurable share of forecasting accuracy, which is the
same distributed-storage picture the Titans ablation finds -- reproduced on a
frozen forecaster with a forecasting metric.

    pip install chronos-forecasting "transformers>=4.40,<4.48"
    python experiments/chronos_ablation.py --mode all
"""
from __future__ import annotations

import argparse, os, shutil, ssl, urllib.request

import numpy as np
import torch

LEVELS = np.arange(0.1, 1.0, 0.1)
ETT = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"


def _download(url: str, path: str) -> str:
    if os.path.exists(path):
        return path
    try:
        ctx = None
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = None
    with urllib.request.urlopen(urllib.request.Request(url), context=ctx) as r, open(path, "wb") as f:
        shutil.copyfileobj(r, f)
    return path


def load_ett(name: str) -> np.ndarray:
    """One ETT CSV -> (timesteps, 7 channels)."""
    p = _download(ETT + name + ".csv", name + ".csv")
    return np.genfromtxt(p, delimiter=",", skip_header=1, usecols=range(1, 8))


def make_windows(arr: np.ndarray, context: int, horizon: int, n_win: int = 8):
    """Non-overlapping evaluation windows, every channel of each."""
    ins, tgt = [], []
    for w in range(n_win):
        s = w * context
        if s + context + horizon > len(arr):
            break
        for c in range(arr.shape[1]):
            ins.append(torch.tensor(arr[s:s + context, c], dtype=torch.float32))
            tgt.append(arr[s + context:s + context + horizon, c])
    return ins, np.stack(tgt)


def wql(quantile_forecast: np.ndarray, y: np.ndarray) -> float:
    """Weighted Quantile Loss -- the metric Chronos reports."""
    d = y[:, :, None] - quantile_forecast
    return float(2 * np.maximum(LEVELS * d, (LEVELS - 1) * d).sum() / np.abs(y).sum())


class Harness:
    def __init__(self, model_id="amazon/chronos-bolt-tiny", device="cpu"):
        from chronos import BaseChronosPipeline
        self.pipe = BaseChronosPipeline.from_pretrained(
            model_id, device_map=device, torch_dtype=torch.float32)
        self.model = self.pipe.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        mods = dict(self.model.named_modules())
        # encoder FFN sits at layer.1, decoder FFN at layer.2
        self.ffn = [mods[f"{s}.block.{b}.layer.{1 if s == 'encoder' else 2}.DenseReluDense"]
                    for s in ("encoder", "decoder") for b in range(4)]
        self.n_feat = self.ffn[0].wi.weight.shape[0]
        self.names = [f"{s}{b}" for s in ("encoder", "decoder") for b in range(4)]

    @torch.no_grad()
    def score(self, ins, tgt, horizon) -> float:
        q, _ = self.pipe.predict_quantiles(inputs=ins, prediction_length=horizon,
                                           quantile_levels=list(LEVELS))
        return wql(q.detach().numpy(), tgt)

    @torch.no_grad()
    def ablate_features(self, picks, ins, tgt, horizon) -> float:
        """picks: iterable of flat indices into (layer, feature). Restores after."""
        saved = []
        for p in picks:
            L, j = int(p) // self.n_feat, int(p) % self.n_feat
            mod = self.ffn[L]
            saved.append((mod, j, mod.wi.weight[j].clone(), mod.wo.weight[:, j].clone()))
            mod.wi.weight[j] = 0
            mod.wo.weight[:, j] = 0
        s = self.score(ins, tgt, horizon)
        for mod, j, row, col in saved:
            mod.wi.weight[j] = row
            mod.wo.weight[:, j] = col
        return s

    @torch.no_grad()
    def ablate_layers(self, layer_idx, ins, tgt, horizon) -> float:
        saved = [(self.ffn[i], self.ffn[i].wi.weight.clone(), self.ffn[i].wo.weight.clone())
                 for i in layer_idx]
        for i in layer_idx:
            self.ffn[i].wi.weight.zero_()
            self.ffn[i].wo.weight.zero_()
        s = self.score(ins, tgt, horizon)
        for mod, wi, wo in saved:
            mod.wi.weight.copy_(wi)
            mod.wo.weight.copy_(wo)
        return s


def naive_wql(ins, tgt, horizon) -> float:
    """Repeat the last observed value at every quantile -- the sanity floor."""
    return wql(np.stack([np.full((horizon, len(LEVELS)), x[-1].item()) for x in ins]), tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="amazon/chronos-bolt-tiny")
    ap.add_argument("--mode", default="all", choices=["single", "group", "all"])
    ap.add_argument("--n-single", type=int, default=64, help="features sampled per layer")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    h = Harness(args.model)
    total = len(h.ffn) * h.n_feat
    print(f"{args.model}: {len(h.ffn)} FFN layers x {h.n_feat} features = {total}")

    data = {n: load_ett(n) for n in ("ETTh1", "ETTm1", "ETTh2")}
    C, H = 512, 64

    for name, arr in data.items():
        ins, tgt = make_windows(arr, C, H)
        base = h.score(ins, tgt, H)
        print(f"\n=== {name} ===  {len(ins)} windows | baseline WQL {base:.4f} "
              f"| naive {naive_wql(ins, tgt, H):.4f}")

        # positive control FIRST -- if this does not break, nothing below means anything
        allz = h.ablate_layers(range(len(h.ffn)), ins, tgt, H)
        print(f"  CONTROL all FFN zeroed : WQL {allz:9.4f}  ({(allz-base)/base*100:+.0f}%)")
        if allz < base * 1.5:
            print("  !! control did not degrade -- ablation is a no-op, stop here")
            continue

        if args.mode in ("group", "all"):
            for frac in (0.25, 0.5, 0.75):
                k = int(total * frac)
                s = np.mean([h.ablate_features(
                    np.random.default_rng(sd).choice(total, k, replace=False), ins, tgt, H)
                    for sd in range(args.seeds)])
                print(f"  {frac:>4.0%} of features removed: WQL {s:9.4f}  ({(s-base)/base*100:+.1f}%)")

        if args.mode in ("single", "all") and name == "ETTh1":
            rng = np.random.default_rng(0)
            worst = []
            for L, lname in enumerate(h.names):
                for j in rng.choice(h.n_feat, args.n_single, replace=False):
                    d = h.ablate_features([L * h.n_feat + int(j)], ins, tgt, H) - base
                    worst.append((d, lname, int(j)))
            worst.sort(reverse=True)
            arr_d = np.array([w[0] for w in worst])
            print(f"  single-feature ablation ({len(worst)} sampled): "
                  f"max {arr_d.max():+.5f} ({arr_d.max()/base*100:+.2f}%)  sd {arr_d.std():.5f}")
            for d, ln, j in worst[:3]:
                print(f"     worst: {ln} f{j}  {d:+.5f}  ({d/base*100:+.2f}%)")


if __name__ == "__main__":
    main()
