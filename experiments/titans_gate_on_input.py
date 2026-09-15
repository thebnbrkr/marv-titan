"""Does the learned horizon adapt to what the memory reads, or is it fixed?

The horizon result measures alpha on a model trained on text and evaluated on text.
That leaves two separable questions:

  (a) does the horizon depend on what the memory was TRAINED on?   -- needs training
  (b) does the horizon depend on what the memory is READING now?   -- needs one forward pass

This answers (b). The forget gate is input-dependent by construction -- the module reads
the chunked sequence -- so feeding different data through the SAME trained weights shows
whether the gate responds to input at all.

If alpha is flat across text, time series and noise, the horizon is a habit settled during
training rather than a live response, and a memory trained in one domain carries its
retention horizon unchanged into another. That is a statement about deployment, and it
costs one forward pass per corpus instead of a training run.

Needs a checkpoint saved by titans_scaled_horizon_colab (titans_dim384_25000.pt).

    python experiments/titans_gate_on_input.py --ckpt titans_dim384_25000.pt --dim 384
"""
from __future__ import annotations

import argparse, gzip, json, os

import numpy as np
import torch

from titans_horizon_timeseries import (quantize, marginal_entropy, build, N_BINS,
                                       load_ett_stream, load_ar)


@torch.no_grad()
def alpha_on(model, passage, device):
    """Mean forget gate on one passage, through unchanged weights."""
    mem = next(g[4] for g in model.layers if g[4] is not None)
    got = {}
    h = mem.to_decay_factor.register_forward_hook(
        lambda mo, i, o: got.__setitem__("a", o.sigmoid().detach()))
    model(passage.unsqueeze(0).to(device), return_cache=True)
    h.remove()
    a = got["a"]
    return float(a.mean()), float(a.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--seg", type=int, default=8)
    ap.add_argument("--passage", type=int, default=1024)
    ap.add_argument("--enwik8", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build(args.dim).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print(f"loaded {args.ckpt} (dim {args.dim}, trained on text)\n")

    passages = {}
    rng = np.random.default_rng(0)
    passages["uniform noise"] = torch.from_numpy(
        rng.integers(0, 256, args.passage)).long()
    for name, phi in (("AR phi=0.00", 0.0), ("AR phi=0.90", 0.9), ("AR phi=0.99", 0.99)):
        _, va = load_ar(phi, n=200_000)
        passages[name] = va[:args.passage]
    for ds in ("ETTm1", "ETTh1"):
        try:
            _, va = load_ett_stream(ds)
            passages[ds] = va[:args.passage]
        except Exception as e:
            print(f"  {ds} skipped ({type(e).__name__})")
    if args.enwik8 and os.path.exists(args.enwik8):
        with gzip.open(args.enwik8) as f:
            d = np.frombuffer(f.read(int(2e6)), dtype=np.uint8).copy()
        passages["enwik8 (training domain)"] = torch.from_numpy(d[-args.passage:]).long()

    print(f"{'passage':<28}{'entropy':>9}{'alpha':>9}{'sd':>8}{'tau (pos)':>11}")
    rows = []
    for name, p in passages.items():
        a, sd = alpha_on(model, p, device)
        tau = -1.0 / np.log(1.0 - a) * args.seg
        rows.append({"passage": name, "alpha": a, "alpha_sd": sd, "tau_positions": tau})
        print(f"{name:<28}{marginal_entropy(p):>9.3f}{a:>9.4f}{sd:>8.4f}{tau:>11.1f}")

    json.dump(rows, open("gate_on_input.json", "w"), indent=2)
    a = [r["alpha"] for r in rows]
    spread = max(a) / max(min(a), 1e-9)
    print(f"\nalpha across input types: {min(a):.4f} - {max(a):.4f}  ({spread:.2f}x)")
    print(f"the WIDTH effect, for comparison: 0.227 - 0.741  (3.3x)")
    print()
    if spread < 1.3:
        print("-> the gate barely moves with input. The horizon is a habit fixed during")
        print("   training, not a live response, so a memory trained in one domain carries")
        print("   its retention horizon unchanged into another.")
    else:
        print("-> the gate DOES respond to input. The horizon is partly set by what the")
        print("   memory is currently reading, not only by training.")


if __name__ == "__main__":
    main()
