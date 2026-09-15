"""Does a test-time memory learn a different horizon on time series than on text?

The horizon result in this repo -- forget gate alpha, so memory time constant
tau = -1/ln(1-alpha) -- was measured on enwik8, i.e. on TEXT. That leaves the
central claim about a temporal mechanism resting on non-temporal data.

This script trains the identical architecture on a quantised TIME SERIES corpus
and re-measures alpha, so the horizon can be compared across data domains at
matched width, depth, context and learning rate. Corpora are quantile-binned to
256 equal-frequency bins, which pins marginal entropy at log(256) = 5.545 nats
and leaves temporal dependence as the thing that differs between them.

Two outcomes, both worth reporting:
  * alpha differs between text and time series -> the horizon a memory settles
    on depends on what it reads, and a horizon measured on text says nothing
    about a forecasting deployment.
  * alpha is the same -> the horizon is a property of the architecture and
    width, not the data, which makes the enwik8 measurement transferable.

    pip install titans-pytorch
    python experiments/titans_horizon_timeseries.py --dim 384 --steps 6000
"""
from __future__ import annotations

import argparse, json, os, shutil, ssl, time, urllib.request

import numpy as np
import torch

N_BINS = 256
ETT = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"


def _download(url: str, path: str) -> str:
    if os.path.exists(path):
        return path
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = None
    with urllib.request.urlopen(urllib.request.Request(url), context=ctx) as r, open(path, "wb") as f:
        shutil.copyfileobj(r, f)
    return path


def quantize(series, n_bins: int = N_BINS, split: float = 0.9):
    """Float series -> (train, val) token streams. Edges from the train portion
    only, so the marginal is pinned without leaking the validation split."""
    s = np.asarray(series, dtype=np.float64)
    s = s[np.isfinite(s)]
    k = int(len(s) * split)
    edges = np.quantile(s[:k], np.linspace(0, 1, n_bins + 1)[1:-1])
    return (torch.from_numpy(np.digitize(s[:k], edges)).long(),
            torch.from_numpy(np.digitize(s[k:], edges)).long())


def marginal_entropy(tok) -> float:
    c = np.bincount(tok.numpy(), minlength=N_BINS).astype(float)
    p = c / c.sum(); p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def load_ett_stream(name: str = "ETTm1"):
    """All 7 channels, quantised per-channel then concatenated."""
    raw = np.genfromtxt(_download(ETT + name + ".csv", name + ".csv"),
                        delimiter=",", skip_header=1, usecols=range(1, 8))
    tr, va = [], []
    for ch in range(raw.shape[1]):
        t, v = quantize(raw[:, ch])
        tr.append(t); va.append(v)
    return torch.cat(tr), torch.cat(va)


def load_ar(phi: float, n: int = 400_000, seed: int = 0):
    """AR(1) control: same token alphabet, temporal dependence set by phi."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return quantize(x)


def build(dim: int, depth: int = 4, seg: int = 8):
    """Same builder as the scaled-horizon run. heads=1 forces dim_head == dim,
    so the memory MLP is dim -> 4*dim -> dim."""
    from titans_pytorch import MemoryAsContextTransformer, MemoryMLP
    return MemoryAsContextTransformer(
        num_tokens=256, dim=dim, depth=depth,
        segment_len=32, neural_memory_segment_len=seg,
        num_persist_mem_tokens=4, num_longterm_mem_tokens=4,
        neural_memory_layers=(2,),
        dim_head=64 if dim >= 256 else 32, heads=6 if dim >= 256 else 2,
        neural_memory_model=MemoryMLP(dim, depth=2, expansion_factor=4.),
        neural_memory_kwargs=dict(dim_head=dim, heads=1),
        use_flex_attn=False)


def sample_batch(data, seq_len, batch_size):
    st = torch.randint(0, data.size(0) - seq_len - 1, (batch_size,))
    return torch.stack([data[s: s + seq_len + 1] for s in st])


@torch.no_grad()
def measure_alpha(model, passage, device) -> float:
    """Mean learned forget gate on a real forward pass -- same hook as
    titans_real_text.inspect_decay_gate, returning rather than printing."""
    mem = next(g[4] for g in model.layers if g[4] is not None)
    got = {}
    h = mem.to_decay_factor.register_forward_hook(
        lambda mo, i, o: got.__setitem__("a", o.sigmoid().detach()))
    model(passage.unsqueeze(0).to(device), return_cache=True)
    h.remove()
    return float(got["a"].mean())


@torch.no_grad()
def val_loss(model, data_val, seq_len, batch, device, n=20) -> float:
    model.eval()
    v = float(np.mean([model(sample_batch(data_val, seq_len, batch).to(device),
                             return_loss=True).item() for _ in range(n)]))
    model.train()
    return v


def run(name, loader, args, device):
    tr, va = loader()
    ent = marginal_entropy(tr)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = build(args.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)   # one optimizer per run
    passage = va[:args.passage]
    t0 = time.time()

    model.train()
    for step in range(args.steps):
        loss = model(sample_batch(tr, args.seq_len, args.batch).to(device), return_loss=True)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()

    a = measure_alpha(model, passage, device)
    tau = -1.0 / np.log(1.0 - a)
    row = {"corpus": name, "tokens": int(len(tr)), "marginal_entropy": ent,
           "val_loss": val_loss(model, va, args.seq_len, args.batch, device),
           "alpha": a, "tau_chunks": float(tau), "tau_positions": float(tau * args.seg),
           "pct_of_context": float(tau * args.seg / args.seq_len * 100),
           "minutes": (time.time() - t0) / 60}
    print(f"  {name:<14} entropy {ent:.3f} | val {row['val_loss']:.3f} | alpha {a:.4f} "
          f"| tau {row['tau_positions']:.1f} pos ({row['pct_of_context']:.1f}% of context) "
          f"| {row['minutes']:.0f} min", flush=True)
    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seg", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--passage", type=int, default=1024)
    ap.add_argument("--enwik8", default=None, help="path to enwik8.gz for the text comparison")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dim {args.dim} | memory {args.dim} -> {args.dim*4} -> {args.dim} "
          f"| {args.steps} steps | {device}\n")
    print("all corpora quantile-binned to 256 bins => marginal entropy pinned at "
          f"{np.log(256):.3f} nats; only temporal dependence differs\n")

    corpora = {
        "ETTm1":      lambda: load_ett_stream("ETTm1"),
        "ETTh1":      lambda: load_ett_stream("ETTh1"),
        "AR_phi0.90": lambda: load_ar(0.90),
        "AR_phi0.00": lambda: load_ar(0.00),          # white-noise control
    }
    if args.enwik8:
        import gzip
        def _enwik8():
            with gzip.open(args.enwik8) as f:
                d = np.frombuffer(f.read(int(20e6)), dtype=np.uint8).copy()
            k = int(len(d) * 0.9)
            return torch.from_numpy(d[:k]).long(), torch.from_numpy(d[k:]).long()
        corpora["enwik8 (text)"] = _enwik8

    rows = []
    for name, loader in corpora.items():
        rows.append(run(name, loader, args, device))
        json.dump(rows, open("horizon_by_corpus.json", "w"), indent=2)

    print(f"\n{'corpus':<16}{'alpha':>8}{'tau (pos)':>11}{'% context':>11}{'val loss':>10}")
    for r in rows:
        print(f"{r['corpus']:<16}{r['alpha']:>8.4f}{r['tau_positions']:>11.1f}"
              f"{r['pct_of_context']:>10.1f}%{r['val_loss']:>10.3f}")
    a = [r["alpha"] for r in rows]
    print(f"\nalpha spread across corpora: {min(a):.3f} - {max(a):.3f} "
          f"({max(a)/max(min(a),1e-9):.1f}x)")
    print("a wide spread means the horizon depends on what the memory reads;")
    print("a narrow one means it is a property of the architecture and width.")


if __name__ == "__main__":
    main()
