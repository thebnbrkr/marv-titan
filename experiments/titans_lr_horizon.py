"""Horizon AND learning rate of a Titans test-time memory across corpora.

Extends titans_horizon_timeseries.py. At fixed width (dim 384) it trains the same
MemoryAsContextTransformer on each corpus and measures BOTH:
  * the forget gate  alpha  (chunk-level)  -> horizon  tau = -1/ln(1-alpha)
  * the token-level learning rate  theta = adaptive_step_transform(to_adaptive_step(seq))

For each it reports mean / std / coefficient-of-variation, so we can separate the
CROSS-corpus change in learning rate from the WITHIN-sequence spread that
distinguishes the adaptive learning rate (theta, token-level) from the
near-frozen gate (alpha, chunk-level). All corpora are quantile-binned to 256
bins => marginal entropy pinned at log(256)=5.545 nats; only temporal structure differs.

New corpus: StarLightCurves (UCR 2018), a real ~9.46M-point periodic sensor stream.

    pip install titans-pytorch datasets scipy pandas
    python titans_lr_horizon.py --dim 384 --steps 6000 --ucr-root /opt/data \
        --s3 s3://<your-bucket>
"""
from __future__ import annotations
import argparse, json, os, time, subprocess
import numpy as np, torch

N_BINS = 256
UNIFORM = float(np.log(N_BINS))


def quantize(series, n_bins=N_BINS, split=0.9):
    """Raw float series -> (train, val) token streams; edges from train only."""
    s = np.asarray(series, dtype=np.float64); s = s[np.isfinite(s)]
    k = int(len(s) * split)
    edges = np.quantile(s[:k], np.linspace(0, 1, n_bins + 1)[1:-1])
    return (torch.from_numpy(np.digitize(s[:k], edges)).long(),
            torch.from_numpy(np.digitize(s[k:], edges)).long())


def marginal_entropy(tok):
    c = np.bincount(tok.numpy(), minlength=N_BINS).astype(float)
    p = c / c.sum(); p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def lag1_mi(tok, coarse=16):
    x = (tok.numpy().astype(int) * coarse) // N_BINS
    J = np.histogram2d(x[:-1], x[1:], bins=[coarse, coarse])[0]; J = J / J.sum()
    px, py = J.sum(1, keepdims=True), J.sum(0, keepdims=True); m = J > 0
    return float((J[m] * np.log(J[m] / (px @ py)[m])).sum())


# ---------------- corpora (return RAW float arrays; quantize centrally) ----------------
def ar1_big(n, phi, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    if phi == 0.0:
        return e
    try:
        from scipy.signal import lfilter
        return lfilter([1.0], [1.0, -phi], e)
    except ImportError:
        x = np.zeros(n)
        for t in range(1, n): x[t] = phi * x[t - 1] + e[t]
        return x


def stream_chronos(config, target_tokens, repo="autogluon/chronos_datasets"):
    from datasets import load_dataset
    ds = load_dataset(repo, config, split="train", streaming=True)
    parts, total, col = [], 0, None
    for row in ds:
        if col is None:
            best, bl = None, 0
            for k, v in row.items():
                if isinstance(v, (list, np.ndarray)) and len(v) > bl:
                    try: float(v[0]); best, bl = k, len(v)
                    except (TypeError, ValueError, IndexError): pass
            col = best
            if col is None: raise ValueError(f"no numeric sequence column in {config}")
        s = np.asarray(row[col], dtype=np.float64); s = s[np.isfinite(s)]
        if len(s) < 200: continue
        parts.append(s); total += len(s)
        if total >= target_tokens: break
    return np.concatenate(parts)


def load_ucr(root, name="StarLightCurves"):
    """Concatenate every UCR series (train+test) into one long stream. Each row is
    label<TAB>v1<TAB>... ; drop the label column. Series are z-normalised already."""
    def rd(p):
        try:
            import pandas as pd
            return pd.read_csv(p, sep="\t", header=None).values[:, 1:]
        except Exception:
            return np.genfromtxt(p, delimiter="\t")[:, 1:]
    paths = []
    for suf in ("_TRAIN.tsv", "_TEST.tsv"):
        for cand in (os.path.join(root, name, name + suf), os.path.join(root, name + suf)):
            if os.path.exists(cand): paths.append(cand); break
    if not paths: raise FileNotFoundError(f"{name} TSVs not under {root}")
    return np.concatenate([rd(p).reshape(-1) for p in paths])


# ---------------- model (identical builder to titans_horizon_timeseries) ----------------
def build(dim, depth=4, seg=8):
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
def val_loss(model, data_val, seq_len, batch, device, n=20):
    model.eval()
    v = float(np.mean([model(sample_batch(data_val, seq_len, batch).to(device),
                             return_loss=True).item() for _ in range(n)]))
    model.train()
    return v


@torch.no_grad()
def collect_memory_signals(model, passage, device):
    """One forward pass; capture the forget gate alpha (post-sigmoid, chunk-level)
    and the token-level learning rate theta = adaptive_step_transform(to_adaptive_step)."""
    mem = next(g[4] for g in model.layers if g[4] is not None)
    buf = {"alpha": [], "theta": []}
    h1 = mem.to_decay_factor.register_forward_hook(
        lambda m, i, o: buf["alpha"].append(o.sigmoid().detach().float().reshape(-1).cpu()))
    h2 = mem.to_adaptive_step.register_forward_hook(
        lambda m, i, o: buf["theta"].append(mem.adaptive_step_transform(o).detach().float().reshape(-1).cpu()))
    model(passage.unsqueeze(0).to(device), return_cache=True)
    h1.remove(); h2.remove()
    return torch.cat(buf["alpha"]).numpy(), torch.cat(buf["theta"]).numpy()


def run(name, tr, va, args, device):
    ent = marginal_entropy(tr); mi = lag1_mi(tr)
    tok_per_param = len(tr) / sum(p.numel() for p in build(args.dim).parameters())
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = build(args.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    passage = va[:args.passage]
    t0 = time.time(); aborted = False

    model.train()
    trace = []; max_gap = -9.9
    for step in range(args.steps):
        loss = model(sample_batch(tr, args.seq_len, args.batch).to(device), return_loss=True)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        if step > 0 and step % 500 == 0:          # dense overfit checks (every 500 steps)
            v = val_loss(model, va, args.seq_len, args.batch, device, n=10)
            gap = v - loss.item(); max_gap = max(max_gap, gap)
            trace.append({"step": step, "train": round(loss.item(), 3), "val": round(v, 3), "gap": round(gap, 3)})
            print(f"    {name} step {step:>5} train {loss.item():.3f} val {v:.3f} gap {gap:+.3f}", flush=True)
            if gap > 1.0:                          # memorisation signature -> abort
                print(f"    !! ABORT {name}: gap {gap:.2f} -> memorising", flush=True)
                aborted = True; break

    if aborted:
        row = {"corpus": name, "status": "aborted_overfit", "dim": args.dim, "train_tokens": int(len(tr)),
               "tokens_per_param": tok_per_param, "marginal_entropy": ent, "lag1_mi": mi,
               "max_gap": round(max_gap, 3), "gap_trace": trace}
    else:
        alpha, theta = collect_memory_signals(model, passage, device)
        a = float(alpha.mean()); tau = -1.0 / np.log(1.0 - a)
        vloss = val_loss(model, va, args.seq_len, args.batch, device)
        row = {"corpus": name, "status": "ok", "dim": args.dim, "train_tokens": int(len(tr)),
               "tokens_per_param": tok_per_param, "marginal_entropy": ent, "lag1_mi": mi,
               "val_loss": vloss,
               "alpha_mean": a, "alpha_std": float(alpha.std()),
               "alpha_cv": float(alpha.std() / (abs(a) + 1e-12)),
               "theta_mean": float(theta.mean()), "theta_std": float(theta.std()),
               "theta_cv": float(theta.std() / (abs(theta.mean()) + 1e-12)),
               "tau_chunks": float(tau), "tau_positions": float(tau * args.seg),
               "pct_of_context": float(tau * args.seg / args.seq_len * 100),
               "max_gap": round(max_gap, 3), "overfit_warn": bool(max_gap > 0.5), "gap_trace": trace,
               "minutes": (time.time() - t0) / 60}
        warn = "  !! OVERFIT_WARN (max_gap %.2f)" % max_gap if row["overfit_warn"] else ""
        print(f"  -> {name}: alpha {a:.4f} (cv {row['alpha_cv']:.3f}) | "
              f"theta {row['theta_mean']:.4g} (cv {row['theta_cv']:.3f}) | "
              f"tau {row['tau_positions']:.1f} pos | val {row['val_loss']:.3f} | "
              f"max_gap {max_gap:+.2f} | {row['minutes']:.1f} min{warn}", flush=True)
    del model, opt
    if device == "cuda": torch.cuda.empty_cache()
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
    ap.add_argument("--token-target", type=int, default=20_000_000)
    ap.add_argument("--ucr-root", default="/opt/data")
    ap.add_argument("--s3", default=None, help="s3:// prefix for incremental result upload")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dim {args.dim} | {args.steps} steps | seq {args.seq_len} | {device}", flush=True)
    T = args.token_target
    corpora = {
        "AR_phi0.00":       lambda: ar1_big(T, 0.00, args.seed),   # white-noise control
        "AR_phi0.90":       lambda: ar1_big(T, 0.90, args.seed),   # predictable synthetic
        "electricity_15min":lambda: stream_chronos("electricity_15min", T),
        "StarLightCurves":  lambda: load_ucr(args.ucr_root, "StarLightCurves"),
    }
    rows = []
    for name, loader in corpora.items():
        try:
            tr, va = quantize(loader())
        except Exception as e:
            rows.append({"corpus": name, "status": "load_fail", "error": str(e)[:300]})
            print(f"  load_fail {name}: {e}", flush=True)
        else:
            print(f"[{name}] train {len(tr):,} tok | entropy {marginal_entropy(tr):.3f} | "
                  f"lag1_mi {lag1_mi(tr):.3f}", flush=True)
            rows.append(run(name, tr, va, args, device))
        json.dump(rows, open("/tmp/lr_results.json", "w"), indent=2)
        if args.s3:
            subprocess.run(["aws", "s3", "cp", "/tmp/lr_results.json", f"{args.s3}/_RESULT_LR",
                            "--region", os.environ.get("AWS_REGION", "us-east-1")], check=False)

    print(f"\n{'corpus':<18}{'alpha':>8}{'a_cv':>7}{'theta':>10}{'th_cv':>7}"
          f"{'tau(pos)':>10}{'val':>8}")
    ok = [r for r in rows if r.get("status") == "ok"]
    for r in ok:
        print(f"{r['corpus']:<18}{r['alpha_mean']:>8.4f}{r['alpha_cv']:>7.3f}"
              f"{r['theta_mean']:>10.4g}{r['theta_cv']:>7.3f}{r['tau_positions']:>10.1f}"
              f"{r['val_loss']:>8.3f}")
    if ok:
        th = [r["theta_mean"] for r in ok]; al = [r["alpha_mean"] for r in ok]
        print(f"\nLEARNING RATE spread across corpora: {min(th):.4g} - {max(th):.4g} "
              f"({max(th)/max(min(th),1e-12):.1f}x)")
        print(f"forget-gate spread across corpora:   {min(al):.4f} - {max(al):.4f} "
              f"({max(al)/max(min(al),1e-12):.1f}x)")
        acv = float(np.mean([r["alpha_cv"] for r in ok]))
        tcv = float(np.mean([r["theta_cv"] for r in ok]))
        print(f"\nwithin-sequence variability (mean CV): theta {tcv:.3f} vs alpha {acv:.3f} "
              f"-> learning rate varies {tcv/max(acv,1e-9):.1f}x more token-to-token than the gate")


if __name__ == "__main__":
    main()
