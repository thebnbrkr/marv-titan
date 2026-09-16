# The Learning Rate, Not the Forget Gate: Locating Test-Time Adaptation in a Titans Memory

This repository is the implementation accompanying the paper. A Titans memory
exposes two learned controls as it reads, a forget gate `alpha_t` and a
learning rate `theta_t`. The code here measures both, and shows that only the
learning rate adapts at test time while the gate is a retention horizon fixed
by the training corpus.

The repository also contains MARV, a weight-space interpretability toolkit that
predates this paper (see `docs/MARV.md`). None of the paper's results depend on it.

## Requirements

```bash
pip install -r requirements.txt
pip install titans-pytorch
```

No dataset download is needed. The AR processes are generated deterministically
from a seed; ETTm1 downloads from the public ETDataset repository at runtime;
enwik8 is read from `data/enwik8.gz` inside `titans-pytorch`; the UCR series are
fetched from the UCR Time Series Archive.

Training needs a GPU. Every measurement is a single forward pass and runs on CPU.

## Training

Each corpus is trained for 6k steps at sequence length 256, batch size 8, with
Adam at learning rate 2e-4. One run takes roughly 20-30 minutes on an A100.

One invocation trains and measures every corpus in turn (ETTm1, ETTh1, and the
two AR processes), writing `horizon_by_corpus.json`:

```bash
python experiments/titans_lr_horizon.py --dim 384
```

Add the text corpus by pointing at the copy shipped inside `titans-pytorch`:

```bash
python experiments/titans_horizon_timeseries.py --dim 384 \
    --enwik8 $(python -c "import titans_pytorch,os;print(os.path.dirname(titans_pytorch.__file__))")/../data/enwik8.gz
```

Use `--dim 64` or `--dim 512` for the width sweep of Appendix A. Other flags:
`--steps`, `--seq-len`, `--batch`, `--lr`, `--seg`, `--seed`, `--passage`.

## Evaluation

Both controls are read by hooking their module during an ordinary forward pass
over a held-out passage. No training or fine-tuning is involved.

The corpus run above reports both controls itself and writes
`horizon_by_corpus.json`. The remaining measurements:

```bash
python experiments/titans_gate_on_input.py --ckpt <checkpoint>   # single-sequence gate probe
python experiments/titans_per_unit.py                            # erasure + per-unit localization
python experiments/titans_ablation.py                            # zeroed-memory control
```

## Results

Both learned controls at fixed width (dim = 384), varying only the training
corpus. Mean over five seeds, three for the two UCR streams.

| Corpus | forget gate a | horizon t (pos.) | learning rate th | cv(th) |
|---|---|---|---|---|
| AR(1), phi = 0.00 (noise) | 0.16 | n/a | 0.997 | 0.00 |
| AR(1), phi = 0.90 | 0.423 | 15 | 0.581 | 0.41 |
| electricity, 15-min (ETTm1) | 0.030 | 262 | 0.090 | 1.50 |
| StarLightCurves (UCR) | 0.66 | 7 | 0.65 | 0.32 |
| UWaveGestureLibrary (UCR) | 0.52 | 11 | 0.25 | 0.62 |

The corpus shifts the horizon more than twentyfold; model width moves it only
threefold. Within a sequence the learning rate's coefficient of variation
averages 1.7x the gate's.

See [REPRODUCE.md](REPRODUCE.md) for the script behind every other number in
the paper, including the erasure threshold, the per-unit localization result and
the residual-attribution control.

## Tests

```bash
python -m pytest -q
```

## Licence

MIT. Contributions are welcome by pull request; please run the test suite first.
