# Workshop paper draft — FMTS 2026

`main.tex` — 4-page submission, anonymized. Thesis: **the effective horizon of a
test-time neural memory**, measured, explained structurally, and traced to a
causal consequence.

## Build
1. Get `neurips_2026.sty` from the workshop's "Download LaTeX template" link
   (it is not on the NeurIPS media server at the usual path).
2. Overleaf is the fastest route: upload `main.tex` + the `.sty`, compile pdfLaTeX.

Body is ~1450 words + 3 tables + abstract, which should land near 3.5 pages.
Check the page count once compiled -- over-length is a desk reject.

## Where each number comes from

| Claim | Source | Status |
|---|---|---|
| alpha = 0.579 / 0.741 / 0.740 | `inspect_decay_gate()` hook, Finding 6 runs | measured |
| tau = -1/ln(1-alpha) = 0.74-1.16 chunks | arithmetic on the above | derived |
| chunk = 8 positions | `chunk_size = self.neural_memory_segment_len`, mac_transformer.py:579 | verified in source |
| gate is one scalar per head | `nn.Linear(dim, heads)` + `Rearrange('b n h -> (b h) n 1')`, neural_memory.py:514 | verified in source |
| Appendix C Eq. 32/33 diag form | arXiv:2501.00663 | verified in source |
| max single-unit ablation effect 0.13 | `titans_ablation.py` | measured |
| 0/8 vs 4/8 seeds localizing | `titans_per_unit.py` | measured, 8 seeds |
| residual floor: 0.96->0.59, 0.97->0.57 | `titans_ablation.py::check_residual` | measured |
| accumulation 3.03/2.86/4.27 | extended notebook, Test A | measured, 1 seed |

## NEEDS VERIFICATION BEFORE SUBMITTING

~~Section 4 trade-off numbers~~ **RESOLVED.** Re-run at the paper's actual
configuration (DIM=64, HIDDEN=256, 8 seeds, un-normalised pairs as in
`titans_per_unit.py::main`). Verified numbers now in the text:
n=12 uniform 0.838 / spread 0.544; n=32 0.606 / 0.398; n=64 uniform 0.000
(max|w| 1.3e-10) / spread 0.303; n=96 uniform 0.000 (max|w| 3.3e-15) /
spread 0.250. The collapse is genuine weight decay to numerical zero, not NaN --
checked.

**Section 6 accumulation numbers** are single-seed. Either re-run with 3 seeds or
keep the hedged wording already in the text.

## Anonymity checklist
- [ ] no author names, affiliations, emails
- [ ] no link to the project repo or HF checkpoint
- [ ] no mention of the wider project name or its lineage (not used in this work)
- [ ] footnote 1 stays URL-free until camera-ready
- [ ] page count within 4 (excluding references)

## Claims deliberately bounded
- Everything is stated about *the public reimplementation*, not Titans as released
  by its authors -- no official code exists, and that distinction must survive editing.
- The horizon measurement is of the *memory's* decay, not the model's total reach;
  attention and the residual stream carry information independently.
- Per-unit decay is assigned, not learned: it shows the diagonal form *permits*
  localization, not that training would find it.
- The appendix's diag form appears in a section arguing Titans generalizes Gated
  DeltaNet, so it may be presentational. The text says so.

## LLM use
If a disclosure line is wanted:
> LLM assistants were used for code implementation and manuscript editing. All
> hypotheses, experimental design, analysis, and claims are the authors'.
