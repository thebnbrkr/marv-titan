# Workshop paper draft

`main.tex` — 4-page submission draft, anonymized for double-blind review.

## To build

1. Download `neurips_2026.sty` from the workshop's "Download LaTeX template" link
   and place it beside `main.tex`. (It is not on the NeurIPS media server at the
   usual path; get it from the workshop site.)
2. Easiest path is Overleaf: upload `main.tex` + the `.sty`, compile with pdfLaTeX.

## Anonymity checklist — verify before submitting

- [ ] No author names, affiliations, emails
- [ ] No link to the project repo
- [ ] No link to the HuggingFace checkpoint
- [ ] No mention of the wider project name or its lineage (not used in this work)
- [ ] Footnote 1 names the third-party implementation *without* pointing at
      anything under your account — citing someone else's public repo is fine,
      linking yours is not
- [ ] Over-length or improperly anonymized => desk reject, so check page count
      after the results land

## What still needs filling

- **Table 2** (`tab:temporal`): run `notebooks/marv_titans_temporal_colab.ipynb`
  on a T4. Lag-1 MI values are already computed and filled in; val loss and
  write-norm ratio come out of the sweep.
- **Section 6** currently says "[Results pending]" — replace with the actual
  direction once the sweep finishes.
- Optionally add `norm_ratio_vs_predictability.png` from the notebook as a figure.

## Claims deliberately bounded

- Everything is stated about *the public reimplementation*, not about Titans as
  released by its authors — no official code exists, so that distinction must
  survive editing.
- The per-unit decay result uses directly assigned rates, not a learned gate; it
  shows per-unit decay *permits* localization, not that training would find it.

## LLM use

If a disclosure line is wanted, the conventional form is:

> LLM assistants were used for code implementation and manuscript editing. All
> hypotheses, experimental design, analysis, and claims are the authors'.

Routine editing and basic code assistance do not require disclosure under the
NeurIPS 2026 policy; substantive, original use does.
