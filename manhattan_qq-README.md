# manhattan_qq.py — Manhattan / volcano + QQ plots

Plots **any** result source — a per-stratum study CSV (uses `P`) or a meta CSV (uses
`P_FE`) — one figure per source, auto-detecting genetic vs non-genetic.

```bash
python3 manhattan_qq.py --results-dir results/ --meta-dir meta/ --output-dir plots/
python3 manhattan_qq.py --input results/linear_bmkX-ALL-<ts>.csv     # a single source
python3 manhattan_qq.py --input meta/META_linear_bmkX_strat-<ts>.csv  # a meta source
```

## Panels
- **Manhattan** (genetic: CHR/POS present) — −log₁₀P vs genomic position. Chromosomes
  alternate **grey / black**; genome-wide hits (P<5e-8) alternate **burgundy / mint** per
  chromosome so they pop while keeping the chromosome banding. Genome-wide (5e-8) and
  per-run Bonferroni lines are drawn.
- **Volcano** (non-genetic / tabular: CHR/POS null) — −log₁₀P vs effect size (β); the
  significant predictors are highlighted (burgundy) and labelled (`adjustText` if
  installed), with the Bonferroni line.
- **QQ** (on every figure, Manhattan *and* volcano) — observed vs expected −log₁₀P, with a
  line and λ-GC **per MAF band** (all / common MAF≥0.05 / low-freq 0.01–0.05 / rare
  MAF<0.01) as well as across all variants. Tabular sources (no MAF) show the `all` band.

## Both individual studies *and* meta
In auto mode, each per-stratum study CSV **and** each `META_*` CSV becomes its own
`<run>-<stratum|META>-manqq.png`. The primary P column is auto-selected (`P` for studies,
`P_FE` for meta), so Manhattan/volcano + QQ work uniformly for both.

## Notes
- MAF is read from `MAF` (study CSVs) or derived from `EAF` (meta CSVs) as `min(EAF, 1−EAF)`.
- The MAF-stratified λ makes rare/low-freq inflation obvious at a glance — the low-freq
  band commonly sits above the common band (residual structure / score-test tail).
- Headless (`Agg`) backend; PNGs at 130 dpi. `adjustText` is optional (nicer volcano labels).
