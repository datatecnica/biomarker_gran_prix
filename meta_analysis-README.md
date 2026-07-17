# meta_analysis.py — IVW-FE + DL-RE meta-analysis

Pools per-stratum association results across studies, joined **on the predictor**.

```bash
# auto-discover: combine each run's strata from results/, write META_<run>-<ts>.csv
python3 meta_analysis.py --results-dir results/ --output-dir meta/ [--run name]

# explicit: combine specific files
python3 meta_analysis.py --inputs X-EUR-*.csv X-AJ-*.csv X-SAS-*.csv --output META_X.csv
```

## What it does
- **Discovery** — groups `<run>-<stratum>-<ts>.csv` files by run, keeping the latest
  timestamp per (run, stratum). `META_*` files are skipped.
- **Join** — outer-join each run's strata on `predictor` (SNP id, or analyte name for
  non-genetic runs). **Any number of studies** (2, 3, …) is supported.
- **Allele harmonization** — the effect allele is aligned to a reference (the first
  stratum with a non-null allele); a stratum whose `effect_allele` differs has its β sign
  and EAF flipped before pooling. A no-op when strata share one `.pgen`, but the check
  stays for multi-dataset meta.
- **Pooling** — inverse-variance **fixed-effects** (IVW-FE) and **DerSimonian–Laird
  random-effects** (DL-RE); 1 study → pass-through. Heterogeneity: Cochran's Q, P_het,
  I², τ², and a per-study `direction` string (`+`/`-`/`?`).
- **Strand check** — flags **palindromic** (A/T, C/G) or allele-**mismatched** variants
  that can't be resolved: `strand_flag = (palindromic | mismatch) & (EAF∈[0.40,0.60] |
  conflicting cross-study EAF)`. Emits `palindromic`, `allele_mismatch`, `strand_flag`.
- **Non-genetic** results are handled transparently: the join is on the analyte name,
  β/SE pool identically, and the allele/strand logic no-ops (no alleles).

## Output (`META_<run>-<ts>.csv`)
Key + carried identity (`CHR`, `POS`, `effect_allele`, `other_allele`, `EAF`), strand
flags, `n_studies`, `N_total`, `direction`, `beta_FE/SE_FE/Z_FE/P_FE`,
`beta_RE/SE_RE/Z_RE/P_RE`, `Q/P_het/I2/tau2`, per-stratum `beta_<s>/SE_<s>/P_<s>/N_<s>/EAF_<s>`,
`bonferroni_threshold`, `significant_FE/RE`, `genomewide_FE/RE`. λ_FE and λ_RE are logged
per run.

## Notes
- **DL-RE is conservative at small #studies** (k=2 often deflates); FE is the primary
  read for a handful of ancestries.
- EAF in the output is the harmonized cross-study mean; per-stratum EAFs are retained.
- Downstream (`manhattan_qq.py`, `build_results_summary.py`) auto-discovers these CSVs.
