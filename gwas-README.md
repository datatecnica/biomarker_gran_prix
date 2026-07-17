# gwas.py — association engine

YAML-driven batch association scan: one config → many runs → one CSV per run×stratum.

```bash
python3 gwas.py --config batch.yaml [--run name1 name2 ...] [--cwd DIR]
```

For each run×stratum: clinical load → global prefilter → `derive:` columns → per-stratum
subset → align to `.psam` → fit the null once (CPU) → stream predictor blocks to the
device → per-predictor β/SE/P + effect-allele/EAF/N → QC + genome-wide sig + λ-GC →
`results/<run>-<stratum>-<timestamp>.csv`.

## YAML schema

### `defaults` (every key has a sensible default; a config can be as short as
`clinical_file` + `genotype_prefix` + one run)

| Key | Default | Meaning |
|---|---|---|
| `clinical_file` | `*.tsv` | LONG clinical table; latest file matching the glob is used |
| `genotype_prefix` | — | plink2 `<prefix>.pgen/.pvar/.psam`; required for genetic mode |
| `sample_id_col` | `IID` | column linking clinical rows to `.psam` |
| `time_col` | — | longitudinal axis (required by `gallop`/`slope`/`trajgwas`) |
| `output_dir` | `results` | where CSVs are written |
| `prefilter` | none | list of pandas-query expressions applied once to all rows |
| `covariates` | `[]` | explicit covariate columns (intercept-only if empty) |
| `derive` | none | computed columns (see below) |
| `strata_col` / `strata` | none → pooled | per-stratum analysis; EAF/MAF/N recomputed per stratum |
| `maf_min`/`mac_min`/`missing_max` | `0.01`/`20`/`0.05` | per-variant QC (genetic) |
| `min_n` | `30` | minimum N per fit; below it the stratum is skipped |
| `predictor_source` | `genetic` | `genetic` (SNPs) or `tabular` (clinical columns) |
| `zscore_predictors`/`zscore_binary`/`zscore_dosage` | `true`/`false`/`false` | non-genetic predictor standardization (proteomics parity) |
| `spa_p_cutoff`/`spa_maf_cutoff` | `0.01`/`0.05` | SPA is applied where score-P < cutoff OR MAF ≤ cutoff |
| `device` | `auto` | `auto`/`cuda`/`cpu` |
| `max_cpus`/`max_ram_gb`/`max_gpu_mem_gb` | `null` | resource caps (null → whole machine) |
| `block_size` | `auto` | SNPs per streamed block (auto-sized to the VRAM/RAM budget) |
| `prefetch` | `true` | overlap pgen I/O with compute |

### `runs` (list; per-run keys override `defaults`)

| Key | Applies to | Meaning |
|---|---|---|
| `name` | all | run name (used in the output filename) |
| `model` | all | `linear`/`logistic`/`cox`/`slope`/`gallop`/`trajgwas` |
| `outcome` | all but cox | the response column |
| `duration_col` / `event_col` | cox | time-to-event + event indicator |
| `sample_filter` | all | pandas-query to pick rows (e.g. `"visit == 'BL'"` for cross-sectional) |
| `predictor_source` | all | `genetic` (default) or `tabular` |
| `predictors` | tabular | `[list]` or `{prefix\|suffix\|regex: ...}` |
| `condition_snps` | genetic | list of variant IDs folded into the null (conditional analysis) |
| `report_term` | gallop/trajgwas | gallop: `SNP`\|`SNPxTIME`; trajgwas: `MEAN`\|`VAR`\|`JOINT` |
| `spa` | logistic/cox/trajgwas | enable SPA tail correction (default true) |
| `add_covariates` / `drop_covariates` | all | adjust the covariate list |
| `strata` | all | override strata; `[null]` forces a pooled run |
| `min_n` | all | per-run N floor |

### `derive:` forms
- **row-wise** — `logLEDD: "log(LEDD + 1)"` (numpy funcs: log, log1p, exp, sqrt, abs, …)
- **per-subject aggregate** — `baseline_of(col)`, `slope_of(col ~ time)`, `delta(col)` (computed within `sample_id_col`, broadcast to all visits)
- **genotype pull** — `G_rs123: "snp(rs123)"` (materializes a variant's dosage as a column; usable as a covariate/outcome — using it in `add_covariates` reproduces `condition_snps`)

## Models

| `model` | Scan | Robustness |
|---|---|---|
| `linear` | closed-form β/SE/P (residualize → gᵀy, gᵀg) | collinearity guard; QC flags |
| `logistic` | batched score test | SPA tail; Firth refit on genome-wide hits |
| `cox` | null-Cox martingale residuals → linear score | SPACox tail; lifelines 3-tier refit on hits |
| `slope` | per-subject slope → unweighted GWAS on slopes | robust cross-check |
| `gallop` | one-stage RI+slope LMM null → batched 2×2 GLS (SNP + SNP×time) | falls back to `slope` on LMM failure |
| `trajgwas` | LMM null → mean (=gallop) + variance-residual τ + 2-df joint | SPA on the variance tail |

## Output columns (stable schema, PLAN §6)

`predictor, CHR, POS, effect_allele, other_allele, EAF, MAF, missing_rate,
predictor_mean, predictor_sd, zscored, N, n_obs, n_subjects, n_events, beta, SE, P,`
model-specific terms (`beta_SNP…`, `beta_SNPxTIME…`, `beta_mean…`, `tau_var…`, `P_joint`),
`model_type, test (wald|score|spa|firth|full), genomewide_sig, bonferroni_threshold,
significant, conditioned_on, strata`.

For **tabular** predictors the SNP-only fields (CHR/POS/alleles/EAF/MAF/missing) are null
and `predictor` + the standardization columns carry the identity — so downstream stays
mode-agnostic.

## Notes
- Genetic dosages are used **raw** (β per allele); non-genetic predictors are **z-scored**
  (continuous only) by default.
- A missing referenced column is a **fatal, loud** error (sensible defaults never mask a typo).
- Stage `.pgen` on ext4, not `/mnt/c` — reads through the WSL bridge are ~5× slower.
