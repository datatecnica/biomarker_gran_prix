# Biomarker Gran Prix — Design Plan

A GPU-accelerated, YAML-driven batch **association engine** for biomarker studies.
It is a companion to [`../proteomics_data_mine`](../proteomics_data_mine) and reuses
that project's architecture (one config → many runs → one CSV per run×stratum →
meta-analysis → plots → summary), generalized on two axes:

- **Predictor axis** — the looped predictor can be **SNPs streamed from a plink2
  dosage fileset** (full GWAS: 10⁵–10⁷ variants, effect-allele/EAF tracking) *or* a
  **list of tabular columns** from the non-genetic data (analytes, biomarkers — the
  proteomics use case), selected by list or pattern.
- **Study axis** — **cross-sectional** (linear / logistic) *and* **longitudinal**
  (repeated measures per subject): rate-of-progression (`gallop` / `slope`),
  trajectory **variability** (`trajgwas`), and time-to-event **incidence** (`cox`).

Named "Gran Prix" — the **GP** nods to the **GPU**-accelerated batched scan, and
Grand Prix racing is all about flat-out speed. GWAS is the most demanding case, but
the engine accommodates any tabular predictor set.

> Status: **design approved in principle; not yet implemented.** This file is the
> contract. Decisions captured from planning are marked **[decided]**.

---

## 1. Goals (from `mike_notes.txt`)

1. Companion tool with **similar architecture** to `proteomics_data_mine`; **same
   YAML interface**. **[decided]**
2. **Pre-analysis filtering on any non-genetic column** before analyses start. **[decided]**
3. Two inputs: a **non-genetic clinical file** and a **plink2 dosage** genotype set. **[decided]**
4. Code **very similar across both projects**, but **GPU-accelerable**. **[decided]** — backend: **PyTorch cu128** with automatic CPU fallback.
5. **Fully synthetic dummy data** so the whole pipeline runs with no real data. **[decided]**
6. Track **effect allele + effect-allele frequency per SNP after subsetting**, plus **N** and standard summary statistics. **[decided]**
7. **Explicit covariates only** — no prefix-based covariate injection. **[decided]** — PCs are pre-computed covariate columns in the clinical file.
8. Identify **where GPUs help**. **[decided]** — see §7.
9. Be **memory- and core-efficient**. **[decided]** — block-streaming + I/O pipeline, see §8.

---

## 2. Environment (probed 2026-07-16)

- **GPU:** NVIDIA RTX 5080 Laptop, 16 GB VRAM, driver 577.05. Blackwell (sm_120) → needs a **CUDA 12.8** PyTorch build.
- Python 3.12; `numpy 2.4`, `pandas 2.3`, `scipy 1.17`, `statsmodels 0.14`, `matplotlib`, `seaborn`, `lifelines 0.30` already present.
- **To install:** `torch` (cu128), `Pgenlib` (plink2 reader). Optional: `plink2` binary for QC/conversions.
- 16 CPU cores, 31 GB RAM.
- ⚠️ Repo lives on `/mnt/c` (Windows mount). Large `.pgen` files should be **staged on the Linux ext4 side (`~`)** — reads through the WSL↔Windows bridge are slow and will bottleneck the scan.

---

## 3. Data model

Two inputs, joined by sample ID:

```
clinical (LONG: one row per subject × visit)        genotype (STATIC: one per subject)
┌───────┬────────┬──────┬─────┬─────┬─────┐          plink2 fileset:
│ IID   │ visit  │ time │ AGE │ PC1 │ bmk │            <prefix>.pgen   (dosages)
├───────┼────────┼──────┼─────┼─────┼─────┤            <prefix>.pvar   (CHR,POS,ID,REF,ALT)
│ S001  │ BL     │ 0.0  │ 61  │ ... │ ... │            <prefix>.psam   (sample IDs, order)
│ S001  │ V02    │ 1.1  │ 62  │ ... │ ... │
│ S001  │ V04    │ 2.0  │ 63  │ ... │ ... │          genotype g_i is broadcast to ALL of
│ S002  │ BL     │ 0.0  │ 55  │ ... │ ... │          subject i's visit rows.
└───────┴────────┴──────┴─────┴─────┴─────┘
```

- **`sample_id_col`** links clinical rows to `.psam`. Subjects present in both are kept; the analysis subset is aligned to `.psam` order once.
- **`time_col`** is the longitudinal axis (e.g. `years_from_baseline`), a continuous covariate — irregular visit spacing is fine.
- Effect allele = the **counted allele (ALT / A1)** in plink2 dosages; frequencies are computed on the **post-filter analysis subset** (§6).

### Derived columns & conditional (SNP) covariates

**Derive covariates or outcomes from existing data** — a `derive:` block computes
new columns (usable anywhere a real column is) once on CPU, after prefilter and
before the GPU scan, from clinical *or* genetic data:

- **Row-wise clinical expressions** (pandas `eval`): `logLEDD: "log(LEDD + 1)"`, `old: "AGE >= 65"`.
- **Per-subject longitudinal aggregates** (computed within `sample_id_col`, broadcast back): `baseline_of(bmk)`, `slope_of(bmk ~ time)`, `delta(bmk)`. Lets a derived **outcome** (e.g. a per-subject slope → a `linear` run) or a derived **covariate** (e.g. baseline value) come straight from the longitudinal data.
- **Genotype pulls**: `G_rs76904798: "snp(rs76904798)"` materializes a specific variant's dosage (by `.pvar` ID) as a named column usable as a covariate or outcome.

**SNPs as covariates → conditional analysis.** Any covariate may be a genotype:
list variant IDs in `condition_snps:` (sugar mirroring plink `--condition-list`)
or reference a `derive`d `snp(...)` column in `covariates:`. Conditioning dosages
are pulled once and **folded into the null/covariate design**, so the per-SNP scan
tests each variant **conditional on** the index SNP(s) — the standard way to
resolve whether secondary signals at a locus are independent. The conditioning set
is recorded per run (output column + log) for provenance. Cost is a few extra
covariate columns — negligible, and fully compatible with the residualize-then-scan
GPU path (they enter the null model, so the batched scan is unchanged).

### Predict-from-baseline (baseline predictor → trajectory / time-to-event)

A common design: does a **baseline** value predict what happens *next* — the
subsequent **trajectory** of a continuous measure, or the **time to an event**?
This is the native shape of the longitudinal models (a static predictor driving a
longitudinal outcome), with one convenience switch:

- **`predictor_at: baseline`** (run or default key) reduces each looped predictor to
  its **per-subject baseline value** (earliest `time_col`, or a `baseline_visit` /
  `baseline_filter` override) and holds it constant across visits. SNP genotypes are
  already static → **no-op in genetic mode**; the switch is what makes *tabular*
  baseline predictors one line instead of a manual `derive: baseline_of(...)`.
- The **outcome still uses all visits** (continuous) or the full follow-up (event) —
  only the *predictor* is pinned to baseline.

| Question | Config |
|---|---|
| Baseline predictor(s) → **trajectory** of a continuous measure (rate of change) | `model: gallop`, `predictor_at: baseline`, `report_term: SNPxTIME` (predictor×time = effect on slope). `model: slope` is the simpler two-stage equivalent. |
| Baseline predictor(s) → **time to an event** (binary) | `model: cox`, `predictor_at: baseline`, `duration_col`/`event_col` → HR per unit of the baseline predictor. |

`baseline_visit: BL` (or `baseline_filter: "visit == 'BL'"`) defines "baseline";
default is the earliest `time_col` per subject.

---

## 4. Repository layout (mirrors the reference's flat-script style)

| File | Role | Reuse from reference |
|---|---|---|
| `make_synthetic_data.py` | Generate synthetic clinical long table + plink2 `pgen/pvar/psam` with dosages and **planted** SNP-main / SNP×time / event signals. | new |
| `gwas.py` | **Core engine** (analog of `regressions.py`): clinical load → prefilter → strata loop → align to `.psam` → null fit → **stream SNP blocks to GPU** → per-SNP stats → effect-allele/EAF/N → numerical filter + genome-wide sig + λ-GC → CSV/run×stratum. | heavy — same skeleton |
| `genoio.py` | `pgenlib` block reader (variant-major), missingness, EAF/MAF, dosage standardization, I/O prefetch. | new |
| `gpu.py` | Backend abstraction (PyTorch, auto CPU fallback) + batched **linear / logistic / GALLOP / slope / cox-score** kernels + **SPA**. | new |
| `meta_analysis.py` | IVW-FE + DL-RE across strata. Join on **SNP** (+ harmonized effect allele) instead of `(predictor, outcome)`. | ~verbatim |
| `manhattan_qq.py` | Manhattan + QQ per run (CHR/POS from `.pvar`). | adapt `plot_results.py` |
| `build_results_summary.py` | Cross-run rollup, λ table, top-hits table. | adapt |
| `batch.yaml`, `smoke_test.yaml` | Full config + fast smoke config against synthetic data. | adapt |
| `requirements.txt`, `README.md`, per-script READMEs | Same doc discipline as the reference. | adapt |

**Design principle — one standalone script per pipeline phase.** Following the
reference, **`meta_analysis.py` and `manhattan_qq.py` are separate, independently
runnable programs**, each with its own CLI and its own **filename-based
auto-discovery** of inputs — *not* functions bundled inside `gwas.py`. The core
engine writes `results/<run>-<stratum>-<ts>.csv`; meta-analysis and plotting are
decoupled downstream steps that consume those CSVs, so either can be re-run (or
re-pointed at a different results dir) without touching the association scan or
each other. `build_results_summary.py` is likewise standalone.

---

## 5. Model menu

Every model shares the pattern **fit the null once (CPU) → batch the per-predictor
association across a block (GPU)**. Fallback tiers mirror the reference's
3-tier LMM/Cox robustness.

### Predictor source: genetic (SNPs) or tabular

The looped predictor axis is pluggable via `predictor_source`:

- **`genetic`** (default) — loop **SNPs streamed from the plink2 fileset**; effect
  allele / EAF / MAF tracked per SNP (§6).
- **`tabular`** — **no genotype fileset needed**: loop a list of **columns from the
  clinical/tabular data** (analytes, biomarkers, any numeric columns) as the tested
  predictor, selected by explicit `[list]` or a `{prefix|suffix|regex: ...}` pattern
  (proteomics-style, e.g. `{suffix: _NPX}`). This makes the tool a direct companion
  to `proteomics_data_mine` — every model below runs identically, so you get
  GPU-accelerated longitudinal LMM / TrajGWAS / Cox over tabular predictors too.

**Every model is predictor-source-agnostic** — the tested column is the looped SNP
*or* the looped tabular column; only the identity/QC output columns differ (§6). In
tabular mode predictors are z-scored (with the reference's `zscore_binary` /
`zscore_dosage` exceptions); "SNP×time" reads as "predictor×time".

### Cross-sectional (one row per subject; use `sample_filter` to pick a visit)

| `model` | Question | GPU scan | Robustness / fallback |
|---|---|---|---|
| `linear` | Quantitative trait vs SNP | closed-form β/SE/P per block (residualize → `gᵀy`, `gᵀg`) | exact; MAF/MAC flag |
| `logistic` | Case/control vs SNP | score-test P per block | **SPA** in tail; **Firth** refit on hits/separation |

### Longitudinal — continuous biomarker → **rate of progression**

| `model` | Question | GPU scan | Robustness / fallback |
|---|---|---|---|
| `gallop` | Does SNP change the **annual slope** of a continuous biomarker? Reports **SNP main** *and* **SNP×time**. | null LMM (random intercept + slope) once → batched closed-form [SNP, SNP×time] β/SE via precomputed projections | tiers: **RI+slope → random-intercept-only → precision-weighted two-stage → OLS-on-slope**; `model_type` records the tier |
| `slope` | Two-stage cross-check: per-subject slope → GWAS on slopes | per-subject slopes (+ their variances) computed once → **precision-weighted** batched linear scan | robust by construction; MAF/MAC flag |
| `trajgwas` | **Location–scale** model: does SNP change the **mean** trajectory *and/or* the **within-subject variability** of a continuous biomarker? Reports a **mean (β)** effect and a **variability (τ)** effect. | null location-scale LMM once (mean random effects + log-linear within-subject-variance model) → batched per-SNP **score tests** for β_mean and τ_var | **SPA** on both tests (native to TrajGWAS); MAF/MAC flag; full refit on hits |

**`gallop` vs `trajgwas`:** `gallop` is the fast, primary answer to "rate of
progression" (mean slope). `trajgwas` is the more complete location–scale model —
its unique contribution is the **variability (τ) test**: SNPs that make a subject's
trajectory more *unstable* over time, independent of the mean level or slope
(Ko et al. 2022, AJHG). It carries a heavier one-time null fit (the dispersion
model) but the same batched per-SNP score-test scan, so genome-wide scale is
preserved.

**`trajgwas` always reports both effects together** — every SNP row carries the
**mean effect** (`beta_mean/SE_mean/P_mean`) *and* the **variability effect**
(`tau_var/SE_var/P_var`) in one pass, so the SNP's effect on level and its effect
on stability are read side by side. `report_term` (`MEAN` | `VAR` | `JOINT`) only
selects which becomes the primary `beta/SE/P` for downstream sorting/plotting;
`JOINT` reports the **2-df combined mean+variance score test** (`P_joint`) that
TrajGWAS uses to flag a variant significant on either axis.

**All three trajgwas statistics are GPU-accelerated.** The per-SNP mean, variance,
and joint (2-df) score statistics are batched GEMVs + reductions over each SNP
block on the GPU — identical cost profile to the other models. Only the **one-time**
null location–scale fit and the **SPA tail** run on CPU (§7), so the genome-wide
scan stays GPU-bound.

### Longitudinal — binary biomarker **incidence over time** → hazard

| `model` | Question | GPU scan | Robustness / fallback |
|---|---|---|---|
| `cox` | Does SNP change the **hazard** (rate) of reaching an event? HR per allele. | null Cox (covariates) once → **score test** on martingale residuals, batched | **SPA** (= SPACox) in tail; **full `lifelines` CoxPHFitter 3-tier** (unpenalized → ridge 0.01 → ridge 0.1) refit on hits/score-failures |

**Which model for "how much does the SNP change rate of progression?"**
- Continuous biomarker level → **`gallop`** (SNP×time is the rate effect); `slope` as a fast cross-check.
- Binary biomarker incidence → **`cox`** (HR per allele = acceleration to event).

**β / SE accuracy:** `linear`, `gallop`, `slope` yield exact effect sizes directly
from the batched algebra. `logistic` / `cox` score tests give the P fast but an
*approximate* β → **hits are refit** (full/Firth/lifelines) so reported effect
sizes are exact for anything that matters.

---

## 6. Effect-allele & summary-statistic output (req #6)

One CSV per run × stratum: `results/<run>-<stratum>-<timestamp>.csv`. Columns:

| Column | Meaning |
|---|---|
| `predictor` | **Universal identity** of the tested column: SNP id (genetic) or tabular column name (tabular). Always populated. |
| `CHR`, `POS` | Genomic position from `.pvar`. **Null for non-SNP predictors.** |
| `effect_allele`, `other_allele` | Effect = counted ALT/A1; other = REF. **Null for non-SNP predictors.** |
| `EAF` | **Effect-allele freq on the post-filter analysis subset** = mean(dosage)/2. **Null for non-SNP predictors.** |
| `MAF`, `missing_rate` | Minor-allele freq and per-SNP missingness on the subset. **Null for non-SNP predictors.** |
| `predictor_mean`, `predictor_sd`, `zscored` | Standardization of the tested column (populated when z-scored — tabular predictors, and any standardized genetic run). |
| `N` | Subjects in the fit. |
| `n_obs`, `n_subjects` | Observations vs unique subjects (longitudinal). |
| `n_events` | Events (cox only). |
| `beta`, `SE`, `P` | Primary term (see `report_term`), 2-sided. |
| `beta_SNP/SE_SNP/P_SNP`, `beta_SNPxTIME/SE_SNPxTIME/P_SNPxTIME` | Both terms for `gallop` (the `SNP*` names read as the tested predictor / predictor×time). |
| `beta_mean/SE_mean/P_mean`, `tau_var/SE_var/P_var` | Mean and within-subject-variability effects for `trajgwas`. |
| `model_type` | Which fit tier/fallback succeeded. |
| `test` | `wald` / `score` / `spa` / `firth` / `full`. |
| `genomewide_sig` | `P < 5e-8` (standard GWAS line). |
| `bonferroni_threshold`, `significant` | Per-run `0.05 / N_predictors` (carried from reference). |
| `conditioned_on` | Conditioning SNP set for this run (empty if none), for provenance. |
| `strata` | Stratum value. |

**One stable schema across both predictor modes.** The column set is fixed; when the
predictor is **not a SNP** (`predictor_source: tabular`), the SNP-specific fields
(`CHR/POS/effect_allele/other_allele/EAF/MAF/missing_rate`) are written **null**, and
`predictor` + the standardization columns carry the identity instead. This keeps every
downstream consumer (meta, plots, summary) mode-agnostic. EAF/MAF/N/missing are
recomputed **per stratum after all filtering**, so they reflect the actual sample
each estimate came from.

---

## 7. Where GPUs help (and where they don't)

**GPU (the win — batched across a whole SNP block, no per-SNP loop):**
1. `linear` / `slope`: residualize block against covariates → `gᵀy`, `gᵀg` → β/SE/t. Millions of SNPs in a few GEMMs.
2. `gallop`: per-SNP [SNP, SNP×time] solve reduces to precomputed per-subject projections + batched 2×2 closed-form solves.
3. `logistic` / `cox`: score statistics = `Gᵀr` (residual) + precomputed variance → GEMV + reductions.
4. Dosage standardization, **EAF/MAF/N/missingness** = column reductions.

**CPU (small or iterative, done once or on the tail):**
- Null-model fits (LMM variance components, null Cox/logistic) — one-time.
- YAML parse, clinical prefilter (pandas).
- **P-values from test statistics** — cheap `scipy` call on the per-SNP stat vector, in float64.
- **SPA root-finding** and **Firth/lifelines refits** — only on the significant/rare tail.

**Numerical care:** dosages stored float32; **reductions accumulated in float64**
(β/SE stable in the significant tail); p-values on CPU in float64.

---

## 8. Memory & core efficiency (req #9)

- **Block-stream variants:** `block_size` SNPs × N subjects per block; only the
  current block is on GPU (float32; a 5k×20k block ≈ 0.4 GB). Bounds host RAM and
  VRAM regardless of dataset size.
- **Prefilter clinical first**, then read genotypes only for retained subjects →
  less I/O and memory.
- **Pipeline I/O:** a prefetch thread decompresses the next block (pgenlib, multi-
  threaded) while the GPU computes the current one — keeps the 5080 fed. I/O, not
  compute, is the expected bottleneck.
- **Stage `.pgen` on ext4 (`~`), not `/mnt/c`.**
- Pinned host buffers + CUDA stream for overlapped host→device transfer.

**Resource policy — max the machine unless capped.** At startup the tool detects
logical cores, total RAM, and free VRAM, and by default uses ~all of them
(`max_cpus`/`max_ram_gb`/`max_gpu_mem_gb` = `null`). `block_size: auto` then picks
the **largest SNP block that fits the VRAM budget** for the current N (block bytes ≈
`N × block_size × 4`, leaving headroom for the design/residual matrices), so the
user never hand-tunes it. Any cap the user *does* set is a hard ceiling: `max_cpus`
bounds the pgen-decompression / prefetch / null-fit thread pools; `max_ram_gb` caps
host buffers (and shrinks `block_size` if needed); `max_gpu_mem_gb` caps VRAM (and
shrinks `block_size`); `device: cpu` forces the CPU path with the same block logic
sized to `max_ram_gb`. On a shared machine, set the caps; on a dedicated box, leave
them null and it saturates.

---

## 9. YAML schema (adapted from the reference)

```yaml
defaults:
  # --- inputs ---
  clinical_file: "synthetic_clinical-*.tsv"    # non-genetic long table; latest glob
  genotype_prefix: "synthetic_geno"            # -> .pgen/.pvar/.psam
  sample_id_col: IID                           # links clinical rows to .psam
  time_col: years_from_baseline                # longitudinal axis
  output_dir: "results"

  # --- pre-analysis filter on ANY clinical column(s), applied once (req #2) ---
  prefilter:
    - "AGE >= 40"
    - "DIAGNOSIS == 'PD'"

  # --- covariates: explicit only, no prefix injection (req #7) ---
  covariates: [AGE, SEX, PC1, PC2, PC3, PC4, PC5]

  # --- predictor axis: genetic SNPs (default) OR tabular columns ---
  predictor_source: genetic          # genetic | tabular
  # genotype_prefix is required for genetic mode; ignored/optional for tabular.
  # predictors: {suffix: _NPX}        # tabular mode: explicit [list] or {prefix|suffix|regex: ...}
  # predictor_at: all                 # all | baseline (baseline -> pin each predictor to its baseline value)
  # baseline_visit: BL                # what "baseline" means (default: earliest time per subject)

  # --- derive covariates/outcomes from existing clinical or genetic data ---
  # derive:
  #   logLEDD: "log(LEDD + 1)"          # row-wise clinical expression
  #   bmk_baseline: "baseline_of(bmk)"  # per-subject aggregate, broadcast to visits
  #   bmk_slope: "slope_of(bmk ~ time)" # per-subject slope as a derivable outcome
  #   G_rs76904798: "snp(rs76904798)"   # genotype pulled by variant ID

  # --- per-ancestry GWAS -> meta ---
  strata_col: ancestry
  strata: [EUR, AJ]

  # --- variant QC ---
  maf_min: 0.01
  mac_min: 20
  missing_max: 0.05
  min_n: 100

  # --- compute / resource caps (unspecified -> use the whole machine) ---
  device: auto           # auto (GPU if present, else CPU) | cuda | cpu
  max_cpus: null         # null -> all logical cores (detected)
  max_ram_gb: null       # null -> ~90% of total system RAM for host buffers
  max_gpu_mem_gb: null   # null -> ~90% of free VRAM
  block_size: auto       # auto -> largest block fitting the VRAM budget given N; or an int
  prefetch: true         # overlap pgen I/O with GPU compute (double-buffer)

runs:
  - name: baseline_bmkX
    model: linear
    outcome: bmkX
    sample_filter: "visit == 'BL'"

  - name: casecontrol_PD
    model: logistic
    outcome: PD_case
    sample_filter: "visit == 'BL'"
    spa: true

  - name: progression_bmkX            # continuous rate-of-progression
    model: gallop
    outcome: bmkX
    report_term: SNPxTIME             # SNP | SNPxTIME (both are always written)

  - name: progression_bmkX_2stage     # fast cross-check
    model: slope
    outcome: bmkX

  - name: variability_bmkX            # location-scale: mean + within-subject variability
    model: trajgwas
    outcome: bmkX
    report_term: VAR                  # MEAN | VAR (both are always written)
    spa: true

  - name: incidence_dementia          # binary incidence over time
    model: cox
    duration_col: tte_dementia_years
    event_col: event_dementia
    spa: true

  - name: locusX_conditional          # conditional analysis: test each SNP GIVEN the index SNP(s)
    model: linear
    outcome: bmkX
    sample_filter: "visit == 'BL'"
    condition_snps: [rs76904798]

  - name: analytes_vs_updrs           # tabular predictors (proteomics-style); no genotype fileset used
    model: linear
    outcome: updrs3_bl
    sample_filter: "visit == 'BL'"
    predictor_source: tabular
    predictors: {suffix: _NPX}        # loop every column ending _NPX as the tested predictor

  - name: baseline_analyte_to_slope   # predict-from-baseline: baseline analyte -> subsequent trajectory
    model: gallop
    outcome: updrs3
    predictor_source: tabular
    predictors: {suffix: _NPX}
    predictor_at: baseline            # pin each analyte to its baseline value; outcome stays longitudinal
    report_term: SNPxTIME             # baseline analyte × time = effect on rate of change
```

Per-run overrides carried from the reference: `sample_filter`, `add_covariates` /
`drop_covariates`, `strata`, `min_n`. **New keys:** `predictor_source`,
`predictors` (tabular selection), `derive`, `condition_snps`, `spa`, and the
variant-QC / compute settings. **Dropped from the reference:** prefix-based
assay-PC injection (req #7). `genotype_prefix` is required only in genetic mode.

### Sensible defaults (every key works unset)

A config can be as short as `clinical_file` + `genotype_prefix` + one run. Effective
defaults when a key is omitted:

| Key | Default | Rationale |
|---|---|---|
| `predictor_source` | `genetic` | GWAS is the primary use case. |
| `predictor_at` | `all` | set `baseline` for predict-from-baseline designs (no-op for SNPs). |
| `strata_col` / `strata` | none → single **pooled** analysis | works without ancestry labels. |
| `covariates` | `[]` (intercept only) | explicit-only (req #7); nothing injected silently. |
| `maf_min` / `mac_min` | `0.01` / `20` | standard common-variant GWAS QC. |
| `missing_max` | `0.05` | drop variants >5% missing. |
| `min_n` | `30` | floor for a stable fit; per-run overridable. |
| `spa` | `true` for `logistic`/`cox`, else n/a | robust tails by default; cost only on the tail. |
| `report_term` | `gallop`→`SNPxTIME`, `trajgwas`→`JOINT` | headline = progression / either-axis; all terms always written. |
| `device` | `auto` | GPU if present, else identical CPU path. |
| `max_cpus`/`max_ram_gb`/`max_gpu_mem_gb` | `null` → ~all of the machine | saturate by default; caps are hard ceilings. |
| `block_size` | `auto` | sized to the VRAM/RAM budget for the current N. |
| `prefetch` | `true` | overlap I/O with compute. |
| significance | genome-wide `5e-8` **and** per-run Bonferroni `0.05/N` | both reported. |
| numeric precision | float32 storage, **float64 reductions**, CPU float64 p-values | stable tails. |

Missing columns referenced by a run remain a **fatal, loud** error (as in the
reference) — sensible defaults never mask a typo'd column.

---

## 10. Downstream (reused)

- **`meta_analysis.py`** — same IVW-FE + DL-RE math; join key becomes `SNP` and
  effect alleles are harmonized across strata (flip β sign + EAF if a stratum's
  counted allele differs — won't happen when strata share one `.pgen`, but the
  check stays). Heterogeneity Q / I² / τ² / direction string as before.
- **`manhattan_qq.py`** — Manhattan (−log₁₀P vs genomic position, per-stratum +
  meta) + QQ with λ-GC annotation, genome-wide + Bonferroni lines.
- **`build_results_summary.py`** — coverage table (runs × λ per stratum/FE/RE),
  top deduped genome-wide hits sorted by P.

---

## 11. Assumptions & limitations (stated up front)

- **Unrelated cohort (current).** The random effect models *within-subject* repeated
  measures only — there is **no genetic relatedness matrix (GRM)**. Relateds must be
  pruned upstream (e.g. KING / `--king-cutoff`); **PCs (pre-computed clinical covariate
  columns) control population structure**.
  - **Future work — related samples via GRM.** A planned, larger-scope upgrade adds a
    **sparse-GRM random effect** (fastGWA-GLMM / TrajGWAS-style) to the null model so
    related individuals no longer have to be pruned. The batched residualize-then-scan
    stays unchanged (the kinship term enters the one-time null fit), consistent with the
    engine's design.
- **Trajectory variability** (variance-QTLs) is covered by the `trajgwas`
  location–scale model (§5); `gallop` alone captures the mean slope only. TrajGWAS's
  dispersion (τ) model assumes the within-subject variance is log-linear in its
  covariates — a modeling choice, not a guarantee.
- **MAR dropout** for the LMM paths (Cox handles censoring). Documented, not corrected.
- **Two-step variance components** fixed from the null fit (standard; λ-GC is the
  calibration check).
- λ-GC reported per run as the honest calibration diagnostic.

---

## 12. Build order

| Phase | Deliverable | Verifies |
|---|---|---|
| 0 | `requirements.txt` + installs (torch cu128, Pgenlib) + `gpu.py` device/CPU-fallback smoke + **resource detection** (cores/RAM/VRAM → auto `block_size`, honoring caps) | GPU reachable; CPU fallback works; auto-sizing respects caps and maxes when null |
| 1 | `make_synthetic_data.py` → clinical long TSV + plink2 pgen/pvar/psam with **planted signals** (SNP-main, SNP×time, **within-subject-variance**, and event effects; plus null SNPs) | pipeline testable with zero real data; each model has a known truth to recover |
| 2 | `genoio.py` block reader + EAF/MAF/N; `gwas.py` skeleton (config, prefilter, strata, align, CSV, `derive:`, `condition_snps`, `predictor_source: genetic\|tabular`) with **`linear`** | end-to-end scan in **both predictor modes**; **planted linear signal recovered**, EAF correct, conditional analysis drops the index signal |
| 3 | `logistic` (+SPA/Firth), `cox` (score + lifelines fallback + SPA) | planted case/control + event signals recovered; rare-variant calibration |
| 4 | `gallop` (+ fallback tiers) and `slope` (precision-weighted) | planted **SNP×time** signal recovered; λ-GC sane |
| 5 | `trajgwas` location–scale: null dispersion fit + batched GPU mean/variance/joint score tests + SPA | planted **variability (τ)** signal recovered; mean axis agrees with `gallop`; λ-GC sane |
| 6 | `meta_analysis.py`, `manhattan_qq.py`, `build_results_summary.py`, `batch.yaml`/`smoke_test.yaml`, READMEs | full pipeline + docs parity with reference |

Each phase is validated against the synthetic data's known planted effects before
moving on (recover the truth, check λ-GC ≈ 1 on null SNPs).

---

## 13. Dependencies (`requirements.txt`)

```
# shared with proteomics_data_mine
pandas>=2.0,<3.0
numpy>=1.24
scipy>=1.11
statsmodels>=0.14
lifelines>=0.29          # cox full-fit fallback
PyYAML>=6.0
matplotlib>=3.8
seaborn>=0.13
adjustText>=1.1

# GWAS-specific
Pgenlib>=0.90            # plink2 pgen/pvar/psam reader
torch                    # install from cu128 index for RTX 5080 (Blackwell):
                         #   pip install torch --index-url https://download.pytorch.org/whl/cu128
```
