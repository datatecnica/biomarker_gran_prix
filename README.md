# Biomarker Gran Prix

> **Note.** This codebase is an evolution and expansion of a proven codebase used in
> PPMI biomarkers analyses. It is currently under UAT, and we offer no warranty, safety,
> or similar in its use or future development.

A GPU-accelerated, YAML-driven batch **association engine** for biomarker studies —
a companion to **proteomics_data_mine** that reuses its
architecture (one config → many runs → one CSV per run×stratum → meta-analysis →
plots → summary) and generalizes it on two axes:

- **Predictor axis** — the looped predictor is either **SNPs streamed from a plink2
  dosage fileset** (full GWAS: 10⁵–10⁷ variants, effect-allele / EAF tracked) *or* a
  **list of tabular columns** from the clinical data (analytes / biomarkers — the
  proteomics use case), selected by list or pattern.
- **Study axis** — **cross-sectional** (linear / logistic) *and* **longitudinal**
  (repeated measures per subject): rate-of-progression (`gallop` / `slope`),
  trajectory **variability** (`trajgwas`), and time-to-event **incidence** (`cox`).

Named "Gran Prix" — the **GP** nods to the **GPU**-accelerated batched scan, and
Grand Prix racing is all about flat-out speed.

> **Status: feature-complete (Phases 0–6).** The full design is in [PLAN.md](PLAN.md)
> (the contract). All six models and the full downstream pipeline run end-to-end on the
> synthetic data; each phase is validated against the planted truth. See
> [Build status](#build-status) below. The GRM upgrade for related samples is the main
> planned extension (see [Assumptions & limitations](#assumptions--limitations)).

## Executive summary

- **Inputs** — two files joined by sample ID (PLAN §3): a **clinical LONG table**
  (one row per subject × visit) and a **plink2 dosage** genotype set
  (`<prefix>.pgen/.pvar/.psam`). Effect allele = counted ALT/A1.
- **Predictors looped per fit** — SNPs (genetic mode, default) or tabular columns
  selected by `[list]` or `{prefix|suffix|regex}` (tabular mode). Every model runs
  identically over both; only the identity/QC output columns differ.
- **Models** — `linear`, `logistic` (score test + SPA + Firth), `cox` (martingale
  score + SPACox + lifelines 3-tier refit), `slope` (two-stage), `gallop` (one-stage
  RI+slope LMM, SNP main + SNP×time), `trajgwas` (location–scale: mean + within-subject
  variability τ + 2-df joint). All six model types run; downstream (meta/plots) is Phase 6.
- **Backend** — PyTorch (CUDA when present, identical CPU path otherwise). The batched
  residualize-then-scan runs ~**7–12× faster on GPU** than the CPU path at GWAS scale
  (I/O-bound at toy scale; stage `.pgen` on ext4). Reductions accumulate in float64;
  p-values computed on CPU in float64.
- **Synthetic data** — `make_synthetic_data.py` generates a complete test bed (clinical
  TSV + plink2 fileset) with **planted signals** for every model, so the whole pipeline
  runs with zero real data and each estimate has a known truth to recover.
- **Strata & significance** — optional per-ancestry stratification; EAF/MAF/N recomputed
  per stratum after filtering. Genome-wide `5e-8` **and** per-run Bonferroni reported;
  λ-GC logged per run as the calibration check.

## Pipeline at a glance

```
make_synthetic_data.py  ─►  synthetic_clinical-<ts>.tsv          (LONG: subject × visit)
   (or your own data)       synthetic_geno.{pgen,pvar,psam}      (plink2 dosages)
                            synthetic_truth-<ts>.tsv             (planted effects, for --validate)
                            │
                            ▼
                       gwas.py  ──[smoke_test.yaml / batch.yaml]──►  results/<run>-<stratum>-<ts>.csv
                            │      clinical load → prefilter → derive: → per-stratum
                            │      → align to .psam → null fit → stream SNP blocks to GPU
                            │      → β/SE/P + effect-allele/EAF/N → QC + λ-GC
                            ▼
                       meta_analysis.py        ──►  meta/META_<run>-<ts>.csv     (Phase 6, planned)
                            ▼
                       manhattan_qq.py         ──►  plots/<run>-<ts>.png         (Phase 6, planned)
                            ▼
                       build_results_summary.py ─►  results_summary.md           (Phase 6, planned)
```

Every downstream step auto-discovers its inputs by filename pattern, so runs can be
re-executed independently (mirrors the reference's one-script-per-phase discipline).

## Repository contents

### Code

| File | What it does | Status |
|---|---|---|
| [make_synthetic_data.py](make_synthetic_data.py) | Generates the synthetic clinical LONG table + plink2 fileset with **planted** SNP-main / SNP×time / within-subject-variance / logistic / cox / tabular-analyte signals (+ null SNPs + an LD pair for conditional analysis). `--validate` recovers every effect and checks λ-GC≈1. | ✅ Phase 1 |
| [genoio.py](genoio.py) | Streaming plink2 reader: `.pvar`/`.psam` parse, sample alignment, variant-major **block iterator** with prefetch, per-variant EAF/MAF/MAC/missing/N, read-by-ID (for `condition_snps` / `derive`). | ✅ Phase 2 |
| [gpu.py](gpu.py) | Compute backend: device resolution + CPU fallback, resource detection, caps-aware auto `block_size`; batched kernels — `scan_linear`, `scan_logistic` (score test), `scan_gallop` (2×2 GLS). | ✅ Phase 0/2/3/4 |
| [lmm.py](lmm.py) | Robust **EM** fitter for the random-intercept+slope null model (grouped/batched by visit count) + GALLOP per-subject precompute. Purpose-built because generic LMM fitters (statsmodels MixedLM) are numerically fragile for random-slope GWAS. | ✅ Phase 4 |
| [robust.py](robust.py) | CPU tail refinements: Lugannani–Rice **SPA** (logistic + SPACox + trajgwas variance), **Firth** logistic, lifelines 3-tier Cox refit — applied only to the significant/rare tail. | ✅ Phase 3 |
| [gwas.py](gwas.py) | **Core engine** (analog of `regressions.py`): config, global prefilter, `derive:`, strata, `.psam` alignment, one-time null fit, block scan, `condition_snps`, `predictor_source: genetic\|tabular`, QC + genome-wide sig + λ-GC + CSV. Models: linear / logistic / cox / slope / gallop. | ✅ Phase 2–4 |
| [meta_analysis.py](meta_analysis.py) | IVW-FE + DL-RE meta across **any number** of strata/datasets, joined on the predictor with **effect-allele harmonization** and **palindromic/strand flagging**; handles genetic *and* non-genetic results. | ✅ Phase 6 |
| [manhattan_qq.py](manhattan_qq.py) | Per source (study **or** meta): **Manhattan** (grey/black chromosomes, burgundy/mint hits) for genetic, **volcano** for tabular, each with a **QQ + λ-GC by MAF band**. | ✅ Phase 6 |
| [build_results_summary.py](build_results_summary.py) | Cross-run rollup: coverage table (λ per stratum/FE/RE), **λ by MAF band**, top deduped genome-wide hits. | ✅ Phase 6 |

### Configs / requirements

| File | Purpose |
|---|---|
| [smoke_test.yaml](smoke_test.yaml) | Fast config against the synthetic data: genetic linear (pooled + per-ancestry), conditional analysis, `derive:`, logistic, cox, slope, gallop, and tabular. Verifies the engine end-to-end in seconds. |
| `batch.yaml` | Full production config. ⏳ Phase 6. |
| [requirements.txt](requirements.txt) | Deps grouped by phase; torch installed separately from the CUDA 12.8 index (see the file header). |

## Model menu

Every model shares the pattern **fit the null once (CPU) → batch the per-predictor
association across a block (GPU)** (PLAN §5). Robustness tiers mirror the reference.

| `model` | Question | Scan | Robustness |
|---|---|---|---|
| `linear` | Quantitative trait vs predictor | closed-form β/SE/P (residualize → `gᵀy`, `gᵀg`) | exact; collinearity guard; MAF/MAC/missing QC |
| `logistic` | Case/control vs predictor | batched **score test** | **SPA** in the tail; **Firth** refit on genome-wide hits |
| `cox` | Change in **hazard** of an event | null Cox martingale residuals → batched score | **SPACox** in the tail; **lifelines 3-tier** refit on hits |
| `slope` | Two-stage: per-subject slope → GWAS on slopes | unweighted "derived-phenotype" GWAS | robust cross-check; calibrated when visit structure is balanced |
| `gallop` | Does the SNP change the **annual slope**? Reports **SNP main** *and* **SNP×time** | one-stage RI+slope LMM null once → batched 2×2 GLS | falls back to `slope` on LMM failure; `report_term` picks the headline |
| `trajgwas` | Location–scale: SNP effect on the **mean** trajectory *and* on **within-subject variability** (τ) | LMM null → mean term reuses GALLOP; τ via a dispersion model + variance-residual score; 2-df **JOINT** | **SPA** on the variance tail; `report_term` MEAN\|VAR\|JOINT (mean axis is identical to `gallop`) |

Genetic dosages are used raw (β per allele); **non-genetic (tabular) predictors are
z-scored** by default (continuous only; `{0,1}` binary and `{0,1,2}` dosage skipped
unless enabled), mirroring `../proteomics_data_mine`. All z-score switches are per-run
overridable (`zscore_predictors` / `zscore_binary` / `zscore_dosage`).

## Performance (measured on 1× RTX 5080 Laptop, 16 GB; 16 CPU threads)

A run's wall time has **four components**, not one — and the fast GPU scan is usually the
*smallest*:

| component | where | cost (measured) |
|---|---|---|
| **1. pgen I/O** (read + decompress) | CPU, multi-threaded | ~64k variants/s at N=2k on ext4; scales ~1/N (each variant = N dosages) → ~**1.3k var/s at N=100k**. Overlaps the scan via prefetch. |
| **2. null-fit** (once per run×stratum) | CPU | linear ~0 · logistic 1–130 ms · cox 34 ms–1.8 s · **gallop/trajgwas LMM-EM 1.1 s → 1.7 min** (N=1k→100k) |
| **3. batched scan** (β/SE/P per variant) | **GPU** | 12k–5.2M variants/s (below) — the GPU-accelerated part, and it is **hidden under I/O** |
| **4. SPA / Firth / lifelines-refit tail** | CPU, per tail-variant | **SPACox 0.6→30 ms/var**, logistic-SPA 0.2→20 ms/var (N=1k→100k), on the rare+significant fraction (~10%) |

**Batched scan throughput (GPU, variants/s)** — the component GPUs accelerate ~7–12× over CPU:

| model | N=1k | N=5k | N=10k | N=100k |
|---|---|---|---|---|
| linear | 5.2M | 1.0M | 442k | 43k |
| logistic | 2.8M | 608k | 325k | 30k |
| gallop (LMM) | 1.5M | 157k | 179k | 19k |
| trajgwas (var) | 1.2M | 246k | 123k | 12k |
| cox | 4.3M | 874k | 438k | 41k |

Because pgen I/O (component 1) is far slower than the scan (component 3) and the two overlap
via prefetch, **wall time ≈ null-fit + I/O + SPA tail** — the scan is effectively free. Realistic
**end-to-end** estimates for a **1M-variant** run (ext4-staged pgen):

| model | N=2k | N=20k | N=100k | bottleneck |
|---|---|---|---|---|
| linear | ~15 s | ~2.5 min | ~13 min | pgen I/O |
| gallop / trajgwas | ~15 s + EM | ~3 min + EM | ~13 min + **1.7 min EM** | I/O + LMM null-fit |
| logistic (+SPA) | ~35 s | ~15 min | **~45 min** | **SPA tail** |
| cox (+SPACox) | ~75 s | ~20 min | **~60 min** | **SPACox tail** |

Host RAM peaks ~1.4 GB (N=1k) → ~3.2 GB (N=100k); GPU memory ~1.5–2.3 GB per block.

**Takeaways / optimization targets:** (1) the batched scan does its job — the association
math is not the bottleneck; (2) **pgen I/O dominates** for the closed-form/score models —
stage on ext4, and the prefetch keeps the GPU fed; (3) the **SPA tail** (per-variant, CPU,
currently Python) dominates logistic/cox/trajgwas at scale and is the top optimization target
(SAIGE/SPACox vectorize it — a planned improvement); (4) the **LMM EM null-fit** dominates
gallop/trajgwas at very large N (once per run×stratum). Numbers are per run×stratum; a full
multi-run config multiplies accordingly.

## Running it

```bash
# one-time install (torch from the CUDA 12.8 index; see requirements.txt header)
python3 -m pip install --user -r requirements.txt
python3 -m pip install --user torch --index-url https://download.pytorch.org/whl/cu128

# generate the synthetic test bed (writes clinical TSV + plink2 fileset + truth file)
python3 make_synthetic_data.py                 # default N=2000, M=10000
python3 make_synthetic_data.py --smoke --validate   # tiny + recover-the-truth checks

# run the engine
python3 gwas.py --config smoke_test.yaml            # all runs
python3 gwas.py --config smoke_test.yaml --run gallop_bmkX   # a single run

# inspect / probe the compute backend
python3 gpu.py                                       # device + CPU-fallback + auto-sizing smoke
```

Each run writes one `results/<run>-<stratum>-<timestamp>.csv` with a stable schema
(PLAN §6) and logs λ-GC. For a real study, point `clinical_file` / `genotype_prefix`
in the config at your own data — **stage the `.pgen` on ext4 (`~`), not `/mnt/c`**;
reads through the WSL↔Windows bridge are ~5× slower and bottleneck the scan.

## Build status

| Phase | Deliverable | State |
|---|---|---|
| 0 | `requirements.txt` + installs + `gpu.py` device/CPU-fallback + resource detection | ✅ |
| 1 | `make_synthetic_data.py` with planted signals + `--validate` | ✅ |
| 2 | `genoio.py` + `gwas.py` skeleton with `linear`, both predictor modes, `derive:`, `condition_snps` | ✅ |
| 3 | `logistic` (+SPA/Firth), `cox` (score + SPACox + lifelines refit) | ✅ |
| 4 | `gallop` (+ slope fallback) and `slope` (two-stage) | ✅ |
| 5 | `trajgwas` location–scale (mean + within-subject variability τ) | ✅ |
| 6 | `meta_analysis.py`, `manhattan_qq.py`, `build_results_summary.py`, `batch.yaml`, per-script READMEs | ✅ |

**All phases complete.** The full pipeline runs end-to-end on the synthetic data:
`make_synthetic_data.py` → `gwas.py --config batch.yaml` → `meta_analysis.py` →
`manhattan_qq.py` → `build_results_summary.py`.

Each phase is validated against the synthetic data's known planted effects before
moving on (recover the truth; check λ-GC ≈ 1 on the null SNPs).

## License

Source-available under a **Noncommercial License** ([LICENSE](LICENSE)): free for
**academic, research, and non-profit** use with attribution; **commercial/for-profit**
use requires a separate license — contact **info@datatecnica.com**. This is a research
tool, not for clinical or diagnostic use. (Note: because it restricts commercial use it
is *source-available*, not an OSI "open-source" license.)

## Assumptions & limitations

- **Unrelated cohort (current).** The random effects model *within-subject* repeated
  measures only — there is **no genetic relatedness matrix (GRM)**. Relateds must be
  pruned upstream, and **PCs (pre-computed clinical covariate columns) control
  population structure**.
  - **Future builds will accommodate related samples via a sparse-GRM random effect**
    (fastGWA-GLMM / TrajGWAS-style) — a deferred, larger-scope upgrade that adds a
    kinship random effect to the null model so related individuals no longer have to be
    pruned. Until then, prune relateds (e.g. KING/`--king-cutoff`) before running.
- **Trajectory variability** (variance-QTLs) is covered by the planned `trajgwas`
  location–scale model; `gallop` alone captures the mean slope only.
- **MAR dropout** is assumed for the LMM paths (Cox handles censoring); documented, not
  corrected.
- **Two-step variance components** fixed from the null fit (standard); λ-GC is the
  calibration check, reported per run.
- **Effect sizes:** `linear` / `gallop` / `slope` yield exact β directly; `logistic` /
  `cox` score tests give the P fast and an approximate β → **hits are refit**
  (Firth / lifelines) so reported effect sizes are exact for anything that matters.
