# make_synthetic_data.py — synthetic test bed

Generates a complete, **100% manufactured** dataset (reads no external data — every value
is drawn from a single seeded NumPy RNG) so the whole pipeline runs with zero real data.

```bash
python3 make_synthetic_data.py                 # default N=2000 subjects, M=10000 variants
python3 make_synthetic_data.py --smoke          # tiny + fast (CI/dev)
python3 make_synthetic_data.py --validate       # + recover-the-truth checks
python3 make_synthetic_data.py --n-subjects 20000 --n-variants 100000 --output-dir ~/gwas
```

## Outputs
- `synthetic_geno.{pgen,pvar,psam}` — plink2 dosage fileset (effect allele = ALT).
- `synthetic_clinical-<ts>.tsv` — clinical LONG table (one row per subject × visit):
  `IID, ancestry (EUR/AJ), DIAGNOSIS, SEX, visit, years_from_baseline, AGE, PC1–PC5, LEDD,
  bmkX, updrs3, updrs3_bl, PD_case, tte_dementia_years, event_dementia`, and `PRTN####_NPX`
  analyte columns.
- `synthetic_truth-<ts>.tsv` — every planted effect (for `--validate` and pipeline checks).

## Planted signals (each model has a known truth to recover)

| kind | drives | recovered by |
|---|---|---|
| `linear_main` | `bmkX` cross-sectional level | linear |
| `snpxtime` | `bmkX` annual slope (zero main effect) | gallop / slope |
| `variance` | within-subject variability of `bmkX` (centered τ) | trajgwas |
| `logistic` | P(PD case) | logistic |
| `cox` | dementia hazard | cox |
| `analyte_main` / `analyte_slope` | `updrs3_bl` level / `updrs3` slope | tabular |
| `ld_tag` | a null SNP in LD (r≈0.78) with index `rs76904798` | conditional analysis |

Also: two ancestries (EUR/AJ) with different allele frequencies (→ per-stratum EAF for
meta), a spectrum of MAFs including **rare variants** (MAF→0.0025), **fractional/imputed**
dosages, and per-variant **missingness** (→16%) — the full QC surface.

## `--validate`
Runs a lightweight recover-the-truth pass (a mini engine): recovers every planted effect
within ~4 SE (SE-based, N-robust tolerances) and checks λ-GC ≈ 1 on the null SNPs. Uses
correctly-specified fits (joint logistic/cox to avoid non-collapsibility bias; directional
+ significance check for the crude variance proxy).

## Notes
- Fully deterministic from `--seed`; no PII, no real genotypes/phenotypes.
- Realistic names (`rs76904798`, `updrs3`, `EUR`/`AJ`, `PD`/`dementia`) are labels only —
  the data behind them is entirely RNG-generated.
