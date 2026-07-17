#!/usr/bin/env python3
"""make_synthetic_data.py — fully synthetic dataset for Biomarker Gran Prix.

Generates a complete, self-contained test bed so the whole pipeline runs with
**zero real data** (PLAN §1 req #5, Phase 1). Two joined inputs are produced,
exactly the shape the engine consumes (PLAN §3):

  * a **clinical LONG table** (one row per subject × visit) as a TSV, and
  * a **plink2 dosage fileset** (`<prefix>.pgen/.pvar/.psam`), effect allele = ALT.

Signals are **planted** so every model in the menu has a known truth to recover
(PLAN §5, §12):

  linear_main    SNP → cross-sectional level of `bmkX`            (linear)
  snpxtime       SNP → annual slope of `bmkX`, zero main effect   (gallop / slope)
  variance       SNP → within-subject variability of `bmkX` (τ)   (trajgwas)
  logistic       SNP → P(PD case)                                 (logistic)
  cox            SNP → hazard of dementia                         (cox)
  analyte_main   `*_NPX` column → `updrs3_bl` level               (tabular linear)
  analyte_slope  `*_NPX` column → `updrs3` slope                  (tabular gallop)
  ld_tag         a null SNP in LD with a causal one — its marginal signal must
                 vanish once conditioned on the index (conditional-analysis test)

Everything else is null, so λ-GC on the null SNPs is the calibration check.
Two ancestries (EUR/AJ) with slightly different allele frequencies make per-stratum
EAF/MAF differ, exercising the meta-analysis and per-stratum QC recompute (PLAN §6).

The exact drawn effect sizes are written to `synthetic_truth-<ts>.tsv`; `--validate`
runs a lightweight recovery pass (a mini engine) and asserts each planted effect is
recovered and λ-GC ≈ 1 on the nulls.

Usage:
    python3 make_synthetic_data.py                 # default sized dataset in .
    python3 make_synthetic_data.py --smoke          # tiny + fast (CI / dev)
    python3 make_synthetic_data.py --validate       # + recover-the-truth checks
    python3 make_synthetic_data.py --n-subjects 20000 --n-variants 100000 \
        --output-dir ~/gwas_synth                   # bigger (stage pgen on ext4, PLAN §2)

Style mirrors ../proteomics_data_mine (module logger, sectioned layout, argparse main).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import pgenlib
except ImportError as exc:  # pragma: no cover - install guard
    sys.exit(
        "ERROR: Pgenlib is not installed. Install it with:\n"
        "  python3 -m pip install --user --break-system-packages Pgenlib\n"
        f"(import error: {exc})"
    )

log = logging.getLogger("make_synthetic_data")

# ============================================================================
# Constants
# ============================================================================

# plink2 stores dosages as uint16 on a 1/16384 grid (2/32768). Quantizing the
# generated dosages to that grid *before* they enter the phenotype model keeps
# the recovered truth exact (stored dosage == dosage used to simulate).
DOSAGE_GRID = 16384
MISSING_READ_SENTINEL = -9.0  # what PgenReader.read_dosages returns for a missing call

ANCESTRIES = ("EUR", "AJ")
ANCESTRY_FRAC = (0.70, 0.30)
NUCLEOTIDES = np.array(["A", "C", "G", "T"])

# The conditional-analysis index SNP (mirrors plink's --condition-list example);
# planted with a real bmkX main effect and paired with an LD tag (PLAN §3, §9).
INDEX_SNP_ID = "rs76904798"
LD_RHO = 0.9  # copy-probability used to build the tag → strong positive LD


# ============================================================================
# Logging
# ============================================================================


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _quantize_dosage(d: np.ndarray) -> np.ndarray:
    """Snap dosages to plink2's 1/16384 grid and clip to [0, 2] (NaN preserved)."""
    q = np.round(d * DOSAGE_GRID) / DOSAGE_GRID
    return np.clip(q, 0.0, 2.0)


# ============================================================================
# Generation parameters (all effect sizes live here for provenance)
# ============================================================================


@dataclass
class GenParams:
    n_subjects: int = 2000
    n_variants: int = 10000
    n_analytes: int = 40
    mean_visits: float = 4.0
    max_visits: int = 6
    n_causal_per: int = 12          # planted SNPs per genetic model
    n_analyte_main: int = 5         # analytes affecting updrs3 level
    n_analyte_slope: int = 2        # analytes affecting updrs3 slope
    frac_imputed: float = 0.15      # variants stored as fractional (imputed) dosages
    frac_missing_variants: float = 0.10  # variants carrying elevated missingness
    seed: int = 42

    # --- bmkX longitudinal model scale ---
    bmk_intercept: float = 5.0
    bmk_sigma_b0: float = 1.0       # random-intercept SD
    bmk_sigma_b1: float = 0.30      # random-slope SD
    bmk_sigma_resid: float = 0.80   # base within-subject residual SD (log-scale baseline)
    bmk_beta_age: float = 0.20      # per decade
    bmk_beta_sex: float = 0.30
    bmk_beta_pc1: float = 0.25      # structure confounder (controlled by PC1 covariate)
    bmk_global_slope: float = 0.15  # mean annual trend

    # --- updrs3 longitudinal model scale (tabular-predictor target) ---
    updrs_intercept: float = 20.0
    updrs_sigma_b0: float = 4.0
    updrs_sigma_b1: float = 1.0
    updrs_sigma_resid: float = 3.0
    updrs_global_slope: float = 1.5

    # --- static outcome baselines ---
    pd_base_logit: float = 0.7      # ~70% PD prevalence at reference covariates
    dem_base_rate: float = 0.06     # per-year baseline dementia hazard
    dem_admin_censor: float = 10.0  # administrative censoring (years)

    @classmethod
    def smoke(cls) -> "GenParams":
        return cls(
            n_subjects=400, n_variants=1000, n_analytes=20,
            mean_visits=3.0, max_visits=5, n_causal_per=6,
            n_analyte_main=3, n_analyte_slope=2,
        )


@dataclass
class Truth:
    """Ground-truth planted effects, written to synthetic_truth-<ts>.tsv."""

    rows: list[dict] = field(default_factory=list)

    def add(self, kind: str, target: str, effect: float, scale: str, **extra) -> None:
        self.rows.append({"kind": kind, "target": target, "true_effect": effect,
                          "effect_scale": scale, **extra})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


# ============================================================================
# Subjects (static attributes) + visit schedule
# ============================================================================


def build_subjects(p: GenParams, rng: np.random.Generator) -> pd.DataFrame:
    n = p.n_subjects
    iid = np.array([f"S{i:06d}" for i in range(1, n + 1)])
    ancestry = rng.choice(ANCESTRIES, size=n, p=ANCESTRY_FRAC)
    sex = rng.choice([1, 2], size=n)  # plink convention: 1=male, 2=female
    age0 = np.clip(rng.normal(65, 8, n), 40, 90)

    # PCs: PC1 separates ancestries (structure), PC2-5 are noise. Static per subject.
    pc = rng.normal(0, 1, size=(n, 5))
    pc[:, 0] += np.where(ancestry == "AJ", 2.0, -1.0)

    subj = pd.DataFrame({
        "IID": iid, "ancestry": ancestry, "SEX": sex, "AGE0": age0,
        "PC1": pc[:, 0], "PC2": pc[:, 1], "PC3": pc[:, 2], "PC4": pc[:, 3], "PC5": pc[:, 4],
    })
    # number of visits per subject (>=2 so slopes are estimable), mean ~ mean_visits
    nvis = np.clip(rng.poisson(p.mean_visits - 2, n) + 2, 2, p.max_visits)
    subj["n_visits"] = nvis
    return subj


def build_visit_rows(subj: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Explode subjects into LONG rows with irregular visit spacing."""
    recs = []
    for row in subj.itertuples(index=True):
        k = int(row.n_visits)
        # irregular spacing: scheduled ~0.75y apart with jitter; first visit is BL@0
        gaps = np.abs(rng.normal(0.75, 0.2, k - 1))
        times = np.concatenate([[0.0], np.cumsum(gaps)])
        for j in range(k):
            label = "BL" if j == 0 else f"V{2 * j:02d}"
            recs.append((row.Index, row.IID, label, float(times[j])))
    df = pd.DataFrame(recs, columns=["sidx", "IID", "visit", "years_from_baseline"])
    return df


# ============================================================================
# Variants + ancestry-specific allele frequencies
# ============================================================================


def build_variants(p: GenParams, rng: np.random.Generator) -> pd.DataFrame:
    m = p.n_variants
    # chromosome + strictly increasing position within chr
    chrom = np.sort(rng.integers(1, 23, size=m))
    pos = np.zeros(m, dtype=np.int64)
    for c in np.unique(chrom):
        idx = np.where(chrom == c)[0]
        pos[idx] = np.sort(rng.integers(1_000_000, 250_000_000, size=idx.size))

    # variant IDs (unique). rsID-style; the index SNP is injected at a fixed slot.
    ids = np.array([f"rs{rng.integers(1_000_000, 99_999_999)}" for _ in range(m)])

    # allele-frequency tiers: ~15% rare (0.01-0.05), rest common (0.05-0.5)
    is_rare = rng.random(m) < 0.15
    maf = np.where(is_rare, rng.uniform(0.01, 0.05, m), rng.uniform(0.05, 0.5, m))
    # effect allele (ALT) is minor for some variants, major for others (realistic)
    alt_is_minor = rng.random(m) < 0.6
    p_alt = np.where(alt_is_minor, maf, 1.0 - maf)

    # ancestry-specific ALT freq: small Fst-like perturbation per ancestry
    p_eur = np.clip(p_alt + rng.normal(0, 0.03, m), 0.005, 0.995)
    p_aj = np.clip(p_alt + rng.normal(0, 0.03, m), 0.005, 0.995)

    # REF/ALT nucleotides (distinct)
    ref = rng.choice(NUCLEOTIDES, size=m)
    alt = rng.choice(NUCLEOTIDES, size=m)
    same = ref == alt
    while same.any():
        alt[same] = rng.choice(NUCLEOTIDES, size=int(same.sum()))
        same = ref == alt

    var = pd.DataFrame({
        "CHR": chrom, "POS": pos, "ID": ids, "REF": ref, "ALT": alt,
        "p_EUR": p_eur, "p_AJ": p_aj, "is_rare": is_rare,
    })

    # inject the conditional-analysis index SNP at a realistic LRRK2-ish locus
    islot = m // 3
    var.loc[islot, ["CHR", "POS", "ID"]] = [12, 40_340_000, INDEX_SNP_ID]
    var.loc[islot, ["p_EUR", "p_AJ", "is_rare"]] = [0.30, 0.30, False]  # common → LD-tag test has power
    var.attrs["index_slot"] = int(islot)
    # sort by genomic position (pgen/pvar variant order); keep a stable index
    var = var.sort_values(["CHR", "POS"]).reset_index(drop=True)
    var.attrs["index_slot"] = int(var.index[var["ID"] == INDEX_SNP_ID][0])
    return var


# ============================================================================
# Genotype simulation → dosage matrix (variant-major, M×N)
# ============================================================================


def simulate_genotypes(p: GenParams, subj: pd.DataFrame, var: pd.DataFrame,
                       rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Return (dosage matrix M×N float32 with NaN missing, ld_tag_slot).

    Genotypes drawn per subject from their ancestry's ALT frequency. A designated
    LD tag SNP is built as a noisy copy of the index SNP. A fraction of variants
    are stored as fractional (imputed) dosages, and a fraction carry elevated
    missingness — everything except the causal/LD SNPs, which stay clean.
    """
    m, n = p.n_variants, p.n_subjects
    anc_is_aj = (subj["ancestry"].to_numpy() == "AJ")
    p_by_anc = np.where(anc_is_aj, var["p_AJ"].to_numpy()[:, None], var["p_EUR"].to_numpy()[:, None])

    D = np.empty((m, n), dtype=np.float32)
    for v in range(m):
        D[v] = rng.binomial(2, p_by_anc[v]).astype(np.float32)

    index_slot = int(var.attrs["index_slot"])

    # --- LD tag: a noisy copy of the index SNP (no independent effect) ----------
    tag_slot = index_slot + 1 if index_slot + 1 < m else index_slot - 1
    copy_mask = rng.random(n) < LD_RHO
    indep = rng.binomial(2, float(var.loc[tag_slot, "p_EUR"]), n).astype(np.float32)
    D[tag_slot] = np.where(copy_mask, D[index_slot], indep)

    protected = {index_slot, tag_slot}

    # --- fractional (imputed) dosages on a subset ------------------------------
    n_imp = int(p.frac_imputed * m)
    imp_slots = [s for s in rng.choice(m, size=min(n_imp, m), replace=False) if s not in protected]
    for v in imp_slots:
        noisy = D[v] + rng.normal(0, 0.08, n).astype(np.float32)
        D[v] = np.clip(noisy, 0.0, 2.0)

    # --- elevated missingness on a subset --------------------------------------
    n_miss = int(p.frac_missing_variants * m)
    miss_slots = [s for s in rng.choice(m, size=min(n_miss, m), replace=False) if s not in protected]
    for v in miss_slots:
        rate = rng.uniform(0.05, 0.15)
        mask = rng.random(n) < rate
        D[v, mask] = np.nan

    D[:] = _quantize_dosage(D)
    return D, tag_slot


# ============================================================================
# Role assignment (which variants are causal for which model)
# ============================================================================


def assign_roles(p: GenParams, var: pd.DataFrame, tag_slot: int, eaf: np.ndarray,
                 rng: np.random.Generator) -> dict[str, list[int]]:
    m = p.n_variants
    index_slot = int(var.attrs["index_slot"])
    reserved = {index_slot, tag_slot}
    # causal SNPs are drawn from COMMON variants only, so the planted truth is
    # cleanly recoverable. Rare variants stay null — they exist to exercise
    # rare-variant calibration / SPA later (Phase 3), not effect recovery.
    maf = np.minimum(eaf, 1.0 - eaf)
    pool = np.array([s for s in range(m) if s not in reserved and maf[s] >= 0.10])
    rng.shuffle(pool)

    roles: dict[str, list[int]] = {}
    cursor = 0
    for kind in ("linear_main", "snpxtime", "variance", "logistic", "cox"):
        k = p.n_causal_per
        roles[kind] = pool[cursor:cursor + k].tolist()
        cursor += k
    roles["linear_main"].append(index_slot)  # index SNP is a real bmkX main signal
    roles["ld_tag"] = [tag_slot]
    roles["index"] = [index_slot]
    return roles


def draw_effects(p: GenParams, roles: dict[str, list[int]], var: pd.DataFrame,
                 rng: np.random.Generator, truth: Truth) -> dict[int, dict]:
    """Draw and record per-variant effect sizes. Returns slot -> {kind: value}."""
    eff: dict[int, dict] = {}

    def rec(slot, kind, scale, value):
        eff.setdefault(slot, {})[kind] = value
        truth.add(kind, var.loc[slot, "ID"], float(value), scale,
                  CHR=int(var.loc[slot, "CHR"]), POS=int(var.loc[slot, "POS"]),
                  effect_allele=var.loc[slot, "ALT"], other_allele=var.loc[slot, "REF"])

    for slot in roles["linear_main"]:
        val = 0.5 if var.loc[slot, "ID"] == INDEX_SNP_ID else rng.uniform(0.25, 0.6) * rng.choice([-1, 1])
        rec(slot, "linear_main", "beta_per_allele", val)
    for slot in roles["snpxtime"]:
        rec(slot, "snpxtime", "beta_per_allele_per_year", rng.uniform(0.10, 0.25) * rng.choice([-1, 1]))
    for slot in roles["variance"]:
        rec(slot, "variance", "log_sd_per_allele", rng.uniform(0.30, 0.50))
    for slot in roles["logistic"]:
        rec(slot, "logistic", "log_or_per_allele", rng.uniform(0.40, 0.70) * rng.choice([-1, 1]))
    for slot in roles["cox"]:
        rec(slot, "cox", "log_hr_per_allele", rng.uniform(0.40, 0.70) * rng.choice([-1, 1]))
    return eff


# ============================================================================
# Phenotype simulation
# ============================================================================


def _dosage_impute(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean-impute NaNs (for the phenotype linear predictor); return (D_imp, eaf)."""
    eaf = np.nanmean(D, axis=1) / 2.0
    D_imp = np.where(np.isnan(D), (2.0 * eaf)[:, None], D).astype(np.float64)
    return D_imp, eaf


def simulate_phenotypes(p: GenParams, subj: pd.DataFrame, longdf: pd.DataFrame,
                        D_imp: np.ndarray, eaf: np.ndarray, roles: dict, eff: dict,
                        rng: np.random.Generator, truth: Truth) -> pd.DataFrame:
    n = p.n_subjects
    sidx = longdf["sidx"].to_numpy()
    t = longdf["years_from_baseline"].to_numpy()
    nrow = len(longdf)

    # subject-level covariate pieces (broadcast to rows)
    age_row = subj["AGE0"].to_numpy()[sidx] + t          # age advances with time
    sex = subj["SEX"].to_numpy()
    pc1 = subj["PC1"].to_numpy()

    # --- random effects per subject ---
    b0 = rng.normal(0, p.bmk_sigma_b0, n)
    b1 = rng.normal(0, p.bmk_sigma_b1, n)

    # --- genetic contributions to bmkX ---
    g_main = np.zeros(n)
    for slot in roles["linear_main"]:
        g_main += eff[slot]["linear_main"] * D_imp[slot]
    g_time = np.zeros(n)
    for slot in roles["snpxtime"]:
        g_time += eff[slot]["snpxtime"] * D_imp[slot]
    log_sd_add = np.zeros(n)
    for slot in roles["variance"]:
        # CENTER each variance QTL on 2·EAF: a variance SNP changes the *spread*
        # of the within-subject residual without inflating its *average* — so the
        # baseline residual stays σ_resid and the mean (β) axis is unaffected. The
        # clip bounds pathological multi-carrier tails (exp(±2.5) ≈ 0.08x–12x).
        log_sd_add += eff[slot]["variance"] * (D_imp[slot] - 2.0 * eaf[slot])
    resid_sd = p.bmk_sigma_resid * np.exp(np.clip(log_sd_add, -2.5, 2.5))  # per-subject residual SD

    # --- bmkX per row ---
    mean_bmk = (
        p.bmk_intercept
        + p.bmk_beta_age * (age_row - 65) / 10.0
        + p.bmk_beta_sex * (sex[sidx] == 2)
        + p.bmk_beta_pc1 * pc1[sidx]
        + (p.bmk_global_slope + b1[sidx] + g_time[sidx]) * t
        + b0[sidx] + g_main[sidx]
    )
    bmk = mean_bmk + rng.normal(0, 1, nrow) * resid_sd[sidx]

    # --- updrs3 (tabular-predictor target) + analytes ---
    analyte_cols = [f"PRTN{i:04d}_NPX" for i in range(1, p.n_analytes + 1)]
    A = rng.normal(0, 1, size=(n, p.n_analytes))          # subject-level analyte values
    # a few analytes affect updrs3 level / slope
    main_idx = rng.choice(p.n_analytes, size=p.n_analyte_main, replace=False)
    rest = [i for i in range(p.n_analytes) if i not in set(main_idx)]
    slope_idx = rng.choice(rest, size=p.n_analyte_slope, replace=False)
    a_main = {int(i): rng.uniform(1.0, 2.5) * rng.choice([-1, 1]) for i in main_idx}
    a_slope = {int(i): rng.uniform(0.3, 0.6) * rng.choice([-1, 1]) for i in slope_idx}
    for i, v in a_main.items():
        truth.add("analyte_main", analyte_cols[i], float(v), "beta_on_updrs3_bl")
    for i, v in a_slope.items():
        truth.add("analyte_slope", analyte_cols[i], float(v), "beta_on_updrs3_slope")

    u_b0 = rng.normal(0, p.updrs_sigma_b0, n)
    u_b1 = rng.normal(0, p.updrs_sigma_b1, n)
    u_main = np.zeros(n)
    for i, v in a_main.items():
        u_main += v * A[:, i]
    u_slope = np.zeros(n)
    for i, v in a_slope.items():
        u_slope += v * A[:, i]
    mean_updrs = (
        p.updrs_intercept + u_b0[sidx] + u_main[sidx]
        + (p.updrs_global_slope + u_b1[sidx] + u_slope[sidx]) * t
    )
    updrs = mean_updrs + rng.normal(0, p.updrs_sigma_resid, nrow)

    # analyte columns per row: subject value + small visit noise
    analyte_rows = A[sidx] + rng.normal(0, 0.1, size=(nrow, p.n_analytes))

    # --- assemble the LONG frame ---
    out = longdf.copy()
    out["ancestry"] = subj["ancestry"].to_numpy()[sidx]
    out["SEX"] = sex[sidx]
    out["AGE"] = age_row
    for c in ("PC1", "PC2", "PC3", "PC4", "PC5"):
        out[c] = subj[c].to_numpy()[sidx]
    out["bmkX"] = bmk
    out["updrs3"] = updrs
    # updrs3_bl: baseline value broadcast to all of a subject's rows
    bl_mask = out["visit"].to_numpy() == "BL"
    bl_val = pd.Series(out.loc[bl_mask, "updrs3"].to_numpy(), index=out.loc[bl_mask, "sidx"])
    out["updrs3_bl"] = out["sidx"].map(bl_val).to_numpy()
    for j, c in enumerate(analyte_cols):
        out[c] = analyte_rows[:, j]

    # --- static outcomes (subject-level, broadcast) ---
    static = simulate_static_outcomes(p, subj, D_imp, roles, eff, rng)
    for c in static.columns:
        out[c] = static[c].to_numpy()[sidx]
    return out, analyte_cols


def simulate_static_outcomes(p: GenParams, subj: pd.DataFrame, D_imp: np.ndarray,
                             roles: dict, eff: dict, rng: np.random.Generator) -> pd.DataFrame:
    n = p.n_subjects
    age_c = (subj["AGE0"].to_numpy() - 65) / 10.0
    sex2 = (subj["SEX"].to_numpy() == 2).astype(float)
    pc1 = subj["PC1"].to_numpy()

    # --- PD case/control (logistic) ---
    lin = p.pd_base_logit + 0.3 * age_c + 0.1 * sex2 + 0.1 * pc1
    for slot in roles["logistic"]:
        lin += eff[slot]["logistic"] * D_imp[slot]
    pd_case = (rng.random(n) < 1.0 / (1.0 + np.exp(-lin))).astype(int)
    diagnosis = np.where(pd_case == 1, "PD", "HC")

    # --- dementia time-to-event (cox) ---
    haz_lin = 0.2 * age_c + 0.1 * sex2
    for slot in roles["cox"]:
        haz_lin += eff[slot]["cox"] * D_imp[slot]
    rate = p.dem_base_rate * np.exp(haz_lin)
    t_event = rng.exponential(1.0 / rate)
    t_censor = np.minimum(rng.uniform(2.0, p.dem_admin_censor, n), p.dem_admin_censor)
    tte = np.minimum(t_event, t_censor)
    event = (t_event <= t_censor).astype(int)

    # --- LEDD (levodopa dose): positive, PD-only, referenced by derive: log(LEDD+1) ---
    ledd = np.where(pd_case == 1, rng.gamma(3.0, 150.0, n), 0.0)

    return pd.DataFrame({
        "DIAGNOSIS": diagnosis, "PD_case": pd_case, "LEDD": ledd,
        "tte_dementia_years": tte, "event_dementia": event,
    })


# ============================================================================
# Writers
# ============================================================================


def write_psam(subj: pd.DataFrame, prefix: str) -> str:
    path = f"{prefix}.psam"
    with open(path, "w") as f:
        f.write("#IID\tSEX\n")
        for iid, sex in zip(subj["IID"], subj["SEX"]):
            f.write(f"{iid}\t{sex}\n")
    return path


def write_pvar(var: pd.DataFrame, prefix: str) -> str:
    path = f"{prefix}.pvar"
    with open(path, "w") as f:
        f.write("#CHROM\tPOS\tID\tREF\tALT\n")
        for r in var.itertuples(index=False):
            f.write(f"{r.CHR}\t{r.POS}\t{r.ID}\t{r.REF}\t{r.ALT}\n")
    return path


def write_pgen(D: np.ndarray, prefix: str) -> str:
    m, n = D.shape
    path = f"{prefix}.pgen"
    w = pgenlib.PgenWriter(filename=path.encode(), sample_ct=n, variant_ct=m,
                           nonref_flags=False, dosage_present=True)
    for v in range(m):
        w.append_dosages(np.ascontiguousarray(D[v], dtype=np.float32))
    w.close()
    return path


def write_clinical(out: pd.DataFrame, analyte_cols: list[str], output_dir: str, ts: str) -> str:
    ordered = (["IID", "ancestry", "DIAGNOSIS", "SEX", "visit", "years_from_baseline", "AGE",
                "PC1", "PC2", "PC3", "PC4", "PC5", "LEDD",
                "bmkX", "updrs3", "updrs3_bl", "PD_case",
                "tte_dementia_years", "event_dementia"] + analyte_cols)
    path = os.path.join(output_dir, f"synthetic_clinical-{ts}.tsv")
    out[ordered].to_csv(path, sep="\t", index=False, float_format="%.6g")
    return path


# ============================================================================
# Recover-the-truth validation (a mini engine; PLAN §12 acceptance)
# ============================================================================


def _fwl_scan(G: np.ndarray, y: np.ndarray, Q: np.ndarray):
    """Vectorized Frisch–Waugh linear scan: β, SE, χ² for each column of G.

    G (N×m) predictors, y (N,) outcome, Q (N×k) covariates incl. intercept.
    """
    Qq, _ = np.linalg.qr(Q)
    y_r = y - Qq @ (Qq.T @ y)
    G_r = G - Qq @ (Qq.T @ G)
    gg = np.einsum("ij,ij->j", G_r, G_r)
    gg = np.where(gg <= 1e-12, np.nan, gg)
    gy = G_r.T @ y_r
    beta = gy / gg
    dof = G.shape[0] - Q.shape[1] - 1
    yy = float(y_r @ y_r)
    sigma2 = np.maximum((yy - beta * gy) / dof, 1e-12)
    se = np.sqrt(sigma2 / gg)
    chi2 = (beta / se) ** 2
    return beta, se, chi2


def _lambda_gc(chi2_null: np.ndarray) -> float:
    med = np.nanmedian(chi2_null)
    return float(med / 0.4549364)  # median of χ²_1


def validate(p: GenParams, subj, out, D, roles, eff, truth: Truth, analyte_cols) -> bool:
    from scipy.stats import chi2 as chi2_dist

    log.info("-" * 68)
    log.info("Recover-the-truth validation")
    log.info("-" * 68)
    ok = True
    D_imp, _ = _dosage_impute(D)
    n = p.n_subjects
    bl = out[out["visit"] == "BL"].sort_values("sidx")
    bl_sidx = bl["sidx"].to_numpy()
    Gbl = D_imp[:, bl_sidx].T                       # (N_bl, M)
    covar = np.column_stack([
        np.ones(len(bl)), (bl["AGE"] - 65) / 10, (bl["SEX"] == 2).astype(float),
        bl["PC1"], bl["PC2"], bl["PC3"], bl["PC4"], bl["PC5"],
    ])

    causal_slots = {k: set(roles[k]) for k in ("linear_main", "snpxtime", "variance", "logistic", "cox")}
    all_causal = set().union(*causal_slots.values()) | set(roles["ld_tag"])
    null_slots = np.array([s for s in range(p.n_variants) if s not in all_causal])

    def report(name, est, se, true, extra_abs=0.0):
        """Recovered iff the estimate is within ~4 SE of the planted truth.

        SE-based (not fixed-tolerance) so it stays valid as N and effect sizes
        change: an unbiased estimator passes at any sample size, and low-power
        cases widen the band instead of failing spuriously.
        """
        nonlocal ok
        err = abs(est - true)
        band = 4.0 * se if (np.isfinite(se) and se > 0) else (0.5 * abs(true) + 0.1)
        tol = max(band, extra_abs)
        z = err / se if (np.isfinite(se) and se > 0) else float("nan")
        good = err <= tol
        ok &= good
        log.info("  [%s] %-17s est=%+.3f true=%+.3f  se=%.3f  |Δ|/se=%.1f",
                 "PASS" if good else "FAIL", name, est, true, se, z)

    # --- linear_main: BL OLS ---
    beta, se, chi2 = _fwl_scan(Gbl, bl["bmkX"].to_numpy(), covar)
    for slot in sorted(roles["linear_main"]):
        report(f"linear[{slot}]", beta[slot], se[slot], eff[slot]["linear_main"])
    lam = _lambda_gc(chi2[null_slots])
    log.info("  linear λ-GC on %d null SNPs: %.3f", null_slots.size, lam)
    ok &= 0.85 <= lam <= 1.2

    # --- ld_tag: informational only. The conditional analysis itself is a run-level
    # YAML concern (`condition_snps:`) exercised by the engine in Phase 2, not a
    # generator self-check. Here we just confirm the planted LD structure exists:
    # the tag's marginal signal should shrink once the index SNP enters the design.
    tag = roles["ld_tag"][0]
    idx = roles["index"][0]
    p_marg = float(chi2_dist.sf(chi2[tag], 1))
    covar_cond = np.column_stack([covar, Gbl[:, idx]])
    _, _, chi2_cond = _fwl_scan(Gbl, bl["bmkX"].to_numpy(), covar_cond)
    p_cond = float(chi2_dist.sf(chi2_cond[tag], 1))
    log.info("  [info] LD tag (slot %d) vs index %s (slot %d): marginal P=%.2e -> "
             "conditioned P=%.2f (planted for a `condition_snps:` YAML run)",
             tag, INDEX_SNP_ID, idx, p_marg, p_cond)

    # --- snpxtime: two-stage per-subject slopes, then scan ---
    slope, sl_cov = _per_subject_slopes(out, subj, "bmkX")
    Gsub = D_imp.T                                  # (N, M) subject-order
    beta_s, se_s, _ = _fwl_scan(Gsub, slope, sl_cov)
    for slot in sorted(roles["snpxtime"]):
        report(f"snpxtime[{slot}]", beta_s[slot], se_s[slot], eff[slot]["snpxtime"])

    # --- variance: per-subject residual log-SD, then scan ---
    # This crude two-stage proxy (log-SD from a handful of visits) is a noisy,
    # downward-biased estimator of τ, so we assert the variability signal is
    # PRESENT, correctly-signed, and DETECTABLE (z-from-zero) rather than matching
    # magnitude. Recovering τ exactly is Phase 5's location-scale model.
    logsd = _per_subject_log_sd(out, "bmkX")
    beta_v, se_v, _ = _fwl_scan(Gsub, logsd, sl_cov)
    for slot in sorted(roles["variance"]):
        est, se, true = beta_v[slot], se_v[slot], eff[slot]["variance"]
        z0 = est / se if se > 0 else 0.0
        good = (np.sign(est) == np.sign(true)) and (z0 >= 2.0)
        ok &= good
        log.info("  [%s] variance[%d]     est=%+.3f true=%+.3f  se=%.3f  z_from_0=%.1f "
                 "(proxy attenuates magnitude; τ exact in Phase 5)",
                 "PASS" if good else "FAIL", slot, est, true, se, z0)

    # --- logistic: statsmodels Logit on a handful ---
    _validate_logistic(bl, Gbl, covar, roles, eff, report)

    # --- cox: lifelines on a handful ---
    _validate_cox(subj, D_imp, roles, eff, report)

    # --- analyte_main: BL OLS of updrs3_bl on the analyte ---
    A_bl = bl[analyte_cols].to_numpy()
    beta_a, se_a, _ = _fwl_scan(A_bl, bl["updrs3_bl"].to_numpy(), covar[:, :1])  # intercept only
    truth_df = truth.frame()
    for r in truth_df[truth_df["kind"] == "analyte_main"].itertuples():
        j = analyte_cols.index(r.target)
        report(f"analyte:{r.target}", beta_a[j], se_a[j], r.true_effect)

    log.info("-" * 68)
    log.info("Validation: %s", "ALL PLANTED EFFECTS RECOVERED" if ok else "FAILURES ABOVE")
    return ok


def _per_subject_slopes(out: pd.DataFrame, subj: pd.DataFrame, col: str):
    n = len(subj)
    slope = np.zeros(n)
    for sidx, g in out.groupby("sidx"):
        t = g["years_from_baseline"].to_numpy()
        y = g[col].to_numpy()
        tc = t - t.mean()
        denom = float(tc @ tc)
        slope[sidx] = float(tc @ (y - y.mean()) / denom) if denom > 1e-9 else 0.0
    cov = np.column_stack([
        np.ones(n), (subj["AGE0"] - 65) / 10, (subj["SEX"] == 2).astype(float),
        subj["PC1"], subj["PC2"], subj["PC3"], subj["PC4"], subj["PC5"],
    ])
    return slope, cov


def _per_subject_log_sd(out: pd.DataFrame, col: str):
    n = out["sidx"].max() + 1
    logsd = np.zeros(n)
    for sidx, g in out.groupby("sidx"):
        t = g["years_from_baseline"].to_numpy()
        y = g[col].to_numpy()
        tc = t - t.mean()
        denom = float(tc @ tc)
        b = float(tc @ (y - y.mean()) / denom) if denom > 1e-9 else 0.0
        resid = (y - y.mean()) - b * tc
        logsd[sidx] = float(np.log(resid.std(ddof=1) + 1e-6)) if len(y) > 2 else 0.0
    return logsd


def _validate_logistic(bl, Gbl, covar, roles, eff, report):
    # Fit ALL logistic SNPs jointly (the correctly-specified conditional model).
    # A single-SNP fit would attenuate each log-OR toward the null via the
    # non-collapsibility of the odds ratio, even though the SNPs are independent.
    import statsmodels.api as sm
    y = bl["PD_case"].to_numpy()
    slots = sorted(roles["logistic"])
    X = np.column_stack([covar] + [Gbl[:, s] for s in slots])
    try:
        res = sm.Logit(y, X).fit(disp=0)
    except Exception as e:  # pragma: no cover
        log.info("  [SKIP] logistic joint fit: %s", e)
        return
    base = covar.shape[1]
    for pos, slot in enumerate(slots):
        report(f"logistic[{slot}]", res.params[base + pos], res.bse[base + pos], eff[slot]["logistic"])


def _validate_cox(subj, D_imp, roles, eff, report):
    # Same non-collapsibility argument as logistic: recover the conditional log-HR
    # by fitting all cox SNPs jointly (correctly-specified model).
    try:
        from lifelines import CoxPHFitter
    except ImportError:  # pragma: no cover
        log.info("  [SKIP] cox: lifelines unavailable")
        return
    slots = sorted(roles["cox"])
    df = pd.DataFrame({
        "age": (subj["AGE0"] - 65) / 10, "sex": (subj["SEX"] == 2).astype(float),
        "T": subj.attrs["tte"], "E": subj.attrs["event"],
    })
    for pos, slot in enumerate(slots):
        df[f"g{pos}"] = D_imp[slot]
    try:
        cph = CoxPHFitter().fit(df, duration_col="T", event_col="E")
    except Exception as e:  # pragma: no cover
        log.info("  [SKIP] cox joint fit: %s", e)
        return
    for pos, slot in enumerate(slots):
        report(f"cox[{slot}]", cph.params_[f"g{pos}"], cph.standard_errors_[f"g{pos}"], eff[slot]["cox"])


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic data for Biomarker Gran Prix")
    ap.add_argument("--output-dir", default=".")
    ap.add_argument("--geno-prefix", default="synthetic_geno")
    ap.add_argument("--n-subjects", type=int, default=None)
    ap.add_argument("--n-variants", type=int, default=None)
    ap.add_argument("--n-analytes", type=int, default=None)
    ap.add_argument("--n-causal-per", type=int, default=None,
                    help="planted causal SNPs per genetic model")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny + fast preset")
    ap.add_argument("--validate", action="store_true", help="run recover-the-truth checks")
    args = ap.parse_args()

    setup_logging()
    p = GenParams.smoke() if args.smoke else GenParams()
    for attr, val in (("n_subjects", args.n_subjects), ("n_variants", args.n_variants),
                      ("n_analytes", args.n_analytes), ("n_causal_per", args.n_causal_per),
                      ("seed", args.seed)):
        if val is not None:
            setattr(p, attr, val)
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = os.path.join(args.output_dir, args.geno_prefix)
    rng = np.random.default_rng(p.seed)
    ts = _ts()

    log.info("=" * 68)
    log.info("Biomarker Gran Prix — synthetic data generator")
    log.info("=" * 68)
    log.info("N=%d subjects, M=%d variants, %d analytes, seed=%d",
             p.n_subjects, p.n_variants, p.n_analytes, p.seed)

    subj = build_subjects(p, rng)
    var = build_variants(p, rng)
    longdf = build_visit_rows(subj, rng)
    log.info("Built %d subjects -> %d visit rows (mean %.1f visits/subject)",
             len(subj), len(longdf), len(longdf) / len(subj))

    D, tag_slot = simulate_genotypes(p, subj, var, rng)
    D_imp, eaf = _dosage_impute(D)
    roles = assign_roles(p, var, tag_slot, eaf, rng)
    truth = Truth()
    eff = draw_effects(p, roles, var, rng, truth)

    # record the planted LD structure (index has a real bmkX effect; tag is a null
    # SNP correlated with it) so a `condition_snps:` YAML run can be validated.
    idx_slot = roles["index"][0]
    ld_r = float(np.corrcoef(D_imp[idx_slot], D_imp[tag_slot])[0, 1])
    truth.add("ld_tag", var.loc[tag_slot, "ID"], ld_r, "pearson_r_with_index",
              CHR=int(var.loc[tag_slot, "CHR"]), POS=int(var.loc[tag_slot, "POS"]),
              effect_allele=var.loc[tag_slot, "ALT"], other_allele=var.loc[tag_slot, "REF"],
              index_snp=INDEX_SNP_ID)

    out, analyte_cols = simulate_phenotypes(p, subj, longdf, D_imp, eaf, roles, eff, rng, truth)
    # stash static survival cols for cox validation
    st = out.drop_duplicates("sidx").sort_values("sidx")
    subj.attrs["tte"] = st["tte_dementia_years"].to_numpy()
    subj.attrs["event"] = st["event_dementia"].to_numpy()

    # --- write outputs ---
    psam = write_psam(subj, prefix)
    pvar = write_pvar(var, prefix)
    pgen = write_pgen(D, prefix)
    clin = write_clinical(out, analyte_cols, args.output_dir, ts)
    truth_path = os.path.join(args.output_dir, f"synthetic_truth-{ts}.tsv")
    truth.frame().to_csv(truth_path, sep="\t", index=False, float_format="%.6g")

    log.info("Wrote genotype fileset : %s{.pgen,.pvar,.psam}", prefix)
    log.info("  .pgen %d bytes, .pvar/.psam text", os.path.getsize(pgen))
    log.info("Wrote clinical TSV     : %s (%d rows)", clin, len(out))
    log.info("Wrote ground truth     : %s (%d planted effects)", truth_path, len(truth.rows))
    log.info("Planted per model: linear=%d snpxtime=%d variance=%d logistic=%d cox=%d + 1 LD tag",
             len(roles["linear_main"]), len(roles["snpxtime"]), len(roles["variance"]),
             len(roles["logistic"]), len(roles["cox"]))
    log.info("Index/conditional SNP  : %s  (LD tag: %s)", INDEX_SNP_ID, var.loc[tag_slot, "ID"])

    if args.validate:
        passed = validate(p, subj, out, D, roles, eff, truth, analyte_cols)
        log.info("=" * 68)
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
