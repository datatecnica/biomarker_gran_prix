#!/usr/bin/env python3
"""gwas.py — core association engine for Biomarker Gran Prix (PLAN §4, §5, §6).

YAML-driven batch association scan: one config → many runs → one CSV per run×stratum.
The pipeline for each run×stratum is the reference's skeleton, generalized to a
genotype predictor axis and a GPU-batched scan:

    clinical load → global prefilter → derive: columns → per-stratum subset →
    align to .psam → fit null once (CPU) → stream predictor blocks to the device
    → per-predictor β/SE/P + effect-allele/EAF/N → QC + genome-wide sig + λ-GC →
    results/<run>-<stratum>-<timestamp>.csv

The looped predictor is either **SNPs streamed from a plink2 fileset**
(`predictor_source: genetic`, default) or **tabular columns** selected by list or
pattern (`predictor_source: tabular`) — every model runs identically over both, only
the identity/QC output columns differ (PLAN §5, §6).

Phase 2 implements the `linear` model (closed-form β/SE/P via residualize-then-scan,
`gpu.scan_linear`), plus `derive:`, `condition_snps:`, and both predictor modes.
Later phases add logistic / cox / gallop / slope / trajgwas on the same skeleton.

Usage:
    python3 gwas.py --config smoke_test.yaml [--run name ...]

Design mirrors ../proteomics_data_mine/regressions.py (config/strata/CSV/λ/Bonferroni).
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import sys
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, t as t_dist

import gpu
import genoio
import robust

log = logging.getLogger("gwas")

# ============================================================================
# Constants / defaults (PLAN §9 "sensible defaults")
# ============================================================================

ALPHA = 0.05
GENOMEWIDE_SIG = 5e-8
DEFAULTS = {
    "predictor_source": "genetic",
    "predictor_at": "all",
    "maf_min": 0.01,
    "mac_min": 20,
    "missing_max": 0.05,
    "min_n": 30,
    "device": "auto",
    "max_cpus": None,
    "max_ram_gb": None,
    "max_gpu_mem_gb": None,
    "block_size": "auto",
    "prefetch": True,
    # SPA (two-sided; GWAS is exploratory) is applied to the tail — a small score-P
    # where the normal approx is unreliable, OR rare variants where it is poor across
    # the whole band (à la SAIGE/SPACox). MAF ≤ 0.05 by default keeps the rare band
    # calibrated; both are per-run overridable.
    "spa_p_cutoff": 0.01,
    "spa_maf_cutoff": 0.05,
    # Standardization of NON-GENETIC (tabular / non-plink) predictors, mirroring
    # ../proteomics_data_mine for reproducibility: z-score continuous predictors,
    # skip {0,1} binary and {0,1,2} dosage-coded ones unless explicitly enabled.
    # Genetic (plink) dosages are never z-scored here (raw → β per allele, PLAN §6).
    # All three are per-run overridable.
    "zscore_predictors": True,   # master switch for non-genetic predictor z-scoring
    "zscore_binary": False,      # also z-score {0,1} predictors
    "zscore_dosage": False,      # also z-score {0,1,2} dosage-coded predictors
}


def resolve_zscore_opts(spec: dict, defaults: dict) -> tuple[bool, bool, bool]:
    """(enabled, zscore_binary, zscore_dosage) for non-genetic predictors, spec>defaults."""
    def pick(key):
        return spec.get(key, defaults.get(key, DEFAULTS[key]))
    return pick("zscore_predictors"), pick("zscore_binary"), pick("zscore_dosage")

# Stable output schema core (PLAN §6). Model-specific term columns
# (beta_SNPxTIME, tau_var, …) are added by their phases.
OUTPUT_COLUMNS = [
    "predictor", "CHR", "POS", "effect_allele", "other_allele",
    "EAF", "MAF", "missing_rate", "predictor_mean", "predictor_sd", "zscored",
    "N", "n_obs", "n_subjects", "n_events", "beta", "SE", "P",
    "beta_SNP", "SE_SNP", "P_SNP", "beta_SNPxTIME", "SE_SNPxTIME", "P_SNPxTIME",
    "beta_mean", "SE_mean", "P_mean", "tau_var", "SE_var", "P_var", "P_joint",
    "model_type", "test", "genomewide_sig",
    "bonferroni_threshold", "significant", "conditioned_on", "strata",
]

# ============================================================================
# Logging / config
# ============================================================================


def setup_logging(log_file: str | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers, force=True)


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "runs" not in cfg or not isinstance(cfg["runs"], list):
        sys.exit(f"ERROR: config {path} must contain a 'runs' list")
    cfg.setdefault("defaults", {})
    return cfg


def get_default(defaults: dict, key: str) -> Any:
    return defaults.get(key, DEFAULTS.get(key))


def resolve_input_path(input_glob: str, cwd: str) -> str:
    hits = sorted(glob.glob(os.path.join(cwd, input_glob)))
    if not hits:
        sys.exit(f"ERROR: no files match clinical_file={input_glob!r} in {cwd}")
    return hits[-1]


def load_clinical(path: str) -> pd.DataFrame:
    log.info(f"Loading clinical: {os.path.basename(path)}")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    log.info(f"  {df.shape[0]:,} rows x {df.shape[1]:,} cols")
    return df


# ============================================================================
# Prefilter
# ============================================================================


def apply_prefilter(df: pd.DataFrame, prefilter: list[str] | None) -> pd.DataFrame:
    if not prefilter:
        return df
    n0 = len(df)
    for expr in prefilter:
        df = df.query(expr)
        log.info(f"  prefilter {expr!r}: {len(df):,} rows remain")
    log.info(f"  prefilter kept {len(df):,}/{n0:,} rows")
    return df.copy()


# ============================================================================
# derive: block (row-wise expressions, per-subject aggregates, genotype pulls)
# ============================================================================

_EVAL_NS = {"log": np.log, "log1p": np.log1p, "exp": np.exp, "sqrt": np.sqrt,
            "abs": np.abs, "log10": np.log10, "minimum": np.minimum, "maximum": np.maximum}

_AGG_RE = re.compile(r"^\s*(baseline_of|slope_of|delta)\s*\((.*)\)\s*$")
_SNP_RE = re.compile(r"^\s*snp\s*\(\s*([^)]+?)\s*\)\s*$")


def apply_derive(df: pd.DataFrame, derive: dict | None, defaults: dict,
                 geno: "genoio.GenoReader | None") -> pd.DataFrame:
    if not derive:
        return df
    df = df.copy()
    sid = defaults.get("sample_id_col", "IID")
    tcol = defaults.get("time_col")
    for name, expr in derive.items():
        m_snp = _SNP_RE.match(expr)
        m_agg = _AGG_RE.match(expr)
        if m_snp:
            df[name] = _derive_snp(df, m_snp.group(1).strip(), sid, geno)
        elif m_agg:
            df[name] = _derive_aggregate(df, m_agg.group(1), m_agg.group(2), sid, tcol)
        else:
            df[name] = _derive_rowwise(df, expr)
        log.info(f"  derive {name} = {expr!r}")
    return df


def _derive_rowwise(df: pd.DataFrame, expr: str) -> pd.Series:
    ns = {c: df[c] for c in df.columns}
    ns.update(_EVAL_NS)
    try:
        return pd.Series(eval(expr, {"__builtins__": {}}, ns), index=df.index)
    except Exception as e:
        sys.exit(f"ERROR: derive expression {expr!r} failed: {e}")


def _derive_aggregate(df: pd.DataFrame, fn: str, arg: str, sid: str, tcol: str | None) -> pd.Series:
    if fn == "slope_of":
        if "~" not in arg:
            sys.exit(f"ERROR: slope_of expects 'col ~ time', got {arg!r}")
        ycol, xcol = (s.strip() for s in arg.split("~", 1))
    else:
        ycol, xcol = arg.strip(), tcol
    if ycol not in df.columns:
        sys.exit(f"ERROR: derive {fn}({arg}) references unknown column {ycol!r}")
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, g in df.groupby(sid, sort=False):
        y = pd.to_numeric(g[ycol], errors="coerce").to_numpy()
        if fn == "baseline_of":
            order = np.argsort(pd.to_numeric(g[xcol], errors="coerce").to_numpy()) if xcol else np.arange(len(g))
            val = y[order[0]] if len(y) else np.nan
        elif fn == "delta":
            order = np.argsort(pd.to_numeric(g[xcol], errors="coerce").to_numpy()) if xcol else np.arange(len(g))
            yo = y[order]
            val = yo[-1] - yo[0] if np.isfinite(yo).sum() >= 2 else np.nan
        else:  # slope_of
            x = pd.to_numeric(g[xcol], errors="coerce").to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 2 and np.ptp(x[ok]) > 0:
                xc = x[ok] - x[ok].mean()
                val = float(xc @ (y[ok] - y[ok].mean()) / (xc @ xc))
            else:
                val = np.nan
        out.loc[g.index] = val
    return out


def _derive_snp(df: pd.DataFrame, vid: str, sid: str, geno: "genoio.GenoReader | None") -> pd.Series:
    if geno is None:
        sys.exit(f"ERROR: derive snp({vid}) requires a genotype_prefix")
    uniq = df[sid].astype(str).drop_duplicates().tolist()
    pos, kept = geno.align_samples(uniq)
    dos, found = geno.read_ids([vid], pos)
    if not found:
        sys.exit(f"ERROR: derive snp({vid}): variant {vid!r} not in fileset")
    d = dos[0]
    d = np.where(np.isnan(d), np.nanmean(d), d)  # mean-impute the pulled dosage
    mp = dict(zip(kept, d))
    return df[sid].astype(str).map(mp).astype(float)


# ============================================================================
# Predictor / covariate resolution
# ============================================================================


def resolve_covariates(defaults_cov: list[str], add: list[str] | None,
                       drop: list[str] | None) -> list[str]:
    out = list(defaults_cov or [])
    for c in add or []:
        if c not in out:
            out.append(c)
    for c in drop or []:
        if c in out:
            out.remove(c)
    return out


def resolve_tabular_predictors(df: pd.DataFrame, spec: Any) -> list[str]:
    """Select tabular predictor columns by explicit list or {prefix|suffix|regex}."""
    numeric = set(df.select_dtypes(include=[np.number]).columns)
    if isinstance(spec, list):
        missing = [c for c in spec if c not in df.columns]
        if missing:
            sys.exit(f"ERROR: predictors not in clinical file: {missing}")
        return spec
    if isinstance(spec, dict):
        if "prefix" in spec:
            cols = [c for c in df.columns if c.startswith(spec["prefix"])]
        elif "suffix" in spec:
            cols = [c for c in df.columns if c.endswith(spec["suffix"])]
        elif "regex" in spec:
            rx = re.compile(spec["regex"])
            cols = [c for c in df.columns if rx.search(c)]
        else:
            sys.exit(f"ERROR: predictors dict must have prefix|suffix|regex, got {spec}")
        cols = [c for c in cols if c in numeric]
        if not cols:
            sys.exit(f"ERROR: no numeric columns matched predictors={spec}")
        return cols
    sys.exit(f"ERROR: predictors must be a list or {{prefix|suffix|regex}}, got {spec!r}")


def _predictor_kind(s: pd.Series) -> str:
    vals = set(pd.to_numeric(s, errors="coerce").dropna().unique())
    if not vals:
        return "other"
    if vals <= {0.0, 1.0}:
        return "binary"
    if vals <= {0.0, 1.0, 2.0}:
        return "dosage"
    return "other"


def _should_zscore(kind: str, zb: bool, zd: bool) -> bool:
    return {"binary": zb, "dosage": zd}.get(kind, True)


# ============================================================================
# Numerical filter / Bonferroni / λ-GC  (reference parity)
# ============================================================================


def filter_numerical_failures(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return df
    before = len(df)
    # P-centric: P must be a valid probability. A finite β additionally requires a valid
    # SE. β/SE are legitimately NaN for the 2-df JOINT test (no single effect), so we do
    # NOT drop on a NaN β when P is valid.
    P = pd.to_numeric(df["P"], errors="coerce")
    bad = ~np.isfinite(P) | (P <= 0) | (P > 1)
    has_beta = np.isfinite(pd.to_numeric(df["beta"], errors="coerce"))
    bad |= has_beta & (~np.isfinite(pd.to_numeric(df["SE"], errors="coerce")) | (df["SE"] <= 0))
    n_bad = int(bad.sum())
    if n_bad:
        log.warning(f"  {label}: dropped {n_bad}/{before} numerical failures")
    return df.loc[~bad].copy()


def apply_bonferroni(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    thresh = ALPHA / len(df)
    df["bonferroni_threshold"] = thresh
    df["significant"] = df["P"] < thresh
    return df


def lambda_gc(pvals: pd.Series) -> tuple[float, int]:
    p = pd.Series(pvals).dropna()
    p = p[(p > 0) & (p <= 1)]
    if len(p) == 0:
        return float("nan"), 0
    lam = float(np.median(chi2.ppf(1 - p, df=1)) / chi2.ppf(0.5, df=1))
    return lam, len(p)


# ============================================================================
# Analysis-frame assembly + alignment
# ============================================================================


def build_stratum_frame(clinical: pd.DataFrame, spec: dict, defaults: dict,
                        stratum: Any) -> pd.DataFrame:
    df = clinical
    strata_col = spec.get("strata_col", defaults.get("strata_col"))
    if strata_col and stratum is not None:
        df = df[df[strata_col].astype(str) == str(stratum)]
    sf = spec.get("sample_filter")
    if sf:
        df = df.query(sf)
    return df.copy()


def _require_columns(df: pd.DataFrame, cols: list[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {ctx}: columns not found: {missing}")


# ============================================================================
# Genetic scan — shared machinery (align → design → QC block stream)
# ============================================================================

from dataclasses import dataclass


@dataclass
class _GPrep:
    sub: pd.DataFrame          # aligned analysis frame (index = IID, .psam order)
    sample_idx: np.ndarray     # .pgen positions of kept subjects
    n: int
    covars: list
    Q: np.ndarray              # design: intercept + covariates (+ conditioning SNPs)
    conditioned_on: str
    cc: Any                    # resolved compute config


def _prep_genetic(df, spec, defaults, geno, stratum, need_cols) -> "_GPrep | None":
    """Align to .psam and build the covariate design shared by every genetic model."""
    sid = defaults.get("sample_id_col", "IID")
    covars = resolve_covariates(defaults.get("covariates", []),
                                spec.get("add_covariates"), spec.get("drop_covariates"))
    _require_columns(df, [sid] + need_cols + covars, f"{spec['name']}|{stratum}")
    sub = df.dropna(subset=[sid] + need_cols + covars)
    if sub[sid].duplicated().any():
        nd = int(sub[sid].duplicated().sum())
        log.warning(f"  {spec['name']}|{stratum}: {nd} duplicate subjects; keeping first each")
        sub = sub.drop_duplicates(subset=sid, keep="first")

    sample_idx, kept = geno.align_samples(sub[sid].astype(str).tolist())
    sub = sub.set_index(sub[sid].astype(str)).loc[kept]
    n = len(kept)
    min_n = spec.get("min_n", get_default(defaults, "min_n"))
    if n < min_n:
        log.warning(f"  {spec['name']}|{stratum}: N={n} < min_n={min_n}; skipping")
        return None

    Q = [np.ones(n)] + [pd.to_numeric(sub[c], errors="coerce").to_numpy(float) for c in covars]
    cond_snps = spec.get("condition_snps") or []
    if cond_snps:
        cdos, cfound = geno.read_ids(cond_snps, sample_idx)
        for row in cdos:
            Q.append(np.where(np.isnan(row), np.nanmean(row), row))
        log.info(f"  conditioning on {cfound}")
    cc = gpu.resolve_compute(n, device=get_default(defaults, "device"),
                             max_cpus=get_default(defaults, "max_cpus"),
                             max_ram_gb=get_default(defaults, "max_ram_gb"),
                             max_gpu_mem_gb=get_default(defaults, "max_gpu_mem_gb"),
                             block_size=get_default(defaults, "block_size"))
    return _GPrep(sub, sample_idx, n, covars, np.column_stack(Q), ",".join(cond_snps), cc)


def _base_cols(meta, st, keep, dk, n) -> dict:
    """Identity + QC + standardization columns for a QC-passing block (PLAN §6)."""
    return {
        "predictor": meta["ID"].to_numpy(),
        "CHR": meta["CHR"].to_numpy(), "POS": meta["POS"].to_numpy(),
        "effect_allele": meta["ALT"].to_numpy(), "other_allele": meta["REF"].to_numpy(),
        "EAF": st.eaf[keep], "MAF": st.maf[keep], "missing_rate": st.missing_rate[keep],
        "predictor_mean": np.nanmean(dk, axis=0), "predictor_sd": np.nanstd(dk, axis=0, ddof=1),
        "zscored": False, "N": n, "n_obs": st.n_obs[keep].astype(int), "n_subjects": n,
    }


def _qc_blocks(geno, prep, defaults):
    """Yield (meta, G_imputed, stats, keep_mask, raw_dosages) for QC-passing variants."""
    maf_min = get_default(defaults, "maf_min")
    mac_min = get_default(defaults, "mac_min")
    missing_max = get_default(defaults, "missing_max")
    scanned = qc = 0
    for block in geno.iter_blocks(prep.sample_idx, prep.cc.block_size,
                                  prefetch=get_default(defaults, "prefetch")):
        st = genoio.block_stats(block.dosages)
        keep = ((st.maf >= maf_min) & (st.mac >= mac_min)
                & (st.missing_rate <= missing_max) & np.isfinite(st.eaf))
        scanned += block.dosages.shape[1]
        if not keep.any():
            continue
        qc += int(keep.sum())
        dk = block.dosages[:, keep]
        yield block.meta.iloc[np.where(keep)[0]], genoio.mean_impute(dk, st.eaf[keep]), st, keep, dk
    log.info(f"  scanned {scanned} variants; {qc} passed QC "
             f"(maf>={maf_min}, mac>={mac_min}, miss<={missing_max})")


# ============================================================================
# Genetic scan — linear
# ============================================================================


def scan_genetic_linear(df, spec, defaults, geno, stratum) -> pd.DataFrame | None:
    prep = _prep_genetic(df, spec, defaults, geno, stratum, [spec["outcome"]])
    if prep is None:
        return None
    y = pd.to_numeric(prep.sub[spec["outcome"]], errors="coerce").to_numpy(float)
    log.info(f"  N={prep.n}  covars={len(prep.covars)}  device={prep.cc.device}  block_size={prep.cc.block_size}")
    null = gpu.prepare_linear_null(y, prep.Q, prep.cc.device)
    parts = []
    for meta, G, st, keep, dk in _qc_blocks(geno, prep, defaults):
        beta, se, tval, dof = gpu.scan_linear(G, null)
        row = _base_cols(meta, st, keep, dk, prep.n)
        row.update(beta=beta, SE=se, P=2.0 * t_dist.sf(np.abs(tval), dof))
        parts.append(pd.DataFrame(row))
    if not parts:
        return None
    res = pd.concat(parts, ignore_index=True)
    res["model_type"], res["test"], res["conditioned_on"] = "linear", "wald", prep.conditioned_on
    return res


# ============================================================================
# Genetic scan — logistic (score test + SPA tail + Firth on hits)
# ============================================================================


def scan_genetic_logistic(df, spec, defaults, geno, stratum) -> pd.DataFrame | None:
    prep = _prep_genetic(df, spec, defaults, geno, stratum, [spec["outcome"]])
    if prep is None:
        return None
    y = pd.to_numeric(prep.sub[spec["outcome"]], errors="coerce").to_numpy(float)
    if not set(np.unique(y)) <= {0.0, 1.0}:
        sys.exit(f"ERROR: logistic outcome {spec['outcome']!r} must be 0/1")
    log.info(f"  N={prep.n}  cases={int(y.sum())}  covars={len(prep.covars)}  device={prep.cc.device}")
    null = gpu.prepare_logistic_null(y, prep.Q, prep.cc.device)
    if not null.converged:
        log.warning("  null logistic model did not fully converge")

    spa_on = spec.get("spa", True)
    spa_p, spa_maf = get_default(defaults, "spa_p_cutoff"), get_default(defaults, "spa_maf_cutoff")
    parts, tail_g, tail_u, tail_pos, base = [], [], [], [], 0
    for meta, G, st, keep, dk in _qc_blocks(geno, prep, defaults):
        U, V, chi2s, beta, se = gpu.scan_logistic(G, null)
        P = chi2.sf(chi2s, 1)
        row = _base_cols(meta, st, keep, dk, prep.n)
        row.update(beta=beta, SE=se, P=P)
        parts.append(pd.DataFrame(row))
        if spa_on:
            maf_j = st.maf[keep]
            tail = np.where(np.isfinite(P) & ((P < spa_p) | (maf_j <= spa_maf)))[0]
            for j in tail:
                tail_g.append(G[:, j].astype(float)); tail_u.append(float(U[j])); tail_pos.append(base + j)
        base += len(P)
    if not parts:
        return None
    res = pd.concat(parts, ignore_index=True)
    res["model_type"], res["test"], res["conditioned_on"] = "logistic", "score", prep.conditioned_on

    pos2g = dict(zip(tail_pos, tail_g))
    # SPA on the tail (rare / imbalanced) — refine the score P
    for g, u, pos in zip(tail_g, tail_u, tail_pos):
        p_spa = robust.spa_logistic(g, null.p_hat, null.Q, u)
        if np.isfinite(p_spa):
            res.at[pos, "P"] = p_spa
            res.at[pos, "test"] = "spa"
    # Firth exact refit on genome-wide hits (also robust to separation)
    for pos in res.index[res["P"] < GENOMEWIDE_SIG]:
        g = pos2g.get(pos)
        if g is None:
            continue
        fb, fse, ok = robust.firth_logistic(np.column_stack([prep.Q, g]), y)
        if ok and np.isfinite(fb[-1]) and fse[-1] > 0:
            res.at[pos, "beta"], res.at[pos, "SE"] = fb[-1], fse[-1]
            res.at[pos, "P"] = 2.0 * norm.sf(abs(fb[-1] / fse[-1]))
            res.at[pos, "test"] = "firth"
    return res


# ============================================================================
# Genetic scan — cox (null martingale residuals → linear score; SPACox + refit)
# ============================================================================


def _cox_null_martingale(sub, covars, duration_col, event_col):
    """Fit the null Cox (covariates only) and return martingale residuals + n_events."""
    from lifelines import CoxPHFitter
    dfm = pd.DataFrame({c: pd.to_numeric(sub[c], errors="coerce").to_numpy(float) for c in covars})
    dfm["_T"] = pd.to_numeric(sub[duration_col], errors="coerce").to_numpy(float)
    dfm["_E"] = pd.to_numeric(sub[event_col], errors="coerce").to_numpy(float)
    # NOTE: lifelines compute_residuals returns rows in a time-sorted index, NOT the
    # input order — reindex back to dfm's order so residuals line up with genotype.
    if covars:
        cph = CoxPHFitter().fit(dfm, duration_col="_T", event_col="_E")
        mart = cph.compute_residuals(dfm, kind="martingale")["martingale"]
    else:
        # no covariates: martingale residual = event − cumulative baseline hazard(t)
        dz = dfm.assign(_z=0.0)
        cph = CoxPHFitter().fit(dz, duration_col="_T", event_col="_E", formula="_z")
        mart = cph.compute_residuals(dz, kind="martingale")["martingale"]
    return mart.reindex(dfm.index).to_numpy(), int(dfm["_E"].sum())


def scan_genetic_cox(df, spec, defaults, geno, stratum) -> pd.DataFrame | None:
    duration_col, event_col = spec["duration_col"], spec["event_col"]
    prep = _prep_genetic(df, spec, defaults, geno, stratum, [duration_col, event_col])
    if prep is None:
        return None
    mart, n_events = _cox_null_martingale(prep.sub, prep.covars, duration_col, event_col)
    log.info(f"  N={prep.n}  events={n_events}  covars={len(prep.covars)}  device={prep.cc.device}")
    # Cox score test ≈ linear regression of martingale residuals on covariate-adjusted
    # genotype: reuse the linear kernel with y = martingale residuals.
    null = gpu.prepare_linear_null(mart, prep.Q, prep.cc.device)
    Qo = null.Qo.cpu().numpy()
    mart_c = mart - mart.mean()

    spa_on = spec.get("spa", True)
    spa_p, spa_maf = get_default(defaults, "spa_p_cutoff"), get_default(defaults, "spa_maf_cutoff")
    parts, tail_g, tail_pos, base = [], [], [], 0
    for meta, G, st, keep, dk in _qc_blocks(geno, prep, defaults):
        beta, se, tval, dof = gpu.scan_linear(G, null)
        P = 2.0 * t_dist.sf(np.abs(tval), dof)
        row = _base_cols(meta, st, keep, dk, prep.n)
        row.update(beta=beta, SE=se, P=P)
        parts.append(pd.DataFrame(row))
        if spa_on:
            maf_j = st.maf[keep]
            tail = np.where(np.isfinite(P) & ((P < spa_p) | (maf_j <= spa_maf)))[0]
            for j in tail:
                tail_g.append(G[:, j].astype(float)); tail_pos.append(base + j)
        base += len(tval)
    if not parts:
        return None
    res = pd.concat(parts, ignore_index=True)
    res["model_type"], res["test"], res["conditioned_on"] = "cox", "score", prep.conditioned_on
    res["n_events"] = n_events

    pos2g = dict(zip(tail_pos, tail_g))
    # SPACox on the tail — empirical-CGF saddlepoint on the martingale-residual score.
    # The CGF of the residuals is precomputed once, so each variant is an O(n) eval.
    if tail_g:
        cgf = robust.MartingaleCGF(mart_c)
        for g, pos in zip(tail_g, tail_pos):
            gt = g - Qo @ (Qo.T @ g)
            p_spa = robust.spa_cox(gt, float(gt @ mart_c), cgf)
            if np.isfinite(p_spa):
                res.at[pos, "P"] = p_spa
                res.at[pos, "test"] = "spa"
    # exact HR (lifelines 3-tier) on genome-wide hits
    covar_df = pd.DataFrame({c: pd.to_numeric(prep.sub[c], errors="coerce").to_numpy(float)
                             for c in prep.covars})
    covar_df["_T"] = pd.to_numeric(prep.sub[duration_col], errors="coerce").to_numpy(float)
    covar_df["_E"] = pd.to_numeric(prep.sub[event_col], errors="coerce").to_numpy(float)
    for pos in res.index[res["P"] < GENOMEWIDE_SIG]:
        g = pos2g.get(pos)
        if g is None:
            continue
        rb, rse, rp, tier = robust.cox_refit_tiers(covar_df.assign(g=g), "_T", "_E", "g")
        if np.isfinite(rb):
            res.at[pos, "beta"], res.at[pos, "SE"], res.at[pos, "P"] = rb, rse, rp
            res.at[pos, "test"], res.at[pos, "model_type"] = "full", tier
    return res


# ============================================================================
# Longitudinal — slope (precision-weighted two-stage)
# ============================================================================


def _per_subject_slopes(df, sid, ycol, tcol) -> pd.DataFrame:
    """Within-subject OLS slope of y on t per subject (vectorized via group sums).

    Returns a frame indexed by subject with `slope`, `w` (∝ time sum-of-squares =
    the slope's relative precision), and `nvis`. Subjects need ≥2 visits with time
    spread; the WLS in stage 2 self-calibrates the residual scale, so `w` is used
    unnormalized.
    """
    d = df[[sid, ycol, tcol]].apply(lambda c: pd.to_numeric(c, errors="coerce")
                                    if c.name != sid else c).dropna()
    t, y = d[tcol].to_numpy(float), d[ycol].to_numpy(float)
    d = d.assign(_t=t, _y=y, _ty=t * y, _tt=t * t, _yy=y * y)
    a = d.groupby(sid).agg(n=("_y", "size"), Sy=("_y", "sum"), St=("_t", "sum"),
                           Sty=("_ty", "sum"), Stt=("_tt", "sum"), Syy=("_yy", "sum"))
    Sxx = a.Stt - a.St ** 2 / a.n
    Sxy = a.Sty - a.St * a.Sy / a.n
    Syy = a.Syy - a.Sy ** 2 / a.n
    slope = Sxy / Sxx
    rss = (Syy - slope * Sxy).clip(lower=0)          # within-subject residual SS
    out = pd.DataFrame({"slope": slope, "Sxx": Sxx, "rss": rss, "nvis": a.n})
    return out[(a.n >= 2) & (Sxx > 1e-9) & np.isfinite(slope)]


def _prep_longitudinal_persubject(df, spec, defaults, geno, stratum, persub_cols):
    """Align a per-subject frame (already reduced from long form) to genotype.

    `persub_cols` is a frame indexed by subject id; covariates are taken from the
    subject's baseline (earliest-time) visit. Returns (_GPrep, persub, covars)."""
    sid = defaults.get("sample_id_col", "IID")
    covars = resolve_covariates(defaults.get("covariates", []),
                                spec.get("add_covariates"), spec.get("drop_covariates"))
    tcol = defaults.get("time_col")
    _require_columns(df, [sid, tcol] + covars, f"{spec['name']}|{stratum}")
    base = (df.dropna(subset=[sid, tcol]).sort_values(tcol)
            .groupby(sid).first()[covars])
    persub = persub_cols.join(base, how="inner").dropna()
    persub.index = persub.index.astype(str)
    sample_idx, kept = geno.align_samples(persub.index.tolist())
    persub = persub.loc[kept]
    n = len(kept)
    min_n = spec.get("min_n", get_default(defaults, "min_n"))
    if n < min_n:
        log.warning(f"  {spec['name']}|{stratum}: N={n} subjects < min_n={min_n}; skipping")
        return None
    cc = gpu.resolve_compute(n, device=get_default(defaults, "device"),
                             max_cpus=get_default(defaults, "max_cpus"),
                             max_ram_gb=get_default(defaults, "max_ram_gb"),
                             max_gpu_mem_gb=get_default(defaults, "max_gpu_mem_gb"),
                             block_size=get_default(defaults, "block_size"))
    prep = _GPrep(persub, sample_idx, n, covars, None, "", cc)
    return prep, persub, covars


def scan_genetic_slope(df, spec, defaults, geno, stratum) -> pd.DataFrame | None:
    outcome, tcol = spec["outcome"], defaults.get("time_col")
    if not tcol:
        sys.exit("ERROR: slope/gallop require defaults.time_col")
    sid = defaults.get("sample_id_col", "IID")
    sl = _per_subject_slopes(df, sid, outcome, tcol)
    if sl.empty:
        return None
    prepared = _prep_longitudinal_persubject(df, spec, defaults, geno, stratum, sl)
    if prepared is None:
        return None
    prep, persub, covars = prepared
    n, n_obs_total = prep.n, int(persub["nvis"].sum())
    # Standard "derived-phenotype" two-stage: GWAS the per-subject slope as a
    # quantitative trait (unweighted OLS). With balanced visit structure the slopes
    # are ~homoscedastic, so this is well-calibrated — inverse-variance weighting only
    # helps with correct variance components (σ²_b1 from an LMM), which is GALLOP's
    # job. "OLS-on-slope" is GALLOP's most robust fallback tier (PLAN §5).
    y = persub["slope"].to_numpy(float)
    Q = np.column_stack([np.ones(n)] + [persub[c].to_numpy(float) for c in covars])
    log.info(f"  N={n} subjects ({n_obs_total} obs)  covars={len(covars)}  device={prep.cc.device}")
    null = gpu.prepare_linear_null(y, Q, prep.cc.device)

    parts = []
    for meta, G, st, keep, dk in _qc_blocks(geno, prep, defaults):
        beta, se, tval, dof = gpu.scan_linear(G, null)
        row = _base_cols(meta, st, keep, dk, n)
        row["n_obs"] = n_obs_total
        P = 2.0 * t_dist.sf(np.abs(tval), dof)
        row.update(beta=beta, SE=se, P=P,
                   beta_SNPxTIME=beta, SE_SNPxTIME=se, P_SNPxTIME=P)
        parts.append(pd.DataFrame(row))
    if not parts:
        return None
    res = pd.concat(parts, ignore_index=True)
    res["model_type"], res["test"], res["conditioned_on"] = "slope", "wald", ""
    return res


# ============================================================================
# Longitudinal — shared RI+slope LMM prep (used by gallop and trajgwas)
# ============================================================================


@dataclass
class _LongPrep:
    long: pd.DataFrame
    codes: np.ndarray
    uniq: np.ndarray
    y: np.ndarray
    tvec: np.ndarray
    Xcov: np.ndarray
    lmm_null: Any
    P: np.ndarray            # gallop precompute, aligned to kept order
    q: np.ndarray
    M: np.ndarray
    Ainv: np.ndarray
    sample_idx: np.ndarray
    kept: list
    kr: list                 # kept subjects' positions in uniq/long order
    nk: int
    n_obs_total: int
    cc: Any
    covars: list
    tier: str


def _longitudinal_lmm_prep(df, spec, defaults, geno, stratum, model: str):
    """Build long arrays, fit the RI+slope LMM null once, precompute GALLOP
    projections, and align subjects to genotype. Raises on hard LMM failure (the
    caller falls back to the two-stage slope). Returns _LongPrep or None (skip)."""
    import lmm
    outcome, tcol = spec["outcome"], defaults.get("time_col")
    if not tcol:
        sys.exit(f"ERROR: {model} requires defaults.time_col")
    sid = defaults.get("sample_id_col", "IID")
    covars = resolve_covariates(defaults.get("covariates", []),
                                spec.get("add_covariates"), spec.get("drop_covariates"))
    _require_columns(df, [sid, outcome, tcol] + covars, f"{spec['name']}|{stratum}")

    geno_set = set(geno.samples)
    long = df.dropna(subset=[sid, outcome, tcol] + covars)
    long = long[long[sid].astype(str).isin(geno_set)].sort_values([sid, tcol])
    if long.empty:
        return None
    codes, uniq = pd.factorize(long[sid].astype(str), sort=False)
    n = len(uniq)
    min_n = spec.get("min_n", get_default(defaults, "min_n"))
    if n < min_n:
        log.warning(f"  {spec['name']}|{stratum}: N={n} subjects < min_n={min_n}; skipping")
        return None

    y = long[outcome].to_numpy(float)
    tvec = long[tcol].to_numpy(float)
    Xcov = np.column_stack([np.ones(len(long))]
                           + [pd.to_numeric(long[c], errors="coerce").to_numpy(float) for c in covars]
                           + [tvec])                      # fixed design: intercept + covars + time
    lmm_null = lmm.fit_ri_slope_lmm(y, Xcov, tvec, codes, n)
    P, q, M, Ainv = lmm.gallop_precompute(lmm_null, Xcov, tvec, codes, n)
    tier = model if lmm_null.converged else f"{model}_unconverged"
    if not lmm_null.converged:
        log.warning("  null LMM hit max iterations; using its (stable) estimates")

    sample_idx, kept = geno.align_samples([str(u) for u in uniq])
    upos = {u: i for i, u in enumerate(str(x) for x in uniq)}
    kr = [upos[k] for k in kept]
    cc = gpu.resolve_compute(len(kept), device=get_default(defaults, "device"),
                             max_cpus=get_default(defaults, "max_cpus"),
                             max_ram_gb=get_default(defaults, "max_ram_gb"),
                             max_gpu_mem_gb=get_default(defaults, "max_gpu_mem_gb"),
                             block_size=get_default(defaults, "block_size"))
    return _LongPrep(long, codes, uniq, y, tvec, Xcov, lmm_null,
                     P[kr], q[kr], M[kr], Ainv, sample_idx, kept, kr,
                     len(kept), len(long), cc, covars, tier)


# ============================================================================
# Longitudinal — gallop (one-stage RI+slope LMM, batched SNP & SNP×time)
# ============================================================================


def scan_genetic_gallop(df, spec, defaults, geno, stratum) -> pd.DataFrame | None:
    report_term = str(spec.get("report_term", "SNPxTIME")).upper()
    try:
        lp = _longitudinal_lmm_prep(df, spec, defaults, geno, stratum, "gallop")
    except Exception as e:  # numerical failure → OLS-on-slope fallback tier (PLAN §5)
        log.warning(f"  null LMM failed ({e}); falling back to slope two-stage")
        return scan_genetic_slope(df, spec, defaults, geno, stratum)
    if lp is None:
        return None
    log.info(f"  N={lp.nk} subjects ({lp.n_obs_total} obs)  RI sd={np.sqrt(lp.lmm_null.D[0,0]):.2f} "
             f"slope sd={np.sqrt(max(lp.lmm_null.D[1,1],0)):.2f}  device={lp.cc.device}  report_term={report_term}")
    gnull = gpu.prepare_gallop_null(lp.P, lp.q, lp.M, lp.Ainv, lp.cc.device)

    prep = _GPrep(None, lp.sample_idx, lp.nk, lp.covars, None, "", lp.cc)
    parts = []
    for meta, G, st, keep, dk in _qc_blocks(geno, prep, defaults):
        bs, ses, bx, sex = gpu.scan_gallop(G, gnull)
        p_snp = 2.0 * norm.sf(np.abs(bs / ses))
        p_xt = 2.0 * norm.sf(np.abs(bx / sex))
        row = _base_cols(meta, st, keep, dk, lp.nk)
        row["n_obs"] = lp.n_obs_total
        row.update(beta_SNP=bs, SE_SNP=ses, P_SNP=p_snp,
                   beta_SNPxTIME=bx, SE_SNPxTIME=sex, P_SNPxTIME=p_xt)
        row.update(**({"beta": bs, "SE": ses, "P": p_snp} if report_term == "SNP"
                      else {"beta": bx, "SE": sex, "P": p_xt}))
        parts.append(pd.DataFrame(row))
    if not parts:
        return None
    res = pd.concat(parts, ignore_index=True)
    res["model_type"], res["test"], res["conditioned_on"] = lp.tier, "wald", ""
    return res


# ============================================================================
# Longitudinal — trajgwas (location–scale: mean β AND within-subject variance τ)
# ============================================================================


def _dispersion_variance_residuals(lp: "_LongPrep"):
    """Fit the within-subject variance model on conditional residuals and reduce to a
    per-subject 'excess variance' v_i = Σ_j (ê²_ij/σ̂²_ij − 1) — the input to the τ test.

    log σ²_ij = [1, t_ij]·τ₀ (variance may drift with time), fit by log-OLS of the
    squared residuals and renormalized so E[ê²/σ̂²]=1 (so v_i has mean ~0 under H0)."""
    import lmm
    ehat = lmm.conditional_residuals(lp.lmm_null, lp.Xcov, lp.tvec, lp.y, lp.codes, len(lp.uniq))
    e2 = ehat ** 2 + 1e-8
    W = np.column_stack([np.ones(len(ehat)), lp.tvec])          # dispersion covariates
    tau0, *_ = np.linalg.lstsq(W, np.log(e2), rcond=None)
    sig2 = np.exp(W @ tau0)
    ratio = e2 / sig2
    ratio /= ratio.mean()                                       # renormalize -> mean 1
    excess = ratio - 1.0
    # per-subject sums (in uniq/long order), then align to kept order
    n_uniq = len(lp.uniq)
    v = np.bincount(lp.codes, weights=excess, minlength=n_uniq)
    nvis = np.bincount(lp.codes, minlength=n_uniq).astype(float)
    return v[lp.kr], nvis[lp.kr]


def scan_genetic_trajgwas(df, spec, defaults, geno, stratum) -> pd.DataFrame | None:
    report_term = str(spec.get("report_term", "JOINT")).upper()
    try:
        lp = _longitudinal_lmm_prep(df, spec, defaults, geno, stratum, "trajgwas")
    except Exception as e:
        log.warning(f"  null LMM failed ({e}); falling back to slope two-stage")
        return scan_genetic_slope(df, spec, defaults, geno, stratum)
    if lp is None:
        return None

    # --- mean axis: reuse GALLOP's SNP-main term (so it agrees with gallop) ---
    gnull = gpu.prepare_gallop_null(lp.P, lp.q, lp.M, lp.Ainv, lp.cc.device)
    # --- variance axis: per-subject excess-variance residual, scanned on genotype,
    #     adjusted for subject-level covariates (baseline visit values) ---
    v, nvis = _dispersion_variance_residuals(lp)
    v_std = v / np.sqrt(2.0 * nvis)                             # ~homoscedastic (Var(v_i)≈2·nᵢ)
    Qsub = _subject_covariate_matrix(lp, defaults)
    vnull = gpu.prepare_linear_null(v_std, Qsub, lp.cc.device)
    v_centered = v - v.mean()

    spa_on = spec.get("spa", True)
    spa_p, spa_maf = get_default(defaults, "spa_p_cutoff"), get_default(defaults, "spa_maf_cutoff")
    log.info(f"  N={lp.nk} subjects ({lp.n_obs_total} obs)  RI sd={np.sqrt(lp.lmm_null.D[0,0]):.2f} "
             f"resid sd={np.sqrt(lp.lmm_null.sigma2):.2f}  device={lp.cc.device}  report_term={report_term}")

    prep = _GPrep(None, lp.sample_idx, lp.nk, lp.covars, None, "", lp.cc)
    parts, tail, base_off = [], [], 0
    for meta, G, st, keep, dk in _qc_blocks(geno, prep, defaults):
        bs, ses, _, _ = gpu.scan_gallop(G, gnull)                    # mean term (β_SNP)
        tv, tse, tt, _ = gpu.scan_linear(G, vnull)                   # variance term (τ)
        p_mean = 2.0 * norm.sf(np.abs(bs / ses))
        p_var = 2.0 * norm.sf(np.abs(tt))
        p_joint = chi2.sf((bs / ses) ** 2 + tt ** 2, 2)              # mean⊥variance -> 2 df
        row = _base_cols(meta, st, keep, dk, lp.nk)
        row["n_obs"] = lp.n_obs_total
        row.update(beta_mean=bs, SE_mean=ses, P_mean=p_mean,
                   tau_var=tv, SE_var=tse, P_var=p_var, P_joint=p_joint)
        prim = {"MEAN": (bs, ses, p_mean), "VAR": (tv, tse, p_var)}.get(report_term)
        if prim is not None:
            row.update(beta=prim[0], SE=prim[1], P=prim[2])
        else:                                                        # JOINT: 2-df, no single β
            row.update(beta=np.nan, SE=np.nan, P=p_joint)
        parts.append(pd.DataFrame(row))
        if spa_on:                                                   # SPA the variance-test tail
            maf_j = st.maf[keep]
            for j in np.where(np.isfinite(p_var) & ((p_var < spa_p) | (maf_j <= spa_maf)))[0]:
                tail.append((base_off + j, G[:, j].astype(float)))
        base_off += len(p_var)
    if not parts:
        return None
    res = pd.concat(parts, ignore_index=True)
    res["model_type"], res["test"], res["conditioned_on"] = lp.tier, "score", ""

    # SPACox-style empirical-CGF SPA on the variance-score tail (v_i as the residual)
    if spa_on and tail:
        Qo = vnull.Qo.cpu().numpy()
        cgf = robust.MartingaleCGF(v_centered)
        for pos, g in tail:
            gt = g - Qo @ (Qo.T @ g)
            p_spa = robust.spa_cox(gt, float(gt @ v_centered), cgf)
            if np.isfinite(p_spa):
                res.at[pos, "P_var"] = p_spa
                if report_term == "VAR":
                    res.at[pos, "P"] = p_spa
                res.at[pos, "test"] = "spa"
    return res


def _subject_covariate_matrix(lp: "_LongPrep", defaults) -> np.ndarray:
    """Baseline (earliest-time) covariate values per kept subject → design for the τ scan."""
    sid = defaults.get("sample_id_col", "IID")
    base = lp.long.sort_values(defaults.get("time_col")).drop_duplicates(sid, keep="first")
    base = base.set_index(base[sid].astype(str)).loc[lp.kept]
    cols = [np.ones(lp.nk)] + [pd.to_numeric(base[c], errors="coerce").to_numpy(float) for c in lp.covars]
    return np.column_stack(cols)


# ============================================================================
# Tabular scan (linear + logistic; no genotype fileset)
# ============================================================================


def _build_tabular_design(df, spec, defaults, stratum, need_cols):
    sid = defaults.get("sample_id_col", "IID")
    covars = resolve_covariates(defaults.get("covariates", []),
                                spec.get("add_covariates"), spec.get("drop_covariates"))
    predictors = resolve_tabular_predictors(df, spec.get("predictors", defaults.get("predictors")))
    _require_columns(df, [sid] + need_cols + covars, f"{spec['name']}|{stratum}")
    sub = df.dropna(subset=[sid] + need_cols + covars)
    if sub[sid].duplicated().any():
        sub = sub.drop_duplicates(subset=sid, keep="first")
    n = len(sub)
    min_n = spec.get("min_n", get_default(defaults, "min_n"))
    if n < min_n:
        log.warning(f"  {spec['name']}|{stratum}: N={n} < min_n={min_n}; skipping")
        return None
    Q = np.column_stack([np.ones(n)] + [pd.to_numeric(sub[c], errors="coerce").to_numpy(float)
                                        for c in covars])
    zscore_on, zb, zd = resolve_zscore_opts(spec, defaults)
    cols, means, sds, zflags, Gcols, n_obs = [], [], [], [], [], []
    for c in predictors:
        raw = pd.to_numeric(sub[c], errors="coerce").to_numpy(float)
        obs = np.isfinite(raw)
        if obs.sum() < min_n or np.nanstd(raw) == 0:
            continue
        mean, sd = np.nanmean(raw), np.nanstd(raw, ddof=1)
        z = zscore_on and _should_zscore(_predictor_kind(sub[c]), zb, zd)
        col = (raw - mean) / sd if z else raw.copy()
        col[~obs] = 0.0 if z else mean
        cols.append(c); means.append(mean); sds.append(sd); zflags.append(z)
        Gcols.append(col.astype(np.float32)); n_obs.append(int(obs.sum()))
    if not cols:
        return None
    G = np.ascontiguousarray(np.column_stack(Gcols))
    cc = gpu.resolve_compute(n, device=get_default(defaults, "device"),
                             max_cpus=get_default(defaults, "max_cpus"),
                             max_gpu_mem_gb=get_default(defaults, "max_gpu_mem_gb"),
                             block_size=get_default(defaults, "block_size"))
    base = {
        "predictor": cols, "CHR": np.nan, "POS": np.nan,
        "effect_allele": None, "other_allele": None,
        "EAF": np.nan, "MAF": np.nan, "missing_rate": 1.0 - np.array(n_obs) / n,
        "predictor_mean": means, "predictor_sd": sds, "zscored": zflags,
        "N": n, "n_obs": n_obs, "n_subjects": n,
    }
    return sub, Q, G, base, cc, n


def scan_tabular_linear(df, spec, defaults, stratum) -> pd.DataFrame | None:
    built = _build_tabular_design(df, spec, defaults, stratum, [spec["outcome"]])
    if built is None:
        return None
    sub, Q, G, base, cc, n = built
    y = pd.to_numeric(sub[spec["outcome"]], errors="coerce").to_numpy(float)
    log.info(f"  N={n}  {len(base['predictor'])} tabular predictors  device={cc.device}")
    beta, se, tval, dof = gpu.scan_linear(G, gpu.prepare_linear_null(y, Q, cc.device))
    res = pd.DataFrame({**base, "beta": beta, "SE": se, "P": 2.0 * t_dist.sf(np.abs(tval), dof),
                        "model_type": "linear", "test": "wald", "conditioned_on": ""})
    return res


def scan_tabular_logistic(df, spec, defaults, stratum) -> pd.DataFrame | None:
    built = _build_tabular_design(df, spec, defaults, stratum, [spec["outcome"]])
    if built is None:
        return None
    sub, Q, G, base, cc, n = built
    y = pd.to_numeric(sub[spec["outcome"]], errors="coerce").to_numpy(float)
    if not set(np.unique(y)) <= {0.0, 1.0}:
        sys.exit(f"ERROR: logistic outcome {spec['outcome']!r} must be 0/1")
    log.info(f"  N={n}  cases={int(y.sum())}  {len(base['predictor'])} tabular predictors  device={cc.device}")
    U, V, chi2s, beta, se = gpu.scan_logistic(G, gpu.prepare_logistic_null(y, Q, cc.device))
    res = pd.DataFrame({**base, "beta": beta, "SE": se, "P": chi2.sf(chi2s, 1),
                        "model_type": "logistic", "test": "score", "conditioned_on": ""})
    return res


# ============================================================================
# Run driver
# ============================================================================


def run_all(config: dict, cwd: str, run_filter: list[str] | None) -> None:
    defaults = config["defaults"]
    clinical = load_clinical(resolve_input_path(defaults.get("clinical_file", "*.tsv"), cwd))
    clinical = apply_prefilter(clinical, defaults.get("prefilter"))

    geno = None
    if defaults.get("genotype_prefix"):
        prefix = os.path.join(cwd, defaults["genotype_prefix"])
        if os.path.exists(prefix + ".pgen"):
            geno = genoio.GenoReader(prefix)
            log.info(f"Genotype fileset: {geno.n_variants:,} variants x {geno.n_samples:,} samples")

    clinical = apply_derive(clinical, defaults.get("derive"), defaults, geno)

    output_dir = os.path.join(cwd, defaults.get("output_dir", "results"))
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strata_default = defaults.get("strata") or [None]

    for spec in config["runs"]:
        name = spec.get("name", "unnamed")
        if run_filter and name not in run_filter:
            continue
        model = spec.get("model", "linear")
        source = spec.get("predictor_source", get_default(defaults, "predictor_source"))
        log.info("")
        log.info("=" * 70)
        log.info(f"RUN: {name}  (model={model}, predictor_source={source})")
        log.info("=" * 70)

        genetic_scanners = {"linear": scan_genetic_linear, "logistic": scan_genetic_logistic,
                            "cox": scan_genetic_cox, "slope": scan_genetic_slope,
                            "gallop": scan_genetic_gallop, "trajgwas": scan_genetic_trajgwas}
        tabular_scanners = {"linear": scan_tabular_linear, "logistic": scan_tabular_logistic}
        scanners = genetic_scanners if source == "genetic" else tabular_scanners
        if model not in scanners:
            log.warning(f"  model={model!r} not available for predictor_source={source!r} "
                        f"(Phase 2–3 = linear/logistic/cox); skipping")
            continue
        if source == "genetic" and geno is None:
            sys.exit(f"ERROR: run {name!r} is genetic but no genotype fileset was loaded")

        for stratum in (spec.get("strata", strata_default) or [None]):
            frame = build_stratum_frame(clinical, spec, defaults, stratum)
            log.info(f"[{name}|{stratum}] {len(frame):,} rows after stratum+sample_filter")
            if source == "genetic":
                results = scanners[model](frame, spec, defaults, geno, stratum)
            else:
                results = scanners[model](frame, spec, defaults, stratum)
            if results is None or results.empty:
                log.warning(f"  [{name}|{stratum}] no results; no CSV written")
                continue

            results = filter_numerical_failures(results, f"{name}|{stratum}")
            results = apply_bonferroni(results)
            results["genomewide_sig"] = results["P"] < GENOMEWIDE_SIG
            results["strata"] = "ALL" if stratum is None else stratum
            results = results.reindex(columns=OUTPUT_COLUMNS)

            suffix = "ALL" if stratum is None else stratum
            out = os.path.join(output_dir, f"{name}-{suffix}-{timestamp}.csv")
            results.to_csv(out, index=False)
            lam, n_lam = lambda_gc(results["P"])
            log.info(f"  [{name}|{stratum}] wrote {len(results)} rows "
                     f"({int(results['significant'].sum())} Bonferroni-sig, "
                     f"{int(results['genomewide_sig'].sum())} genome-wide) -> {os.path.basename(out)}")
            log.info(f"    λ-GC = {lam:.4f}  (N_tests = {n_lam})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Biomarker Gran Prix association engine")
    ap.add_argument("--config", required=True, help="YAML config")
    ap.add_argument("--run", nargs="+", default=None, help="only these run names")
    ap.add_argument("--cwd", default=".", help="working dir for globs/outputs")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    run_all(cfg, os.path.abspath(args.cwd), args.run)


if __name__ == "__main__":
    main()
