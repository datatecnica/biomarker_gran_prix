#!/usr/bin/env python3
"""meta_analysis.py — IVW fixed-effects + DerSimonian-Laird random-effects
meta-analysis over per-stratum association results from gwas.py.

Auto-discovers per-stratum CSVs by the `<run>-<stratum>-<timestamp>.csv` filename
convention, joins each run's strata **on the predictor (SNP)** — harmonizing effect
alleles so β/EAF are on the same counted allele across strata — pools β/SE, and emits
a timestamped META CSV with FE + RE statistics, heterogeneity (Q / P_het / I² / τ²),
a per-study direction string, and CHR/POS carried through for downstream Manhattan plots.

Effect-allele harmonization is a no-op when the strata share one `.pgen` (the counted
allele is identical), but the check stays: if a stratum's `effect_allele` differs, its
β sign and EAF are flipped before pooling (PLAN §10).

Usage:
    python3 meta_analysis.py --results-dir results/ [--output-dir meta/]
    python3 meta_analysis.py --inputs X-EUR-*.csv X-AJ-*.csv --output META_X.csv

Design mirrors ../proteomics_data_mine/meta_analysis.py (same FE/RE math), generalized
to a genotype join key.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import sys
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

log = logging.getLogger("meta_analysis")
ALPHA = 0.05
GENOMEWIDE_SIG = 5e-8

# "<name>-<stratum>-<YYYYMMDD_HHMMSS>.csv"; <name> may contain hyphens.
_FILENAME_RE = re.compile(r"^(?P<name>.+)-(?P<stratum>[^-]+)-(?P<ts>\d{8}_\d{6})\.csv$")

# Identity columns carried through from the first stratum that has them.
_IDENT_COLS = ["CHR", "POS", "effect_allele", "other_allele", "model_type"]


def setup_logging(log_file: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
                        force=True)


# ============================================================================
# File discovery
# ============================================================================


def discover_runs(results_dir: str) -> dict[str, list[tuple[str, str, str]]]:
    """{run_name: [(stratum, ts, path), ...]}, keeping the latest ts per (run, stratum)."""
    groups: dict[tuple[str, str], tuple[str, str]] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.csv"))):
        fname = os.path.basename(path)
        if fname.startswith("META_"):
            continue
        m = _FILENAME_RE.match(fname)
        if not m:
            log.warning(f"  skipping non-conforming filename: {fname}")
            continue
        key = (m.group("name"), m.group("stratum"))
        ts = m.group("ts")
        if key not in groups or groups[key][0] < ts:
            groups[key] = (ts, path)
    out: dict[str, list[tuple[str, str, str]]] = {}
    for (name, stratum), (ts, path) in groups.items():
        out.setdefault(name, []).append((stratum, ts, path))
    return out


# ============================================================================
# Meta math (IVW-FE + DerSimonian-Laird RE) — identical to the reference
# ============================================================================


def _direction_string(betas: np.ndarray) -> str:
    return "".join("?" if not np.isfinite(b) else "+" if b > 0 else "-" if b < 0 else "0"
                   for b in betas)


def meta_row(betas: np.ndarray, ses: np.ndarray, ns: np.ndarray) -> dict:
    """IVW-FE + DL-RE for one predictor across studies. 1 study → pass-through."""
    mask = np.isfinite(betas) & np.isfinite(ses) & (ses > 0)
    betas, ses, ns = betas[mask], ses[mask], ns[mask]
    k = len(betas)
    out = {"n_studies": k, "N_total": int(np.nansum(ns)) if k else 0,
           "direction": _direction_string(betas),
           "beta_FE": np.nan, "SE_FE": np.nan, "Z_FE": np.nan, "P_FE": np.nan,
           "beta_RE": np.nan, "SE_RE": np.nan, "Z_RE": np.nan, "P_RE": np.nan,
           "Q": np.nan, "P_het": np.nan, "I2": np.nan, "tau2": np.nan}
    if k == 0:
        return out

    w = 1.0 / ses ** 2
    beta_fe = float(np.sum(w * betas) / np.sum(w))
    se_fe = float(np.sqrt(1.0 / np.sum(w)))
    z_fe = beta_fe / se_fe
    out.update(beta_FE=beta_fe, SE_FE=se_fe, Z_FE=float(z_fe), P_FE=float(2 * norm.sf(abs(z_fe))))
    if k == 1:
        out.update(beta_RE=beta_fe, SE_RE=se_fe, Z_RE=float(z_fe), P_RE=out["P_FE"])
        return out

    Q = float(np.sum(w * (betas - beta_fe) ** 2))
    dfree = k - 1
    P_het = float(chi2.sf(Q, dfree))
    I2 = float(max(0.0, (Q - dfree) / Q * 100)) if Q > 0 else 0.0
    sum_w, sum_w2 = float(np.sum(w)), float(np.sum(w ** 2))
    denom = sum_w - sum_w2 / sum_w if sum_w > 0 else 0.0
    tau2 = max(0.0, (Q - dfree) / denom) if denom > 0 else 0.0
    w_star = 1.0 / (ses ** 2 + tau2)
    beta_re = float(np.sum(w_star * betas) / np.sum(w_star))
    se_re = float(np.sqrt(1.0 / np.sum(w_star)))
    z_re = beta_re / se_re
    out.update(Q=Q, P_het=P_het, I2=I2, tau2=tau2, beta_RE=beta_re, SE_RE=se_re,
               Z_RE=float(z_re), P_RE=float(2 * norm.sf(abs(z_re))))
    return out


# ============================================================================
# Combine one run's stratum files (join on predictor + harmonize alleles)
# ============================================================================


# Effect axes to pool, in preference order. β/SE are NaN for a 2-df JOINT trajgwas run,
# so meta falls back to the mean or variance axis (each has a poolable effect).
_EFFECT_AXES = [("beta", "SE", "P"), ("beta_mean", "SE_mean", "P_mean"),
                ("tau_var", "SE_var", "P_var"), ("beta_SNPxTIME", "SE_SNPxTIME", "P_SNPxTIME")]


def _effect_cols(df: pd.DataFrame) -> tuple[str, str, str]:
    for b, s, p in _EFFECT_AXES:
        if b in df.columns and s in df.columns and df[b].notna().any():
            return b, s, (p if p in df.columns else "P")
    return "beta", "SE", "P"


def combine_run(run_name: str, files: list[tuple[str, str, str]]) -> pd.DataFrame:
    strata_frames: dict[str, pd.DataFrame] = {}
    for stratum, _ts, path in files:
        df = pd.read_csv(path)
        if "predictor" not in df.columns or "N" not in df.columns:
            log.warning(f"  {os.path.basename(path)} missing predictor/N; skipping")
            continue
        strata_frames[stratum] = df
    if not strata_frames:
        return pd.DataFrame()
    strata_order = sorted(strata_frames)
    bc, sc, pc = _effect_cols(next(iter(strata_frames.values())))
    if bc != "beta":
        log.info(f"  pooling effect axis '{bc}' (no single β; e.g. trajgwas JOINT)")

    # unified predictor index + carried identity columns (coalesced across strata)
    index_df = (pd.concat([f[["predictor"]] for f in strata_frames.values()], ignore_index=True)
                .drop_duplicates("predictor").reset_index(drop=True))
    ident = index_df.copy()
    for col in _IDENT_COLS + ["EAF"]:
        series = None
        for f in strata_frames.values():
            if col in f.columns:
                m = (f[["predictor", col]].dropna(subset=[col])
                     .drop_duplicates("predictor").set_index("predictor")[col])
                series = m if series is None else series.combine_first(m)
        if series is not None:
            ident[col] = ident["predictor"].map(series)

    merged = index_df
    for s in strata_order:
        f = strata_frames[s]
        keep = ["predictor", bc, sc, pc, "N"]
        for c in ("EAF", "effect_allele"):
            if c in f.columns:
                keep.append(c)
        sub = f[keep].rename(columns={bc: f"beta_{s}", sc: f"SE_{s}", pc: f"P_{s}",
                                      "N": f"N_{s}", "EAF": f"EAF_{s}", "effect_allele": f"EA_{s}"})
        merged = merged.merge(sub, on="predictor", how="left")

    # --- effect-allele harmonization: flip β/EAF where a stratum's counted allele
    #     differs from the reference (first stratum with a non-null allele) ---
    ref_ea = None
    for s in strata_order:
        if f"EA_{s}" in merged.columns:
            ref_ea = merged[f"EA_{s}"] if ref_ea is None else ref_ea.combine_first(merged[f"EA_{s}"])
    n_flipped = 0
    if ref_ea is not None:
        for s in strata_order:
            eac = f"EA_{s}"
            if eac not in merged.columns:
                continue
            flip = merged[eac].notna() & ref_ea.notna() & (merged[eac] != ref_ea)
            if flip.any():
                n_flipped += int(flip.sum())
                merged.loc[flip, f"beta_{s}"] *= -1
                if f"EAF_{s}" in merged.columns:
                    merged.loc[flip, f"EAF_{s}"] = 1.0 - merged.loc[flip, f"EAF_{s}"]
    if n_flipped:
        log.info(f"  harmonized {n_flipped} stratum-variant allele flips")

    log.info(f"  {run_name}: {len(merged):,} predictors across strata {strata_order}")
    results = []
    for _, row in merged.iterrows():
        betas = np.array([row.get(f"beta_{s}", np.nan) for s in strata_order], float)
        ses = np.array([row.get(f"SE_{s}", np.nan) for s in strata_order], float)
        ns = np.array([row.get(f"N_{s}", np.nan) for s in strata_order], float)
        stats = meta_row(betas, ses, ns)
        stats["predictor"] = row["predictor"]
        for s in strata_order:
            for c in ("beta", "SE", "P", "N"):
                stats[f"{c}_{s}"] = row.get(f"{c}_{s}", np.nan)
            stats[f"EAF_{s}"] = row.get(f"EAF_{s}", np.nan)
            stats[f"EA_{s}"] = row.get(f"EA_{s}", np.nan)
        results.append(stats)
    meta_df = pd.DataFrame(results).merge(ident, on="predictor", how="left")
    meta_df = _add_strand_flags(meta_df, strata_order)

    n = len(meta_df)
    thresh = ALPHA / n if n else np.nan
    meta_df["bonferroni_threshold"] = thresh
    meta_df["significant_FE"] = meta_df["P_FE"] < thresh
    meta_df["significant_RE"] = meta_df["P_RE"] < thresh
    meta_df["genomewide_FE"] = meta_df["P_FE"] < GENOMEWIDE_SIG
    meta_df["genomewide_RE"] = meta_df["P_RE"] < GENOMEWIDE_SIG

    col_order = (["predictor", "CHR", "POS", "effect_allele", "other_allele", "EAF",
                  "palindromic", "allele_mismatch", "strand_flag",
                  "n_studies", "N_total", "direction",
                  "beta_FE", "SE_FE", "Z_FE", "P_FE", "beta_RE", "SE_RE", "Z_RE", "P_RE",
                  "Q", "P_het", "I2", "tau2"]
                 + [f"{c}_{s}" for s in strata_order for c in ("beta", "SE", "P", "N", "EAF")]
                 + ["model_type", "bonferroni_threshold", "significant_FE", "significant_RE",
                    "genomewide_FE", "genomewide_RE"])
    return meta_df[[c for c in col_order if c in meta_df.columns]]


# ============================================================================
# Strand-ambiguity flagging (palindromic / mismatched alleles)
# ============================================================================

_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}
_MAF_AMBIG_LO, _MAF_AMBIG_HI = 0.40, 0.60   # EAF band where strand can't be resolved by freq
_FREQ_CONFLICT = 0.15                        # cross-study EAF spread suggesting a strand flip


def _is_palindromic(ea, oa) -> bool:
    ea, oa = str(ea).upper(), str(oa).upper()
    return ea in _COMPLEMENT and _COMPLEMENT[ea] == oa


def _add_strand_flags(meta_df: pd.DataFrame, strata_order: list[str]) -> pd.DataFrame:
    """Flag potential strand issues on **palindromic** (A/T, C/G) or **mismatched**
    variants, when they can't be resolved: EAF in the 40–60% band, or conflicting
    EAF across studies (a hallmark of an uncorrected strand flip). See PLAN §10.
    """
    n = len(meta_df)
    ea = meta_df["effect_allele"] if "effect_allele" in meta_df else pd.Series([np.nan] * n)
    oa = meta_df["other_allele"] if "other_allele" in meta_df else pd.Series([np.nan] * n)
    palindromic = np.array([_is_palindromic(a, b) if pd.notna(a) and pd.notna(b) else False
                            for a, b in zip(ea, oa)])

    # allele mismatch: a stratum's effect allele isn't one of the two reference alleles
    allowed = [{str(a).upper(), str(b).upper()} for a, b in zip(ea, oa)]
    mismatch = np.zeros(n, bool)
    for s in strata_order:
        col = f"EA_{s}"
        if col in meta_df:
            for i, v in enumerate(meta_df[col]):
                if pd.notna(v) and str(v).upper() not in allowed[i]:
                    mismatch[i] = True

    # harmonized meta EAF + cross-study spread
    eaf_cols = [f"EAF_{s}" for s in strata_order if f"EAF_{s}" in meta_df]
    if eaf_cols:
        E = meta_df[eaf_cols].to_numpy(float)
        with np.errstate(invalid="ignore"):
            meta_eaf = np.nanmean(E, axis=1)
            spread = np.nanmax(E, axis=1) - np.nanmin(E, axis=1)
        spread = np.where(np.isfinite(spread), spread, 0.0)
    else:
        meta_eaf = meta_df["EAF"].to_numpy(float) if "EAF" in meta_df else np.full(n, np.nan)
        spread = np.zeros(n)

    maf_ambiguous = (meta_eaf >= _MAF_AMBIG_LO) & (meta_eaf <= _MAF_AMBIG_HI)
    freq_conflict = spread > _FREQ_CONFLICT
    strand_flag = (palindromic | mismatch) & (maf_ambiguous | freq_conflict)

    if eaf_cols:
        meta_df["EAF"] = meta_eaf                       # report the harmonized meta EAF
    meta_df["palindromic"] = palindromic
    meta_df["allele_mismatch"] = mismatch
    meta_df["strand_flag"] = strand_flag
    if int(strand_flag.sum()):
        log.info(f"  strand check: {int(palindromic.sum())} palindromic, "
                 f"{int(mismatch.sum())} allele-mismatch → {int(strand_flag.sum())} flagged ambiguous")
    return meta_df


def lambda_gc(pvals: Iterable[float]) -> tuple[float, int]:
    p = pd.Series(pvals).dropna()
    p = p[(p > 0) & (p <= 1)]
    if len(p) == 0:
        return float("nan"), 0
    return float(np.median(chi2.ppf(1 - p, df=1)) / chi2.ppf(0.5, df=1)), len(p)


# ============================================================================
# Drivers
# ============================================================================


def run_auto(results_dir: str, output_dir: str, run_filter: str | None) -> None:
    groups = discover_runs(results_dir)
    if not groups:
        sys.exit(f"ERROR: no conforming CSVs found in {results_dir}")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info(f"Discovered {len(groups)} run group(s) in {results_dir}")
    for name, files in sorted(groups.items()):
        if run_filter and name != run_filter:
            continue
        log.info("")
        log.info("=" * 70)
        log.info(f"META: {name}  ({len(files)} stratum file(s): {sorted(s for s, _, _ in files)})")
        log.info("=" * 70)
        meta_df = combine_run(name, files)
        if meta_df.empty:
            log.warning(f"  {name}: no rows; skipping")
            continue
        out = os.path.join(output_dir, f"META_{name}-{timestamp}.csv")
        meta_df.to_csv(out, index=False)
        thr = float(meta_df["bonferroni_threshold"].iloc[0])
        log.info(f"  wrote {len(meta_df):,} rows ({int(meta_df['significant_FE'].sum())} sig FE, "
                 f"{int(meta_df['significant_RE'].sum())} sig RE @ {thr:.2e}, "
                 f"{int(meta_df['genomewide_FE'].sum())} genome-wide FE) → {os.path.basename(out)}")
        lam_fe, n_fe = lambda_gc(meta_df["P_FE"])
        lam_re, _ = lambda_gc(meta_df["P_RE"])
        log.info(f"  λ_FE = {lam_fe:.4f}  λ_RE = {lam_re:.4f}  (N={n_fe})")


def run_explicit(inputs: list[str], output_path: str) -> None:
    files, name = [], None
    for path in inputs:
        m = _FILENAME_RE.match(os.path.basename(path))
        if m:
            files.append((m.group("stratum"), m.group("ts"), path))
            name = name or m.group("name")
        else:
            files.append((os.path.splitext(os.path.basename(path))[0], "00000000_000000", path))
    meta_df = combine_run(name or "manual", files)
    if meta_df.empty:
        sys.exit("ERROR: no rows after combining inputs")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    meta_df.to_csv(output_path, index=False)
    log.info(f"Wrote {len(meta_df):,} rows → {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Meta-analyze per-stratum GWAS results")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--output-dir", default="meta")
    p.add_argument("--run", default=None, help="auto mode: only this run group")
    p.add_argument("--inputs", nargs="+", default=None, help="explicit mode: per-stratum files")
    p.add_argument("--output", default=None, help="explicit mode: output CSV path")
    args = p.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.dirname(os.path.abspath(args.output or args.output_dir)) or "."
    os.makedirs(out_dir, exist_ok=True)
    setup_logging(os.path.join(out_dir, f"meta_analysis_{timestamp}.log"))
    if args.inputs:
        if not args.output:
            sys.exit("ERROR: --output is required with --inputs")
        run_explicit(args.inputs, args.output)
    else:
        run_auto(args.results_dir, args.output_dir, args.run)
    log.info("\nDone.")


if __name__ == "__main__":
    main()
