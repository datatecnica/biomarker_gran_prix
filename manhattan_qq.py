#!/usr/bin/env python3
"""manhattan_qq.py — Manhattan / volcano + QQ plots for Biomarker Gran Prix.

Plots **any** result source produced by the pipeline — a per-stratum association CSV
from gwas.py (uses the `P` column) *or* a meta-analysis CSV from meta_analysis.py
(uses `P_FE`) — auto-detecting which:

  * **genetic** predictors (CHR/POS present) → a **Manhattan** plot: chromosomes
    alternate **grey / black**, and genome-wide hits alternate **burgundy / mint**
    per chromosome so they pop while keeping the chromosome banding.
  * **non-genetic** predictors (tabular; CHR/POS null) → a **volcano** plot
    (−log₁₀P vs effect size), top hits labelled.
  * every figure carries a **QQ** panel with λ-GC reported **at several MAF bins**
    (rare / low-freq / common) as well as across **all** variants.

Usage:
    python3 manhattan_qq.py --results-dir results/ --meta-dir meta/ [--output-dir plots/]
    python3 manhattan_qq.py --input results/linear_bmkX-ALL-<ts>.csv
    python3 manhattan_qq.py --input meta/META_linear_bmkX_strat-<ts>.csv

Design adapts ../proteomics_data_mine/plot_results.py (per-source panels, λ reporting).
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2

try:
    from adjustText import adjust_text
    _HAS_ADJUST = True
except ImportError:
    _HAS_ADJUST = False

log = logging.getLogger("manhattan_qq")

GENOMEWIDE_SIG = 5e-8
ALPHA = 0.05

# Palette (user spec): chromosomes alternate grey/black; hits alternate burgundy/mint.
CHR_GREY, CHR_BLACK = "#9a9a9a", "#333333"
HIT_BURGUNDY, HIT_MINT = "#8c1d40", "#2fae8f"
GW_LINE, BONF_LINE = "#c0392b", "#2980b9"

# MAF bins for the QQ panel: (label, lo, hi) with lo ≤ MAF < hi.
_MAF_BINS = [("all", 0.0, 1.0), ("common MAF≥0.05", 0.05, 0.5001),
             ("low-freq 0.01–0.05", 0.01, 0.05), ("rare MAF<0.01", 0.0, 0.01)]

_META_RE = re.compile(r"^META_(?P<name>.+)-(?P<ts>\d{8}_\d{6})\.csv$")
_STUDY_RE = re.compile(r"^(?P<name>.+)-(?P<stratum>[^-]+)-(?P<ts>\d{8}_\d{6})\.csv$")


def setup_logging(log_file: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
                        force=True)


# ============================================================================
# Helpers
# ============================================================================


def lambda_gc(pvals) -> float:
    p = pd.Series(pvals).dropna()
    p = p[(p > 0) & (p <= 1)]
    if len(p) == 0:
        return float("nan")
    return float(np.median(chi2.ppf(1 - p, df=1)) / chi2.ppf(0.5, df=1))


def _pcol(df: pd.DataFrame) -> str | None:
    """Which P column to plot: 'P' (study) or 'P_FE' (meta), else None."""
    for c in ("P", "P_FE"):
        if c in df.columns and df[c].notna().any():
            return c
    return None


def _betacol(df: pd.DataFrame) -> str | None:
    for c in ("beta", "beta_FE"):
        if c in df.columns and df[c].notna().any():
            return c
    return None


def _is_genetic(df: pd.DataFrame) -> bool:
    return "CHR" in df.columns and "POS" in df.columns and df["CHR"].notna().any() \
        and df["POS"].notna().any()


def _maf(df: pd.DataFrame) -> pd.Series:
    if "MAF" in df.columns and df["MAF"].notna().any():
        return pd.to_numeric(df["MAF"], errors="coerce")
    if "EAF" in df.columns:
        eaf = pd.to_numeric(df["EAF"], errors="coerce")
        return np.minimum(eaf, 1 - eaf)
    return pd.Series(np.nan, index=df.index)


def _chr_key(c: str):
    c = str(c).replace("chr", "")
    return (0, int(c)) if c.isdigit() else (1, c)


# ============================================================================
# Panels
# ============================================================================


def plot_manhattan(ax, df: pd.DataFrame, pcol: str) -> None:
    d = df.dropna(subset=[pcol, "CHR", "POS"]).copy()
    d = d[(d[pcol] > 0) & (d[pcol] <= 1)]
    if d.empty:
        ax.text(0.5, 0.5, "no data", ha="center"); return
    d["CHR"] = d["CHR"].astype(str)
    d["mlog"] = -np.log10(d[pcol])
    chrs = sorted(d["CHR"].unique(), key=_chr_key)

    offset, xticks, xticklabels = 0.0, [], []
    for i, c in enumerate(chrs):
        m = d["CHR"] == c
        pos = pd.to_numeric(d.loc[m, "POS"], errors="coerce").astype(float)
        span = (pos.max() - pos.min()) if len(pos) > 1 else 1.0
        x = offset + (pos - pos.min())
        hit = d.loc[m, pcol] < GENOMEWIDE_SIG
        base_c, hit_c = (CHR_GREY, HIT_BURGUNDY) if i % 2 == 0 else (CHR_BLACK, HIT_MINT)
        ax.scatter(x[~hit], d.loc[m, "mlog"][~hit], s=5, c=base_c, linewidths=0, rasterized=True)
        if hit.any():
            ax.scatter(x[hit], d.loc[m, "mlog"][hit], s=22, c=hit_c, edgecolors="k",
                       linewidths=0.3, zorder=3)
        xticks.append(offset + span / 2)
        xticklabels.append(c)
        offset += span * 1.02 + 1

    ax.axhline(-np.log10(GENOMEWIDE_SIG), color=GW_LINE, ls="--", lw=0.8, label="genome-wide 5e-8")
    bonf = ALPHA / len(d)
    ax.axhline(-np.log10(bonf), color=BONF_LINE, ls=":", lw=0.8, label=f"Bonferroni {bonf:.1e}")
    ax.set_xticks(xticks); ax.set_xticklabels(xticklabels, fontsize=6)
    ax.set_xlabel("chromosome"); ax.set_ylabel(r"$-\log_{10}P$")
    ax.set_xlim(-offset * 0.01, offset * 1.01)
    ax.legend(fontsize=6, loc="upper right")


def plot_volcano(ax, df: pd.DataFrame, pcol: str, betacol: str) -> None:
    d = df.dropna(subset=[pcol, betacol]).copy()
    d = d[(d[pcol] > 0) & (d[pcol] <= 1)]
    if d.empty:
        ax.text(0.5, 0.5, "no data", ha="center"); return
    d["mlog"] = -np.log10(d[pcol])
    bonf = ALPHA / len(d)
    sig = d[pcol] < bonf
    ax.scatter(d.loc[~sig, betacol], d.loc[~sig, "mlog"], s=12, c=CHR_GREY, linewidths=0)
    ax.scatter(d.loc[sig, betacol], d.loc[sig, "mlog"], s=26, c=HIT_BURGUNDY,
               edgecolors="k", linewidths=0.3, zorder=3)
    ax.axhline(-np.log10(bonf), color=BONF_LINE, ls=":", lw=0.8, label=f"Bonferroni {bonf:.1e}")
    ax.axvline(0, color="k", lw=0.5)
    texts = []
    for _, r in d.nsmallest(min(12, int(sig.sum()) or 1), pcol).iterrows():
        if r[pcol] < bonf:
            texts.append(ax.text(r[betacol], r["mlog"], str(r["predictor"]), fontsize=6))
    if texts and _HAS_ADJUST:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="grey", lw=0.4))
    ax.set_xlabel("effect (β)"); ax.set_ylabel(r"$-\log_{10}P$")
    ax.legend(fontsize=6, loc="upper right")


def plot_qq(ax, df: pd.DataFrame, pcol: str) -> dict:
    """QQ with a line per MAF bin; returns {bin: λ}. Reports λ in the legend."""
    maf = _maf(df)
    have_maf = maf.notna().any()
    lambdas = {}
    colors = ["#222222", "#2980b9", "#e08a1e", "#8c1d40"]
    max_v = 1.0
    for (label, lo, hi), color in zip(_MAF_BINS, colors):
        d = df.copy()
        d["_p"] = pd.to_numeric(d[pcol], errors="coerce")
        d = d[(d["_p"] > 0) & (d["_p"] <= 1)]
        if label != "all":
            if not have_maf:
                continue
            d = d[(maf.loc[d.index] >= lo) & (maf.loc[d.index] < hi)]
        if len(d) < 20:
            continue
        p = np.sort(d["_p"].to_numpy())
        obs = -np.log10(p)
        exp = -np.log10((np.arange(1, len(p) + 1) - 0.5) / len(p))
        lam = lambda_gc(d["_p"])
        lambdas[label] = lam
        max_v = max(max_v, exp.max(), obs.max())
        ax.plot(exp, obs, ".", ms=2.5, color=color, alpha=0.8,
                label=f"{label}: λ={lam:.3f} (n={len(d):,})")
    ax.plot([0, max_v], [0, max_v], "k-", lw=0.6)
    ax.set_xlabel(r"expected $-\log_{10}P$"); ax.set_ylabel(r"observed $-\log_{10}P$")
    ax.legend(fontsize=6, loc="upper left")
    return lambdas


# ============================================================================
# Figure assembly (one per source: study or meta)
# ============================================================================


def make_figure(df: pd.DataFrame, label: str, out_path: str) -> dict:
    pcol = _pcol(df)
    if pcol is None:
        log.warning(f"  {label}: no usable P/P_FE column — skipping")
        return {}
    genetic = _is_genetic(df)
    fig, (ax_main, ax_qq) = plt.subplots(1, 2, figsize=(15, 5),
                                         gridspec_kw={"width_ratios": [2.2, 1]})
    if genetic:
        plot_manhattan(ax_main, df, pcol)
        ax_main.set_title(f"{label} — Manhattan  ({pcol})", fontsize=10)
    else:
        bcol = _betacol(df)
        if bcol is None:
            plt.close(fig); return {}
        plot_volcano(ax_main, df, pcol, bcol)
        ax_main.set_title(f"{label} — volcano (tabular)  ({pcol})", fontsize=10)
    lambdas = plot_qq(ax_qq, df, pcol)
    ax_qq.set_title(f"{label} — QQ", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return lambdas


# ============================================================================
# Discovery + drivers
# ============================================================================


def _latest_by_key(paths, regex, keyfields):
    best = {}
    for p in paths:
        m = regex.match(os.path.basename(p))
        if not m:
            continue
        key = tuple(m.group(f) for f in keyfields)
        ts = m.group("ts")
        if key not in best or best[key][0] < ts:
            best[key] = (ts, p)
    return best


def run_auto(results_dir: str, meta_dir: str | None, output_dir: str, run_filter: str | None) -> None:
    os.makedirs(output_dir, exist_ok=True)
    study = _latest_by_key(glob.glob(os.path.join(results_dir, "*.csv")),
                           _STUDY_RE, ("name", "stratum")) if results_dir else {}
    meta = _latest_by_key(glob.glob(os.path.join(meta_dir, "*.csv")),
                          _META_RE, ("name",)) if meta_dir and os.path.isdir(meta_dir) else {}
    sources = []
    for (name, stratum), (_ts, path) in study.items():
        sources.append((name, stratum, path))
    for (name,), (_ts, path) in meta.items():
        sources.append((name, "META", path))
    if not sources:
        sys.exit(f"ERROR: no CSVs found in {results_dir} / {meta_dir}")
    log.info(f"{len(sources)} source(s) to plot")
    for name, tag, path in sorted(sources):
        if run_filter and name != run_filter:
            continue
        df = pd.read_csv(path)
        out = os.path.join(output_dir, f"{name}-{tag}-manqq.png")
        lambdas = make_figure(df, f"{name} [{tag}]", out)
        lam_str = "  ".join(f"{k}:λ={v:.3f}" for k, v in lambdas.items())
        log.info(f"  {name} [{tag}] → {os.path.basename(out)}   {lam_str}")


def main() -> None:
    p = argparse.ArgumentParser(description="Manhattan/volcano + QQ for GWAS or meta results")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--meta-dir", default="meta")
    p.add_argument("--output-dir", default="plots")
    p.add_argument("--run", default=None, help="only this run name")
    p.add_argument("--input", default=None, help="plot a single CSV (study or meta) and exit")
    args = p.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, f"manhattan_qq_{timestamp}.log"))

    if args.input:
        df = pd.read_csv(args.input)
        label = os.path.splitext(os.path.basename(args.input))[0]
        out = os.path.join(args.output_dir, f"{label}-manqq.png")
        lambdas = make_figure(df, label, out)
        log.info(f"{label} → {out}   " + "  ".join(f"{k}:λ={v:.3f}" for k, v in lambdas.items()))
    else:
        run_auto(args.results_dir, args.meta_dir, args.output_dir, args.run)
    log.info("Done.")


if __name__ == "__main__":
    main()
