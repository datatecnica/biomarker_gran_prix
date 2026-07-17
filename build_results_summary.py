#!/usr/bin/env python3
"""build_results_summary.py — cross-run rollup for Biomarker Gran Prix.

Surveys the per-stratum association CSVs (from gwas.py) and the meta CSVs (from
meta_analysis.py) and emits a human-readable Markdown report:

  * **coverage table** — one row per run: model, N, #variants tested, genome-wide
    hits, and λ-GC for every source (each stratum + FE + RE),
  * **calibration table** — λ-GC by MAF band (all / common / low-freq / rare) for
    each run's primary source, surfacing rare-variant inflation at a glance,
  * **top hits** — genome-wide-significant variants across all runs, deduplicated to
    one row per (run, predictor) and sorted by P globally.

Usage:
    python3 build_results_summary.py [--results-dir results/] [--meta-dir meta/]
                                     [--output results_summary.md]

Design adapts ../proteomics_data_mine/build_results_summary.py, generalized to a
genotype join key and arbitrary strata.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2

GENOMEWIDE_SIG = 5e-8
_STUDY_RE = re.compile(r"^(?P<name>.+)-(?P<stratum>[^-]+)-(?P<ts>\d{8}_\d{6})\.csv$")
_META_RE = re.compile(r"^META_(?P<name>.+)-(?P<ts>\d{8}_\d{6})\.csv$")
_MAF_BANDS = [("all", 0.0, 1.0), ("common", 0.05, 0.5001),
              ("low-freq", 0.01, 0.05), ("rare", 0.0, 0.01)]


def lambda_gc(pvals) -> float:
    p = pd.Series(pvals).dropna()
    p = p[(p > 0) & (p <= 1)]
    if len(p) == 0:
        return float("nan")
    return float(np.median(chi2.ppf(1 - p, df=1)) / chi2.ppf(0.5, df=1))


def _maf(df: pd.DataFrame) -> pd.Series:
    if "MAF" in df.columns and df["MAF"].notna().any():
        return pd.to_numeric(df["MAF"], errors="coerce")
    if "EAF" in df.columns:
        eaf = pd.to_numeric(df["EAF"], errors="coerce")
        return np.minimum(eaf, 1 - eaf)
    return pd.Series(np.nan, index=df.index)


def _latest(paths, regex, keyfields):
    best = {}
    for p in paths:
        m = regex.match(os.path.basename(p))
        if not m:
            continue
        key = tuple(m.group(f) for f in keyfields)
        if key not in best or best[key][0] < m.group("ts"):
            best[key] = (m.group("ts"), p)
    return best


def _fmt(x, spec="{:.3f}"):
    return spec.format(x) if pd.notna(x) and np.isfinite(x) else "—"


def _md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored Markdown table (no tabulate dep)."""
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in df.columns) + " |")
    return "\n".join(rows)


# ============================================================================
# Assembly
# ============================================================================


def build(results_dir: str, meta_dir: str) -> tuple[str, int]:
    study = _latest(glob.glob(os.path.join(results_dir, "*.csv")), _STUDY_RE, ("name", "stratum"))
    meta = _latest(glob.glob(os.path.join(meta_dir, "*.csv")), _META_RE, ("name",)) \
        if os.path.isdir(meta_dir) else {}
    runs: dict[str, dict] = {}
    for (name, stratum), (_ts, path) in study.items():
        runs.setdefault(name, {"strata": {}})["strata"][stratum] = path
    for (name,), (_ts, path) in meta.items():
        runs.setdefault(name, {"strata": {}})["meta"] = path
    if not runs:
        sys.exit(f"ERROR: no result CSVs found in {results_dir} / {meta_dir}")

    coverage, calib, hit_frames = [], [], []
    all_strata: set[str] = set()

    for name in sorted(runs):
        info = runs[name]
        strata_dfs = {s: pd.read_csv(p) for s, p in info["strata"].items()}
        meta_df = pd.read_csv(info["meta"]) if "meta" in info else None
        all_strata.update(strata_dfs)

        any_df = next(iter(strata_dfs.values()), meta_df)
        model = str(any_df["model_type"].iloc[0]) if any_df is not None and "model_type" in any_df else "?"
        # primary source for hit counts / calibration: meta if present else pooled/first stratum
        primary, pcol = (meta_df, "P_FE") if meta_df is not None else (any_df, "P")
        n_tests = len(primary) if primary is not None else 0
        gw = int((pd.to_numeric(primary[pcol], errors="coerce") < GENOMEWIDE_SIG).sum()) \
            if primary is not None and pcol in primary else 0
        Ns = pd.concat([pd.to_numeric(d.get("N"), errors="coerce") for d in strata_dfs.values()]) \
            if strata_dfs else pd.Series(dtype=float)
        n_str = f"{int(Ns.min())}–{int(Ns.max())}" if len(Ns.dropna()) else "—"

        row = {"run": name, "model": model, "N": n_str, "variants": f"{n_tests:,}", "GW hits": gw}
        for s, d in strata_dfs.items():
            row[f"λ_{s}"] = lambda_gc(d["P"]) if "P" in d else np.nan
        if meta_df is not None:
            row["λ_FE"] = lambda_gc(meta_df["P_FE"]) if "P_FE" in meta_df else np.nan
            row["λ_RE"] = lambda_gc(meta_df["P_RE"]) if "P_RE" in meta_df else np.nan
        coverage.append(row)

        # MAF-band calibration on the primary source
        if primary is not None and pcol in primary:
            maf = _maf(primary)
            cr = {"run": name, "source": "FE" if meta_df is not None else "pooled"}
            for label, lo, hi in _MAF_BANDS:
                sub = primary if label == "all" else primary[(maf >= lo) & (maf < hi)]
                cr[f"λ_{label}"] = lambda_gc(sub[pcol]) if len(sub) >= 20 else np.nan
            calib.append(cr)

        # genome-wide hits across sources
        for src, d, pc, bc in ([(s, d, "P", "beta") for s, d in strata_dfs.items()]
                               + ([("FE", meta_df, "P_FE", "beta_FE")] if meta_df is not None else [])):
            if d is None or pc not in d:
                continue
            sig = d[pd.to_numeric(d[pc], errors="coerce") < GENOMEWIDE_SIG].copy()
            if sig.empty:
                continue
            hit_frames.append(pd.DataFrame({
                "run": name, "predictor": sig["predictor"], "source": src,
                "CHR": sig.get("CHR"), "POS": sig.get("POS"),
                "beta": pd.to_numeric(sig.get(bc), errors="coerce"),
                "P": pd.to_numeric(sig[pc], errors="coerce")}))

    # dedup hits: best P per (run, predictor)
    if hit_frames:
        hits = pd.concat(hit_frames, ignore_index=True)
        hits = hits.sort_values("P").drop_duplicates(["run", "predictor"], keep="first")
        hits = hits.sort_values("P").reset_index(drop=True)
    else:
        hits = pd.DataFrame()

    # --- render markdown ---
    strata_cols = sorted(all_strata)
    cov_cols = ["run", "model", "N", "variants", "GW hits"] \
        + [f"λ_{s}" for s in strata_cols] + ["λ_FE", "λ_RE"]
    cov = pd.DataFrame(coverage)
    for c in cov_cols:
        if c not in cov:
            cov[c] = np.nan
    for c in cov_cols:
        if c.startswith("λ_"):
            cov[c] = cov[c].map(lambda v: _fmt(v))
    cov = cov[cov_cols]

    lines = ["# Biomarker Gran Prix — results summary\n"]
    lines.append(f"Runs surveyed: **{len(coverage)}** · genome-wide hits (5e-8): "
                 f"**{len(hits)}** unique (run, predictor).\n")
    lines.append("## Coverage & calibration (λ-GC per source)\n")
    lines.append(cov.pipe(_md_table))
    lines.append("\n> λ close to 1.0 = well-calibrated. FE/RE are the meta fixed/random-effects "
                 "λ (RE is conservative at small #studies).\n")

    if calib:
        cal = pd.DataFrame(calib)
        for c in [c for c in cal.columns if c.startswith("λ_")]:
            cal[c] = cal[c].map(lambda v: _fmt(v))
        lines.append("## λ-GC by MAF band (primary source)\n")
        lines.append(cal.pipe(_md_table))
        lines.append("\n> Rare/low-freq inflation above the common band flags residual "
                     "stratification or score-test miscalibration in that band.\n")

    lines.append("## Genome-wide-significant hits (deduped, sorted by P)\n")
    if hits.empty:
        lines.append("_None below 5e-8._\n")
    else:
        show = hits.head(50).copy()
        show["beta"] = show["beta"].map(lambda x: _fmt(x, "{:+.3g}"))
        show["P"] = show["P"].map(lambda x: f"{x:.2e}")
        show["CHR"] = show["CHR"].map(lambda x: str(int(x)) if pd.notna(x) else "—")
        show["POS"] = show["POS"].map(lambda x: f"{int(x):,}" if pd.notna(x) else "—")
        lines.append(show[["run", "predictor", "CHR", "POS", "source", "beta", "P"]]
                     .pipe(_md_table))
        if len(hits) > 50:
            lines.append(f"\n_…{len(hits) - 50} more hits omitted._\n")
    return "\n".join(lines) + "\n", len(hits)


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-run rollup of GWAS + meta results")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--meta-dir", default="meta")
    p.add_argument("--output", default="results_summary.md")
    args = p.parse_args()
    md, n_hits = build(args.results_dir, args.meta_dir)
    with open(args.output, "w") as f:
        f.write(md)
    print(f"Wrote {args.output} ({n_hits} genome-wide hits).")


if __name__ == "__main__":
    main()
