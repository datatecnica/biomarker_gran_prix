#!/usr/bin/env python3
"""genoio.py — plink2 dosage I/O for Biomarker Gran Prix (PLAN §4, §8).

A thin, streaming reader over a plink2 `<prefix>.pgen/.pvar/.psam` fileset:

  * parse `.pvar` (CHR/POS/ID/REF/ALT; effect allele = ALT) and `.psam` sample order,
  * map the analysis subset (IIDs kept after prefilter/stratum) to `.pgen` positions,
  * **block-stream** dosages variant-major — only `block_size` variants × N subjects
    are resident at once (PLAN §8) — with an optional **prefetch** thread so the next
    block decompresses while the GPU scans the current one,
  * compute per-variant **EAF / MAF / MAC / missing-rate / N** on the subset, and
  * pull specific variants by ID (for `condition_snps` and `derive: snp(...)`).

Missing calls read back from pgenlib as the sentinel `-9.0` (not NaN); we normalize
them to NaN and mean-impute (to 2·EAF) only for the scan itself. Dosages are float32
(plink2's 1/16384 grid); EAF/MAF are computed on the non-missing subset.

See [[pgenlib-api-contract]] for the verified writer/reader calls.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import pgenlib
except ImportError as exc:  # pragma: no cover - install guard
    raise ImportError(
        "Pgenlib is required for genetic mode. Install:\n"
        "  python3 -m pip install --user --break-system-packages Pgenlib"
    ) from exc

MISSING_SENTINEL = -9.0  # PgenReader.read_dosages returns this for a missing call


# ============================================================================
# .pvar / .psam parsing
# ============================================================================


def _read_pvar(path: str) -> pd.DataFrame:
    """Parse a .pvar into CHR/POS/ID/REF/ALT, tolerating ## meta and extra columns."""
    header_row = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if line.startswith("##"):
                continue
            header_row = i
            has_header = line.startswith("#")
            break
    if has_header:
        df = pd.read_csv(path, sep="\t", skiprows=header_row, dtype=str)
        df = df.rename(columns={c: c.lstrip("#") for c in df.columns})
        df = df.rename(columns={"CHROM": "CHR"})
    else:
        df = pd.read_csv(path, sep="\t", header=None, comment="#", dtype=str,
                         names=["CHR", "POS", "ID", "REF", "ALT"])
    for need in ("CHR", "POS", "ID", "REF", "ALT"):
        if need not in df.columns:
            raise ValueError(f"{path}: missing required .pvar column {need!r}")
    out = df[["CHR", "POS", "ID", "REF", "ALT"]].copy()
    out["POS"] = pd.to_numeric(out["POS"], errors="coerce").astype("Int64")
    return out.reset_index(drop=True)


def _read_psam(path: str) -> list[str]:
    """Return sample IIDs in .psam (== .pgen) order."""
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f if not ln.startswith("##")]
    header = lines[0].lstrip("#").split("\t")
    rows = [ln.split("\t") for ln in lines[1:] if ln.strip()]
    frame = pd.DataFrame(rows, columns=header)
    iid_col = "IID" if "IID" in frame.columns else frame.columns[0]
    return frame[iid_col].astype(str).tolist()


# ============================================================================
# Per-block variant statistics
# ============================================================================


@dataclass
class BlockStats:
    eaf: np.ndarray           # effect-allele (ALT) frequency on the subset
    maf: np.ndarray           # minor-allele frequency
    mac: np.ndarray           # expected minor-allele count
    missing_rate: np.ndarray  # fraction of subjects with a missing call
    n_obs: np.ndarray         # subjects with a non-missing call


def block_stats(dosages: np.ndarray) -> BlockStats:
    """Per-variant EAF/MAF/MAC/missing on a (N_subjects, B) NaN-missing block."""
    with np.errstate(invalid="ignore"):
        n_obs = np.sum(~np.isnan(dosages), axis=0)
        eaf = np.nanmean(dosages, axis=0) / 2.0
    eaf = np.where(n_obs > 0, eaf, np.nan)
    maf = np.minimum(eaf, 1.0 - eaf)
    mac = maf * 2.0 * n_obs
    missing_rate = 1.0 - n_obs / dosages.shape[0]
    return BlockStats(eaf=eaf, maf=maf, mac=mac, missing_rate=missing_rate, n_obs=n_obs)


def mean_impute(dosages: np.ndarray, eaf: np.ndarray) -> np.ndarray:
    """Replace NaN with the per-variant mean (2·EAF) for the scan. Returns f32 C-order."""
    fill = np.where(np.isnan(eaf), 0.0, 2.0 * eaf).astype(np.float32)
    out = np.where(np.isnan(dosages), fill[np.newaxis, :], dosages)
    return np.ascontiguousarray(out, dtype=np.float32)


# ============================================================================
# Prefetch wrapper (overlap I/O with compute)
# ============================================================================


def _prefetch(gen, depth: int = 2):
    """Run a generator in a background thread, buffering up to `depth` items."""
    q: queue.Queue = queue.Queue(maxsize=depth)
    sentinel = object()

    def worker():
        try:
            for item in gen:
                q.put(item)
        except Exception as exc:  # surface reader errors to the consumer
            q.put((sentinel, exc))
        else:
            q.put(sentinel)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        if isinstance(item, tuple) and len(item) == 2 and item[0] is sentinel:
            raise item[1]
        yield item
    th.join()


# ============================================================================
# Reader
# ============================================================================


@dataclass
class Block:
    start: int                 # first variant index (file order)
    end: int                   # one-past-last variant index
    meta: pd.DataFrame         # .pvar slice for these variants
    dosages: np.ndarray        # (N_kept, B) float32, NaN = missing


class GenoReader:
    """Streaming reader over a plink2 dosage fileset, aligned to an analysis subset."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.variants = _read_pvar(f"{prefix}.pvar")
        self.samples = _read_psam(f"{prefix}.psam")
        self.n_variants = len(self.variants)
        self.n_samples = len(self.samples)
        self._pgen = f"{prefix}.pgen".encode()

        r = pgenlib.PgenReader(self._pgen)
        vc, sc = r.get_variant_ct(), r.get_raw_sample_ct()
        r.close()
        if vc != self.n_variants or sc != self.n_samples:
            raise ValueError(
                f"{prefix}: .pgen has {vc} variants x {sc} samples but "
                f".pvar/.psam say {self.n_variants} x {self.n_samples}"
            )
        self._sample_pos = {s: i for i, s in enumerate(self.samples)}
        self._id_pos = {v: i for i, v in enumerate(self.variants["ID"])}

    # -- sample alignment ---------------------------------------------------

    def align_samples(self, iids) -> tuple[np.ndarray, list[str]]:
        """Map an ordered IID list to (pgen positions, kept IIDs) — order preserved.

        Only IIDs present in the fileset are kept; the returned positions index
        the .pgen sample axis, and the kept-IID list is the caller's row order for
        the design (y, Q) so genotype columns line up with design rows.
        """
        pos, kept = [], []
        for s in iids:
            i = self._sample_pos.get(str(s))
            if i is not None:
                pos.append(i)
                kept.append(str(s))
        return np.asarray(pos, dtype=np.int64), kept

    # -- targeted reads (condition_snps, derive: snp) -----------------------

    def read_ids(self, ids, sample_idx: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Read specific variants by ID for a sample subset.

        Returns (dosages (len(found), N_kept) NaN-missing, found_ids). Unknown IDs
        are skipped (the caller decides whether that is fatal).
        """
        found = [i for i in ids if i in self._id_pos]
        out = np.full((len(found), len(sample_idx)), np.nan, dtype=np.float32)
        if found:
            r = pgenlib.PgenReader(self._pgen)
            buf = np.empty(self.n_samples, dtype=np.float32)
            for k, vid in enumerate(found):
                r.read_dosages(self._id_pos[vid], buf)
                out[k] = buf[sample_idx]
            r.close()
            out[out == MISSING_SENTINEL] = np.nan
        return out, found

    # -- block streaming ----------------------------------------------------

    def iter_blocks(self, sample_idx: np.ndarray, block_size: int, prefetch: bool = True):
        """Yield Block(s) of dosages for the aligned subset, variant-major.

        Reads contiguous variant ranges (efficient `read_dosages_range`), subsets to
        `sample_idx`, and normalizes the missing sentinel to NaN.
        """
        n_kept = len(sample_idx)

        def gen():
            r = pgenlib.PgenReader(self._pgen)
            buf = np.empty((block_size, self.n_samples), dtype=np.float32)
            try:
                for start in range(0, self.n_variants, block_size):
                    end = min(start + block_size, self.n_variants)
                    b = end - start
                    r.read_dosages_range(start, end, buf[:b])
                    dos = buf[:b, sample_idx].T                      # (n_kept, b)
                    dos = np.ascontiguousarray(dos, dtype=np.float32)
                    dos[dos == MISSING_SENTINEL] = np.nan
                    yield Block(start, end, self.variants.iloc[start:end], dos)
            finally:
                r.close()

        if n_kept == 0:
            return
        yield from (_prefetch(gen()) if prefetch else gen())
