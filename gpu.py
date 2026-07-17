#!/usr/bin/env python3
"""gpu.py — compute backend for Biomarker Gran Prix (device + resource policy).

The engine's numerical work (residualize-then-scan linear/logistic/GALLOP/slope/
cox-score kernels + SPA) runs on a **PyTorch** backend that transparently uses the
GPU when present and the identical CPU code path otherwise. This module is the
foundation those kernels sit on; in Phase 0 it provides:

  * `resolve_device(...)`  — pick cuda/cpu with **automatic CPU fallback** when CUDA
    is absent, was requested but is unavailable, or is present but cannot actually
    run a kernel (e.g. a torch build lacking this GPU's compute capability).
  * `detect_resources(...)` — logical cores, total/available host RAM, free/total
    VRAM, and whether the GPU is functionally usable.
  * `resolve_compute(...)`  — turn the config's resource knobs
    (`device`, `max_cpus`, `max_ram_gb`, `max_gpu_mem_gb`, `block_size`) into a
    concrete `ComputeConfig`. Unset caps (`None`) mean *use the whole machine*; any
    cap the user sets is a **hard ceiling** (PLAN §8). `block_size: auto` picks the
    largest SNP block that fits the memory budget for the current subject count N.

Run directly for a self-check / probe:
    python3 gpu.py                       # full smoke: GPU reachable, CPU fallback,
                                         #   auto-sizing respects caps & maxes when null
    python3 gpu.py --n-subjects 20000    # show the resolved compute config for N
    python3 gpu.py --device cpu --max-cpus 4 --max-ram-gb 8 --n-subjects 5000

Design: mirrors ../proteomics_data_mine's flat-script style (module logger,
`setup_logging`, argparse `main`). Reductions that feed β/SE are accumulated in
float64 by the kernels (Phase 2+); this module only sizes and places the work.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover - install guard
    sys.exit(
        "ERROR: PyTorch is not installed. Install the CUDA 12.8 build (RTX 5080 / "
        "Blackwell needs cu128) or the CPU build:\n"
        "  python3 -m pip install --user torch --index-url "
        "https://download.pytorch.org/whl/cu128\n"
        f"(import error: {exc})"
    )

import numpy as np

log = logging.getLogger("gpu")

# ============================================================================
# Resource / block-sizing policy constants
# ============================================================================

# Fraction of the *free* VRAM / *total* RAM used as the default compute budget
# when the corresponding cap is unset. Leaves ~10% headroom for the CUDA context,
# allocator fragmentation, and the OS.
VRAM_SAFETY_FRACTION = 0.90
RAM_SAFETY_FRACTION = 0.90

# Fraction of the *memory budget* handed to the single streaming genotype block.
# The remainder covers the prefetch double-buffer (a second in-flight block),
# a standardized/residualized working copy, the covariate/design & residual
# matrices, and GEMM temporaries — so ~4 block-equivalents fit under the budget.
BLOCK_MEM_FRACTION = 0.25

# Clamp + rounding for the resolved block size (SNPs per block).
BLOCK_MIN = 512
BLOCK_MAX = 32768
BLOCK_MULTIPLE = 256

_BYTES_PER_DOSAGE = 4  # float32 dosage column, one value per subject

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


# ============================================================================
# Low-level detection helpers
# ============================================================================


def _detect_cpus() -> int:
    """Logical cores actually available to this process (respects taskset/cgroup)."""
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _detect_ram_gb() -> tuple[float, float]:
    """(total_ram_gb, available_ram_gb). Available falls back to total if unknown."""
    total_bytes: int | None = None
    avail_bytes: int | None = None

    # /proc/meminfo gives MemTotal + MemAvailable (Linux); values are in kB.
    try:
        with open("/proc/meminfo") as f:
            info: dict[str, int] = {}
            for line in f:
                key, _, rest = line.partition(":")
                fields = rest.split()
                if fields:
                    info[key.strip()] = int(fields[0]) * 1024
        total_bytes = info.get("MemTotal")
        avail_bytes = info.get("MemAvailable")
    except (OSError, ValueError):
        pass

    # Portable fallback for total via sysconf.
    if total_bytes is None:
        try:
            total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, AttributeError, OSError):
            total_bytes = 0

    if avail_bytes is None:
        avail_bytes = total_bytes
    return total_bytes / 1e9, avail_bytes / 1e9


def _cuda_functional() -> bool:
    """True only if a CUDA device is present AND can actually run a kernel.

    `torch.cuda.is_available()` can be True while kernel launches fail (e.g. a
    torch build without this GPU's compute capability — the classic "no kernel
    image is available for execution on the device" on a new architecture). We
    probe with a tiny op so the fallback is based on reality, not just presence.
    """
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.ones(8, device="cuda")
        _ = float((x * 2).sum().item())
        return True
    except Exception as exc:  # RuntimeError, and defensively anything else
        log.warning("CUDA is present but not functional (%s) — will use CPU.", exc)
        return False


def _detect_vram_gb(device: torch.device) -> tuple[float | None, float | None]:
    """(free_vram_gb, total_vram_gb) for a cuda device; (None, None) otherwise."""
    if device.type != "cuda":
        return None, None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return free_bytes / 1e9, total_bytes / 1e9
    except Exception as exc:  # pragma: no cover - driver edge cases
        log.warning("Could not query VRAM (%s).", exc)
        return None, None


# ============================================================================
# Device resolution (with automatic CPU fallback)
# ============================================================================


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve `device:` (auto|cuda|cpu) to a torch.device, falling back to CPU.

    * ``cpu``           -> CPU always.
    * ``cuda``/``gpu``  -> CUDA if functional, else a warned CPU fallback.
    * ``auto``          -> CUDA if functional, else CPU.
    """
    req = (requested or "auto").strip().lower()
    if req == "cpu":
        return torch.device("cpu")

    functional = _cuda_functional()
    if req in ("cuda", "gpu"):
        if functional:
            return torch.device("cuda")
        log.warning("device='%s' requested but no functional CUDA device — using CPU.", requested)
        return torch.device("cpu")

    if req != "auto":
        log.warning("Unknown device='%s'; treating as 'auto'.", requested)
    return torch.device("cuda") if functional else torch.device("cpu")


# ============================================================================
# Resource snapshot + resolved compute config
# ============================================================================


@dataclass
class Resources:
    """What the machine actually has, for the resolved device."""

    device: torch.device
    n_cpus: int
    total_ram_gb: float
    avail_ram_gb: float
    cuda_functional: bool
    gpu_name: str | None = None
    free_vram_gb: float | None = None
    total_vram_gb: float | None = None


def detect_resources(device: str = "auto") -> Resources:
    """Snapshot cores/RAM/VRAM for the device that `device` resolves to."""
    dev = resolve_device(device)
    n_cpus = _detect_cpus()
    total_ram, avail_ram = _detect_ram_gb()

    gpu_name = None
    free_vram = total_vram = None
    if dev.type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(dev)
        except Exception:  # pragma: no cover
            gpu_name = "CUDA device"
        free_vram, total_vram = _detect_vram_gb(dev)

    return Resources(
        device=dev,
        n_cpus=n_cpus,
        total_ram_gb=total_ram,
        avail_ram_gb=avail_ram,
        cuda_functional=(dev.type == "cuda"),
        gpu_name=gpu_name,
        free_vram_gb=free_vram,
        total_vram_gb=total_vram,
    )


@dataclass
class ComputeConfig:
    """Fully-resolved compute policy handed to the engine."""

    device: torch.device
    n_cpus: int                 # effective worker count (after cap)
    ram_budget_gb: float        # host-buffer budget (after cap)
    vram_budget_gb: float | None  # VRAM budget (after cap); None on CPU
    block_size: int | None      # SNPs per streamed block; None if N unknown & auto
    n_subjects: int | None
    resources: Resources = field(repr=False)
    caps: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def on_gpu(self) -> bool:
        return self.device.type == "cuda"

    def block_bytes(self) -> int | None:
        """VRAM (or host RAM on CPU) held by one genotype block, in bytes."""
        if self.block_size is None or self.n_subjects is None:
            return None
        return self.n_subjects * self.block_size * _BYTES_PER_DOSAGE


def auto_block_size(
    n_subjects: int,
    mem_budget_gb: float,
    *,
    block_fraction: float = BLOCK_MEM_FRACTION,
    block_min: int = BLOCK_MIN,
    block_max: int = BLOCK_MAX,
    multiple: int = BLOCK_MULTIPLE,
) -> int:
    """Largest SNP block (float32, N subjects/column) fitting the memory budget.

    block_bytes ≈ N × block_size × 4; we spend `block_fraction` of the budget on
    the block and reserve the rest for the double-buffer, working copies, and
    design matrices (PLAN §8). Result is rounded down to a multiple and clamped
    to [block_min, block_max].
    """
    if n_subjects is None or n_subjects <= 0:
        raise ValueError("auto_block_size needs a positive n_subjects")
    if mem_budget_gb <= 0:
        raise ValueError("auto_block_size needs a positive memory budget")

    budget_bytes = mem_budget_gb * 1e9 * block_fraction
    per_variant_bytes = n_subjects * _BYTES_PER_DOSAGE
    raw = int(budget_bytes // per_variant_bytes)
    raw = (raw // multiple) * multiple  # round down to a whole multiple

    if raw < block_min:
        needed = block_min * per_variant_bytes / (block_fraction * 1e9)
        log.warning(
            "Memory budget %.2f GB is tight for N=%d: block clamped up to the "
            "floor of %d SNPs (wants ~%.2f GB budget). Consider a larger "
            "max_gpu_mem_gb/max_ram_gb or fewer subjects.",
            mem_budget_gb, n_subjects, block_min, needed,
        )
        return block_min
    return min(block_max, raw)


def resolve_compute(
    n_subjects: int | None = None,
    *,
    device: str = "auto",
    max_cpus: int | None = None,
    max_ram_gb: float | None = None,
    max_gpu_mem_gb: float | None = None,
    block_size: Any = "auto",
) -> ComputeConfig:
    """Resolve the config's resource knobs into a concrete ComputeConfig.

    Unset caps (None) -> use ~the whole machine. Any cap that is set is a hard
    ceiling: it can only *lower* the effective budget, never raise it above what
    the machine actually has (PLAN §8).
    """
    res = detect_resources(device)
    dev = res.device

    # --- CPU workers -------------------------------------------------------
    n_cpus = res.n_cpus if max_cpus is None else max(1, min(res.n_cpus, int(max_cpus)))

    # --- host RAM budget ---------------------------------------------------
    ram_default = RAM_SAFETY_FRACTION * res.total_ram_gb
    ram_budget = ram_default if max_ram_gb is None else min(ram_default, float(max_ram_gb))

    # --- VRAM budget -------------------------------------------------------
    if dev.type == "cuda" and res.free_vram_gb:
        vram_default = VRAM_SAFETY_FRACTION * res.free_vram_gb
        vram_budget = (
            vram_default if max_gpu_mem_gb is None else min(vram_default, float(max_gpu_mem_gb))
        )
    else:
        vram_budget = None

    # --- block size --------------------------------------------------------
    # On GPU the block lives in VRAM; on CPU it lives in host RAM (same logic).
    mem_budget = vram_budget if dev.type == "cuda" else ram_budget
    if isinstance(block_size, str) and block_size.strip().lower() == "auto":
        bsize: int | None = (
            auto_block_size(n_subjects, mem_budget) if (n_subjects and mem_budget) else None
        )
    else:
        bsize = max(1, int(block_size))

    return ComputeConfig(
        device=dev,
        n_cpus=n_cpus,
        ram_budget_gb=ram_budget,
        vram_budget_gb=vram_budget,
        block_size=bsize,
        n_subjects=n_subjects,
        resources=res,
        caps={
            "device": device,
            "max_cpus": max_cpus,
            "max_ram_gb": max_ram_gb,
            "max_gpu_mem_gb": max_gpu_mem_gb,
            "block_size": block_size,
        },
    )


def describe_compute(cfg: ComputeConfig) -> str:
    """Human-readable one-block summary for the run log."""
    res = cfg.resources
    lines = ["Compute configuration:"]
    lines.append(f"  device        : {cfg.device}")
    if cfg.on_gpu:
        lines.append(f"  gpu           : {res.gpu_name}")
        vram = f"{res.free_vram_gb:.1f} free / {res.total_vram_gb:.1f} GB total" \
            if res.free_vram_gb is not None else "unknown"
        lines.append(f"  vram          : {vram}")
        lines.append(f"  vram budget   : {cfg.vram_budget_gb:.1f} GB")
    lines.append(f"  cpu workers   : {cfg.n_cpus} / {res.n_cpus} logical cores")
    lines.append(f"  ram           : {res.avail_ram_gb:.1f} avail / {res.total_ram_gb:.1f} GB total")
    lines.append(f"  ram budget    : {cfg.ram_budget_gb:.1f} GB")
    if cfg.block_size is not None:
        bb = cfg.block_bytes()
        bb_str = f" (~{bb / 1e9:.2f} GB/block at N={cfg.n_subjects})" if bb else ""
        lines.append(f"  block_size    : {cfg.block_size} SNPs{bb_str}")
    else:
        lines.append("  block_size    : auto (deferred — needs subject count N)")
    return "\n".join(lines)


# ============================================================================
# Batched association kernels (residualize-then-scan)
# ============================================================================
#
# Every model shares the pattern: fit the null once on CPU, then batch the
# per-predictor association across a block on the device (PLAN §5, §7). The linear
# kernel is the foundation — `linear` and `slope` use it directly, and the
# score-test models reuse the same residualize/reduce primitives.
#
# Frisch–Waugh–Lovell: to test predictor g controlling covariates Q, residualize
# both y and g on Q, then regress the residuals. With Qo = orthonormal basis of Q,
#   β = (g̃·ỹ)/(g̃·g̃),   RSS = ỹ·ỹ − β·(g̃·ỹ),   SE = sqrt(RSS/dof / (g̃·g̃)).
# The null (Qo, ỹ, ỹ·ỹ) is computed once; each block costs two GEMMs (residualize)
# plus two column reductions. Storage is float32; reductions accumulate in float64
# for a stable significant tail (PLAN §7). P-values are computed on CPU in float64.


@dataclass
class LinearNull:
    """Residualized null model, reused across every block of a scan."""

    Qo: "torch.Tensor"       # (N, k) orthonormal basis of the design (device, f32)
    y_resid: "torch.Tensor"  # (N,) outcome residualized on Q (device, f32)
    yty: float               # ỹ·ỹ  (float64 scalar)
    n: int                   # subjects
    k: int                   # design columns (intercept + covariates + conditioning)
    device: "torch.device"


def prepare_linear_null(y, Q, device: "torch.device") -> LinearNull:
    """Build the residualized null: QR of the design, residualize the outcome."""
    yt = torch.as_tensor(np.asarray(y, dtype=np.float32), device=device).reshape(-1)
    Qt = torch.as_tensor(np.asarray(Q, dtype=np.float32), device=device)
    if Qt.ndim == 1:
        Qt = Qt.reshape(-1, 1)
    Qo, _ = torch.linalg.qr(Qt, mode="reduced")          # (N, k) orthonormal
    y_resid = yt - Qo @ (Qo.t() @ yt)
    yty = float((y_resid.double() @ y_resid.double()).item())
    return LinearNull(Qo=Qo, y_resid=y_resid, yty=yty, n=Qt.shape[0], k=Qt.shape[1], device=device)


def scan_linear(G, null: LinearNull):
    """Batched linear association of each column of G against the null.

    G: (N, B) float32 device tensor — predictors with missing already imputed
    (raw dosages for genetic, z-scored for tabular). Returns numpy float64
    (beta, se, t) length B plus the integer dof; monomorphic/degenerate columns
    come back as NaN for the downstream numerical filter to drop.
    """
    Gt = G if isinstance(G, torch.Tensor) else torch.as_tensor(np.asarray(G, dtype=np.float32),
                                                               device=null.device)
    if Gt.device != null.device:
        Gt = Gt.to(null.device)
    Qo = null.Qo
    G_resid = Gt - Qo @ (Qo.t() @ Gt)                    # (N, B) — the two residualizing GEMMs
    # column reductions accumulated in float64
    gg = (G_resid * G_resid).sum(0, dtype=torch.float64)                 # residual SS (B,)
    gy = (G_resid * null.y_resid.unsqueeze(1)).sum(0, dtype=torch.float64)  # (B,)
    dof = null.n - null.k - 1
    # Collinearity guard: a predictor whose residual variance is ~0 relative to its
    # total (centered) variance is (nearly) explained by the covariates — e.g. a SNP
    # being conditioned on, or a constant column. Its β is undefined (0/0), so mark it
    # NaN for the numerical filter rather than letting float32 noise blow β up.
    g_center = Gt - Gt.mean(0, keepdim=True)
    g_tss = (g_center * g_center).sum(0, dtype=torch.float64)
    valid = (g_tss > 0) & (gg > 1e-6 * g_tss)
    gg_safe = torch.where(valid, gg, torch.full_like(gg, float("nan")))
    beta = gy / gg_safe
    rss = torch.clamp(null.yty - beta * gy, min=0.0)
    sigma2 = rss / max(dof, 1)
    se = torch.sqrt(sigma2 / gg_safe)
    t = beta / se
    return (beta.cpu().numpy(), se.cpu().numpy(), t.cpu().numpy(), int(dof))


# ---------------------------------------------------------------------------
# Logistic score test (batched)
# ---------------------------------------------------------------------------
#
# The null logistic model (case/control ~ covariates) is fit once on CPU, giving
# working residuals r = y − p̂ and weights w = p̂(1−p̂). Adding a predictor g, the
# efficient score and its covariate-adjusted variance are
#     U = gᵀr,   V = gᵀWg − (gᵀWQ)(QᵀWQ)⁻¹(QᵀWg),   T = U²/V ~ χ²₁,
# because the null MLE makes Qᵀr = 0. Each block is one GEMV (U), one weighted
# reduction (gᵀWg), and one small GEMM + contraction (the correction) — all
# batched. The score β̂ ≈ U/V is a one-Newton-step approximation; genome-wide hits
# are refit exactly (Firth/full) downstream, and the rare/tail P is refined by SPA.


@dataclass
class LogisticNull:
    r: "torch.Tensor"          # (N,) working residual y − p̂ (device, f32)
    w: "torch.Tensor"          # (N,) weight p̂(1−p̂) (device, f32)
    WQ: "torch.Tensor"         # (N, k) w·Q (device, f32)
    QtWQ_inv: "torch.Tensor"   # (k, k) (QᵀWQ)⁻¹ (device, f32)
    n: int
    k: int
    device: "torch.device"
    # kept on host for the SPA tail / Firth refit
    p_hat: np.ndarray = field(repr=False, default=None)
    Q: np.ndarray = field(repr=False, default=None)
    y: np.ndarray = field(repr=False, default=None)
    converged: bool = True


def prepare_logistic_null(y, Q, device: "torch.device", max_iter: int = 100) -> LogisticNull:
    """Fit the null logistic model (IRLS) of y on Q; cache residual/weight pieces."""
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    Qm = np.asarray(Q, dtype=np.float64)
    if Qm.ndim == 1:
        Qm = Qm.reshape(-1, 1)
    n, k = Qm.shape
    beta = np.zeros(k)
    p = np.full(n, yv.mean())
    converged = False
    for _ in range(max_iter):
        eta = Qm @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        p = np.clip(p, 1e-8, 1 - 1e-8)
        w = p * (1 - p)
        # IRLS working response
        z = eta + (yv - p) / w
        QtW = Qm.T * w
        try:
            beta_new = np.linalg.solve(QtW @ Qm, QtW @ z)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            converged = True
            break
        beta = beta_new
    eta = Qm @ beta
    p = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-8, 1 - 1e-8)
    w = p * (1 - p)
    r = yv - p
    QtWQ_inv = np.linalg.inv((Qm.T * w) @ Qm)

    def dev(a, dt=torch.float32):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)

    return LogisticNull(
        r=dev(r), w=dev(w), WQ=dev(Qm * w[:, None]), QtWQ_inv=dev(QtWQ_inv),
        n=n, k=k, device=device, p_hat=p, Q=Qm, y=yv, converged=converged,
    )


def scan_logistic(G, null: LogisticNull):
    """Batched logistic score test of each column of G. Returns numpy float64
    (U, V, chi2, beta_score, se_score) length B — degenerate columns are NaN."""
    Gt = G if isinstance(G, torch.Tensor) else torch.as_tensor(
        np.asarray(G, dtype=np.float32), device=null.device)
    if Gt.device != null.device:
        Gt = Gt.to(null.device)
    U = (Gt * null.r.unsqueeze(1)).sum(0, dtype=torch.float64)               # gᵀr  (B,)
    gWg = (null.w.unsqueeze(1) * Gt * Gt).sum(0, dtype=torch.float64)        # gᵀWg (B,)
    GtWQ = (Gt.t().double()) @ null.WQ.double()                             # (B, k)
    corr = ((GtWQ @ null.QtWQ_inv.double()) * GtWQ).sum(1)                   # (B,)
    V = gWg - corr
    valid = V > 1e-12
    V_safe = torch.where(valid, V, torch.full_like(V, float("nan")))
    chi2_stat = U * U / V_safe
    beta = U / V_safe
    se = torch.sqrt(1.0 / V_safe)
    return (U.cpu().numpy(), V.cpu().numpy(), chi2_stat.cpu().numpy(),
            beta.cpu().numpy(), se.cpu().numpy())


# ---------------------------------------------------------------------------
# GALLOP batched kernel (SNP main + SNP×time in one 2×2 GLS per variant)
# ---------------------------------------------------------------------------
#
# From the null RI+slope LMM (lmm.py), each subject i contributes precomputed
# 2-vectors/matrices: P_i = Z_iᵀV_i⁻¹Z_i, q_i = Z_iᵀV_i⁻¹r_i, M_i = Z_iᵀV_i⁻¹X_i, and
# the global A⁻¹ = (Σ X_iᵀV_i⁻¹X_i)⁻¹. Since a SNP's design is G_i·Z_i, for each SNP:
#   C_s = Σ G_i² P_i,   B_s = Σ G_i M_i,   rhs_s = Σ G_i q_i,
#   [β_SNP, β_SNP×TIME] = (C_s − B_s A⁻¹ B_sᵀ)⁻¹ rhs_s,   Cov = (C_s − B_s A⁻¹ B_sᵀ)⁻¹.
# The three sums are GEMMs over subjects (batched across the SNP block); the final
# per-SNP 2×2 solve is done in closed form with a determinant guard.


@dataclass
class GallopNull:
    P_flat: "torch.Tensor"   # (N,4)  per-subject Z_iᵀV_i⁻¹Z_i, flattened
    q: "torch.Tensor"        # (N,2)  per-subject Z_iᵀV_i⁻¹r_i
    M_flat: "torch.Tensor"   # (N,2p) per-subject Z_iᵀV_i⁻¹X_i, flattened
    Ainv: "torch.Tensor"     # (p,p)
    p: int
    n: int
    device: "torch.device"


def prepare_gallop_null(P, q, M, Ainv, device: "torch.device") -> GallopNull:
    """Move the per-subject GALLOP projections (from lmm.gallop_precompute) to device."""
    n = P.shape[0]
    p = M.shape[2]
    def dev(a):
        return torch.as_tensor(np.ascontiguousarray(a, dtype=np.float32), device=device)
    return GallopNull(P_flat=dev(P.reshape(n, 4)), q=dev(q), M_flat=dev(M.reshape(n, 2 * p)),
                      Ainv=dev(Ainv), p=p, n=n, device=device)


def scan_gallop(G, null: GallopNull):
    """Batched GALLOP scan. G: (N,B) per-subject genotype. Returns numpy float64
    (beta_SNP, se_SNP, beta_SNPxTIME, se_SNPxTIME); degenerate variants are NaN."""
    Gt = G if isinstance(G, torch.Tensor) else torch.as_tensor(
        np.asarray(G, dtype=np.float32), device=null.device)
    if Gt.device != null.device:
        Gt = Gt.to(null.device)
    p = null.p
    G2 = Gt * Gt
    # subject sums (float64 accumulation)
    C = (G2.t().double() @ null.P_flat.double()).reshape(-1, 2, 2)      # (B,2,2)
    rhs = Gt.t().double() @ null.q.double()                             # (B,2)
    Bmat = (Gt.t().double() @ null.M_flat.double()).reshape(-1, 2, p)   # (B,2,p)
    Ainv = null.Ainv.double()
    BA = torch.einsum("bip,pq->biq", Bmat, Ainv)                        # (B,2,p)
    BAB = torch.einsum("biq,bjq->bij", BA, Bmat)                        # (B,2,2)
    Info = C - BAB
    a, b = Info[:, 0, 0], Info[:, 0, 1]
    c, d = Info[:, 1, 0], Info[:, 1, 1]
    det = a * d - b * c
    ok = torch.isfinite(det) & (det.abs() > 1e-12) & (a > 0) & (d > 0)
    det_safe = torch.where(ok, det, torch.full_like(det, float("nan")))
    # Info^{-1} (= Cov of [β_SNP, β_SNPxTIME])
    i00, i01 = d / det_safe, -b / det_safe
    i10, i11 = -c / det_safe, a / det_safe
    r0, r1 = rhs[:, 0], rhs[:, 1]
    beta_snp = i00 * r0 + i01 * r1
    beta_snpxt = i10 * r0 + i11 * r1
    se_snp = torch.sqrt(i00)
    se_snpxt = torch.sqrt(i11)
    return (beta_snp.cpu().numpy(), se_snp.cpu().numpy(),
            beta_snpxt.cpu().numpy(), se_snpxt.cpu().numpy())


# ============================================================================
# Self-check / smoke (Phase 0 acceptance)
# ============================================================================


def _matmul_agrees(device: torch.device, n: int = 512, seed: int = 0) -> tuple[bool, float]:
    """Run a matmul on `device`, compare to a numpy float64 reference."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    a_cpu = torch.randn(n, n, generator=gen, dtype=torch.float32)
    b_cpu = torch.randn(n, n, generator=gen, dtype=torch.float32)
    got = (a_cpu.to(device) @ b_cpu.to(device)).cpu().numpy()
    ref = a_cpu.numpy().astype(np.float64) @ b_cpu.numpy().astype(np.float64)
    denom = np.abs(ref).max() or 1.0
    max_rel = float(np.abs(got.astype(np.float64) - ref).max() / denom)
    return max_rel < 1e-3, max_rel


def smoke() -> bool:
    """Phase 0 acceptance checks. Returns True iff every check passes."""
    ok = True
    res = detect_resources("auto")

    log.info("=" * 68)
    log.info("Biomarker Gran Prix — gpu.py Phase 0 smoke test")
    log.info("=" * 68)
    log.info("Detected: %d cores, %.1f/%.1f GB RAM (avail/total)",
             res.n_cpus, res.avail_ram_gb, res.total_ram_gb)
    if res.device.type == "cuda":
        log.info("Detected GPU: %s, %.1f/%.1f GB VRAM (free/total)",
                 res.gpu_name, res.free_vram_gb or 0.0, res.total_vram_gb or 0.0)
    else:
        log.info("No functional CUDA device — CPU-only machine or torch/driver mismatch.")

    # -- Check 1: GPU reachable (skipped-but-noted on CPU-only hosts) --------
    if res.device.type == "cuda":
        agree, rel = _matmul_agrees(torch.device("cuda"))
        log.info("[%s] GPU reachable: 512x512 matmul, max rel err %.2e",
                 "PASS" if agree else "FAIL", rel)
        ok &= agree
    else:
        log.info("[SKIP] GPU reachable: no CUDA device on this host (CPU path still verified below)")

    # -- Check 2: CPU fallback works (identical code path on device=cpu) -----
    cpu_dev = resolve_device("cpu")
    agree, rel = _matmul_agrees(cpu_dev)
    log.info("[%s] CPU fallback: same kernel on %s, max rel err %.2e",
             "PASS" if agree else "FAIL", cpu_dev, rel)
    ok &= agree

    # request 'cuda' on a host without one must fall back, not crash
    if res.device.type != "cuda":
        fb = resolve_device("cuda")
        fb_ok = fb.type == "cpu"
        log.info("[%s] cuda-requested-but-absent falls back to %s",
                 "PASS" if fb_ok else "FAIL", fb)
        ok &= fb_ok

    # -- Check 3: auto block-size respects caps and maxes when null ----------
    log.info("-" * 68)
    log.info("Auto block-size (device=%s):", res.device.type)
    for n in (2_000, 20_000):
        full = resolve_compute(n, device="auto")            # null caps -> whole machine
        mem_key = "max_gpu_mem_gb" if full.on_gpu else "max_ram_gb"
        capped = resolve_compute(n, device="auto", **{mem_key: 1.0})  # tiny 1 GB cap

        budget_full = full.vram_budget_gb if full.on_gpu else full.ram_budget_gb
        budget_cap = capped.vram_budget_gb if capped.on_gpu else capped.ram_budget_gb
        log.info("  N=%-6d  null-caps: block=%-6d (%s budget %.1f GB)  |  1GB-cap: block=%-6d (budget %.1f GB)",
                 n, full.block_size, mem_key.replace("max_", "").replace("_gb", ""),
                 budget_full, capped.block_size, budget_cap)

        checks = {
            "block>0 (null)": full.block_size and full.block_size > 0,
            "block>0 (cap)": capped.block_size and capped.block_size > 0,
            "cap <= null": capped.block_size <= full.block_size,
        }
        # A binding cap must strictly shrink the block (unless already at BLOCK_MIN).
        if budget_cap < budget_full and full.block_size > BLOCK_MIN:
            checks["cap shrinks block"] = capped.block_size < full.block_size
        for label, passed in checks.items():
            if not passed:
                log.info("    [FAIL] %s", label)
                ok = False

    # caps are hard ceilings; nulls take ~all of the machine. Compare each config
    # against *its own* resource snapshot: free VRAM drifts between calls (the CUDA
    # context + prior allocations move it), so budgets are validated against the
    # free/total the config actually saw, not a stale outer snapshot.
    full = resolve_compute(20_000, device="auto")
    fr = full.resources
    cap_checks = {
        "null cpus == all cores": full.n_cpus == fr.n_cpus,
        "null ram budget ~= 90% total":
            abs(full.ram_budget_gb - RAM_SAFETY_FRACTION * fr.total_ram_gb) < 1e-6,
        "cpu cap is a ceiling":
            resolve_compute(20_000, device="auto", max_cpus=2).n_cpus == min(2, fr.n_cpus),
        "ram cap is a ceiling":
            resolve_compute(20_000, device="auto", max_ram_gb=4.0).ram_budget_gb
            == min(4.0, RAM_SAFETY_FRACTION * fr.total_ram_gb),
    }
    if res.device.type == "cuda":
        cap_checks["null vram budget ~= 90% free"] = (
            abs(full.vram_budget_gb - VRAM_SAFETY_FRACTION * (fr.free_vram_gb or 0)) < 1e-6
        )
        # A 2 GB cap binds only if it is below 90% of free; assert the ceiling took.
        capped_vram = resolve_compute(20_000, device="auto", max_gpu_mem_gb=2.0)
        cap_checks["vram cap is a ceiling"] = (
            capped_vram.vram_budget_gb
            == min(2.0, VRAM_SAFETY_FRACTION * (capped_vram.resources.free_vram_gb or 0))
        )
    log.info("-" * 68)
    for label, passed in cap_checks.items():
        log.info("  [%s] %s", "PASS" if passed else "FAIL", label)
        ok &= bool(passed)

    log.info("=" * 68)
    log.info("Phase 0 smoke: %s", "ALL CHECKS PASSED" if ok else "FAILURES ABOVE")
    log.info("=" * 68)
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description="Biomarker Gran Prix compute backend probe / smoke")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--n-subjects", type=int, default=None,
                   help="Subject count N; enables block_size auto-sizing in the probe")
    p.add_argument("--max-cpus", type=int, default=None)
    p.add_argument("--max-ram-gb", type=float, default=None)
    p.add_argument("--max-gpu-mem-gb", type=float, default=None)
    p.add_argument("--block-size", default="auto", help="'auto' or an integer")
    p.add_argument("--no-smoke", action="store_true",
                   help="Only print the resolved config for the given flags (skip the smoke checks)")
    args = p.parse_args()

    setup_logging()

    if args.no_smoke or args.n_subjects is not None:
        cfg = resolve_compute(
            args.n_subjects,
            device=args.device,
            max_cpus=args.max_cpus,
            max_ram_gb=args.max_ram_gb,
            max_gpu_mem_gb=args.max_gpu_mem_gb,
            block_size=args.block_size,
        )
        log.info(describe_compute(cfg))
        if args.no_smoke:
            return

    passed = smoke()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
