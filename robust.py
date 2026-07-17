#!/usr/bin/env python3
"""robust.py — CPU tail refinements for Biomarker Gran Prix (PLAN §5, §7).

The batched score tests (`gpu.scan_logistic`, and the martingale-residual linear
scan used for `cox`) give fast, calibrated P-values in the bulk. The tail needs
care, and only the tail — so these run on CPU, per-variant:

  * **SPA** (saddlepoint approximation, Lugannani–Rice) — accurate P-values where the
    normal approximation of the score is poor: rare variants and case/control or
    event imbalance. `spa_logistic` uses the exact Bernoulli CGF of the null score;
    `spa_cox` (= SPACox) uses the empirical CGF of the null martingale residuals.
  * **Firth** penalized logistic — exact, separation-robust β/SE, refit on hits.
  * **lifelines 3-tier Cox refit** (unpenalized → ridge 0.01 → ridge 0.1) — exact HR
    on hits / score-test failures.

Everything here is called only on the small significant/rare subset (PLAN §7).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

# ============================================================================
# Covariate-adjusted genotype
# ============================================================================


def residualize_weighted(g: np.ndarray, Q: np.ndarray, w: np.ndarray) -> np.ndarray:
    """g̃ = g − Q (QᵀWQ)⁻¹ QᵀW g — genotype orthogonalized to Q in the W metric."""
    WQ = Q * w[:, None]
    coef = np.linalg.solve(WQ.T @ Q, WQ.T @ g)
    return g - Q @ coef


# ============================================================================
# Saddlepoint approximation (Lugannani–Rice)
# ============================================================================


def _lugannani_rice(t_hat: float, s: float, K: float, Kpp: float) -> float:
    """Two-sided tail probability from a solved saddlepoint (mean-0 statistic)."""
    if abs(t_hat) < 1e-7 or Kpp <= 0:
        return float(np.nan)
    w = np.sign(t_hat) * np.sqrt(max(2.0 * (t_hat * s - K), 0.0))
    v = t_hat * np.sqrt(Kpp)
    if abs(w) < 1e-7:
        return float(np.nan)
    one_side = norm.sf(w) + norm.pdf(w) * (1.0 / w - 1.0 / v)
    one_side = min(max(one_side, 0.0), 1.0)
    return float(min(2.0 * min(one_side, 1.0 - one_side), 1.0))


def _solve_saddle(Kp, Kpp, s: float, lo: float = -100.0, hi: float = 100.0) -> float | None:
    """Solve K'(t) = s for t by bracketed Newton/bisection (K' is increasing)."""
    if Kp(0.0) == s:
        return 0.0
    # bracket
    a, b = lo, hi
    fa, fb = Kp(a) - s, Kp(b) - s
    if fa > 0 or fb < 0:
        return None
    t = 0.0
    for _ in range(60):
        ft = Kp(t) - s
        if abs(ft) < 1e-8:
            return t
        if ft > 0:
            b = t
        else:
            a = t
        d = Kpp(t)
        step = ft / d if d > 1e-12 else 0.0
        nt = t - step
        if not (a < nt < b):
            nt = 0.5 * (a + b)
        t = nt
    return t


def spa_logistic(g: np.ndarray, p_hat: np.ndarray, Q: np.ndarray, s_obs: float,
                 w: np.ndarray | None = None) -> float:
    """SPA two-sided P for a logistic score S = g̃ᵀ(y − p̂) (SAIGE-style).

    Uses the exact CGF of S under H0 (y_i ~ Bernoulli(p̂_i)):
        K(t)  = Σ log(1 − p̂ + p̂ e^{g̃ t}) − t Σ g̃ p̂
    """
    if w is None:
        w = p_hat * (1 - p_hat)
    gt = residualize_weighted(g, Q, w)
    gp = gt * p_hat

    def K(t):
        return np.sum(np.log1p(p_hat * np.expm1(gt * t))) - t * gp.sum()

    def Kp(t):
        e = np.exp(gt * t)
        num = p_hat * e
        return np.sum(gt * num / (1 - p_hat + num)) - gp.sum()

    def Kpp(t):
        e = np.exp(gt * t)
        denom = 1 - p_hat + p_hat * e
        num = p_hat * e
        return np.sum(gt * gt * num * (1 - p_hat) / (denom * denom))

    t_hat = _solve_saddle(Kp, Kpp, s_obs)
    if t_hat is None:
        return float(np.nan)
    return _lugannani_rice(t_hat, s_obs, K(t_hat), Kpp(t_hat))


class MartingaleCGF:
    """Empirical CGF of the null martingale residuals, precomputed once per Cox run.

    SPACox's per-term CGF is K_R(s) = log mean_j e^{s M_j} — a smooth 1-D function of a
    scalar. We tabulate K_R, K_R', K_R'' on a fixed grid over s (log-sum-exp stabilized),
    so every subsequent SPA evaluation is O(n) interpolation with no exp in the loop —
    turning per-variant SPACox from O(n·grid) into O(n), which scales to genome-wide.
    """

    def __init__(self, M: np.ndarray, s_max: float = 30.0, n_grid: int = 4001):
        M = np.asarray(M, dtype=np.float64)
        self.m = len(M)
        s = np.linspace(-s_max, s_max, n_grid)
        A = s[:, None] * M[None, :]                            # (n_grid, m)
        amax = A.max(1, keepdims=True)
        e = np.exp(A - amax)
        se = e.sum(1)
        w = e / se[:, None]
        mean = w @ M
        self.s = s
        self.K = amax[:, 0] + np.log(se) - np.log(self.m)      # K_R(s)
        self.Kp = mean                                         # K_R'(s)  (∈ [min M, max M])
        self.Kpp = np.clip(w @ (M * M) - mean * mean, 0.0, None)  # K_R''(s)

    def eval(self, svals: np.ndarray):
        return (np.interp(svals, self.s, self.K),
                np.interp(svals, self.s, self.Kp),
                np.interp(svals, self.s, self.Kpp))


def spa_cox(g_tilde: np.ndarray, s_obs: float, cgf: MartingaleCGF) -> float:
    """SPACox two-sided P for a Cox score S = g̃ᵀM (Bi et al. 2020), using a precomputed
    empirical CGF of the martingale residuals. K(t) = Σ_i K_R(t g̃_i)."""
    def _stats(t):
        Kr, Krp, Krpp = cgf.eval(t * g_tilde)
        return float(Kr.sum()), float(np.sum(g_tilde * Krp)), float(np.sum(g_tilde * g_tilde * Krpp))

    t_hat = _solve_saddle(lambda t: _stats(t)[1], lambda t: _stats(t)[2], s_obs)
    if t_hat is None:
        return float(np.nan)
    K, _, Kpp = _stats(t_hat)
    return _lugannani_rice(t_hat, s_obs, K, Kpp)


# ============================================================================
# Firth penalized logistic (exact β/SE on hits / separation)
# ============================================================================


def firth_logistic(X: np.ndarray, y: np.ndarray, max_iter: int = 50,
                   tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray, bool]:
    """Firth-penalized logistic regression (Jeffreys-prior bias reduction).

    Returns (beta, se, converged). The penalized score adds h_i(½ − p_i) where h_i
    are the hat-matrix diagonals — stable under separation, unlike plain MLE.
    """
    n, k = X.shape
    beta = np.zeros(k)
    converged = False
    for _ in range(max_iter):
        eta = X @ beta
        p = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-10, 1 - 1e-10)
        w = p * (1 - p)
        XtWX = (X * w[:, None]).T @ X
        try:
            XtWX_inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            break
        # hat diagonals h_i = w_i x_iᵀ (XᵀWX)⁻¹ x_i
        XW = X * np.sqrt(w)[:, None]
        h = np.einsum("ij,jk,ik->i", XW, XtWX_inv, XW)
        U = X.T @ (y - p + h * (0.5 - p))          # penalized score
        step = XtWX_inv @ U
        beta_new = beta + step
        if np.max(np.abs(step)) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new
    eta = X @ beta
    p = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-10, 1 - 1e-10)
    w = p * (1 - p)
    try:
        se = np.sqrt(np.diag(np.linalg.inv((X * w[:, None]).T @ X)))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    return beta, se, converged


# ============================================================================
# Cox exact refit (lifelines 3-tier: unpenalized → ridge 0.01 → ridge 0.1)
# ============================================================================


def cox_refit_tiers(df, duration_col: str, event_col: str, snp_col: str = "g"):
    """Exact Cox HR for one SNP via lifelines, escalating ridge on failure.

    Returns (beta, se, P, tier) or (nan, nan, nan, 'failed'). `df` holds the SNP
    column, covariates, duration, and event.
    """
    try:
        from lifelines import CoxPHFitter
    except ImportError:  # pragma: no cover
        return np.nan, np.nan, np.nan, "no_lifelines"
    for penalizer, tier in ((0.0, "cox"), (0.01, "cox_ridge0.01"), (0.1, "cox_ridge0.1")):
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(df, duration_col=duration_col, event_col=event_col)
            return (float(cph.params_[snp_col]), float(cph.standard_errors_[snp_col]),
                    float(cph.summary.loc[snp_col, "p"]), tier)
        except Exception:
            continue
    return np.nan, np.nan, np.nan, "failed"
