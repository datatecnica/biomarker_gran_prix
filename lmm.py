#!/usr/bin/env python3
"""lmm.py — random-intercept + random-slope LMM null fit for GALLOP (PLAN §5).

GALLOP (Sikorska et al. 2018) tests, for a continuous longitudinal outcome, both the
SNP main effect and the SNP×time effect under a mixed model with a per-subject random
intercept and random slope. The null model (covariates + time, no SNP) is fit **once**;
its variance components then let every SNP's [β_SNP, β_SNP×TIME] be recovered by a
batched closed-form 2×2 GLS (`gpu.scan_gallop`).

Generic LMM fitters (e.g. statsmodels MixedLM) are numerically fragile for the
random-slope model — they routinely fail to converge on GWAS-scale designs. So the
null fit here is a purpose-built **EM** for the two-random-effect model: stable,
grouped-and-batched by visit count, and it returns exactly the per-subject
projections GALLOP's scan needs.

    fit_ri_slope_lmm(...)  -> LMMNull(alpha, D, sigma2, converged, ...)
    gallop_precompute(...) -> per-subject P_i, q_i, M_i (2×·) and A⁻¹ for the scan
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LMMNull:
    alpha: np.ndarray     # fixed-effect estimates (p,) for X = [1, covariates, time]
    D: np.ndarray         # (2,2) random-effect covariance [intercept, slope]
    sigma2: float         # residual variance
    resid: np.ndarray     # r = y − Xα  (n_obs,)
    converged: bool
    n_iter: int


def _group_by_visitcount(group_idx: np.ndarray, n_groups: int):
    """Return, per distinct visit-count, the arrays needed to batch the EM.

    Yields (rows, subj_ids) where `rows` is the (m, n_i) index matrix into the long
    arrays for the m subjects that have exactly n_i visits, and `subj_ids` their ids.
    """
    counts = np.bincount(group_idx, minlength=n_groups)
    # positions of each subject's rows (assumes rows for a subject are contiguous OR not —
    # we gather explicitly)
    order = np.argsort(group_idx, kind="stable")
    sorted_g = group_idx[order]
    starts = np.searchsorted(sorted_g, np.arange(n_groups))
    for nc in np.unique(counts[counts > 0]):
        subj = np.where(counts == nc)[0]
        rows = np.empty((len(subj), nc), dtype=np.int64)
        for k, s in enumerate(subj):
            rows[k] = order[starts[s]:starts[s] + nc]
        yield int(nc), rows, subj


def fit_ri_slope_lmm(y, X, tvec, group_idx, n_groups, max_iter=500, tol=1e-5) -> LMMNull:
    """EM fit of y = Xα + Z b + ε, Z_i = [1, t_i], b_i ~ N(0, D), ε ~ N(0, σ²I)."""
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    tvec = np.asarray(tvec, dtype=np.float64)
    n_obs, p = X.shape
    groups = list(_group_by_visitcount(group_idx, n_groups))

    # --- initialize: fixed effects by OLS, variance components from per-subject OLS
    # intercept/slope moments. A data-driven D avoids EM getting stuck at a near-zero
    # random-slope variance (the boundary), which collapses the slope component.
    alpha, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid0 = y - X @ alpha
    coefs, within_var = [], []
    for nc, rows, subj in groups:
        Zg = np.stack([np.ones((len(subj), nc)), tvec[rows]], axis=-1)   # (m, nc, 2)
        rg = resid0[rows]
        ZtZ = np.einsum("mna,mnb->mab", Zg, Zg)
        Ztr = np.einsum("mna,mn->ma", Zg, rg)
        c = np.linalg.solve(ZtZ + 1e-6 * np.eye(2), Ztr[..., None])[..., 0]   # (m, 2)
        coefs.append(c)
        if nc > 2:
            fit = np.einsum("mna,ma->mn", Zg, c)
            within_var.append(((rg - fit) ** 2).sum(1) / (nc - 2))
    C = np.vstack(coefs)
    D = np.cov(C.T)
    D = np.atleast_2d(D) + 1e-4 * np.eye(2)
    sigma2 = float(np.concatenate(within_var).mean()) if within_var else max(resid0.var() * 0.5, 1e-4)

    converged = False
    for it in range(max_iter):
        XtViX = np.zeros((p, p))
        XtViy = np.zeros(p)
        # cache per-group Vinv pieces for the E-step reuse
        cache = []
        for nc, rows, subj in groups:
            Xg = X[rows]                       # (m, nc, p)
            yg = y[rows]                       # (m, nc)
            tg = tvec[rows]                    # (m, nc)
            Zg = np.stack([np.ones_like(tg), tg], axis=-1)   # (m, nc, 2)
            ZDZ = np.einsum("mna,ab,mkb->mnk", Zg, D, Zg)    # (m, nc, nc)
            Vg = ZDZ + sigma2 * np.eye(nc)[None]
            Vinv = np.linalg.inv(Vg)                          # (m, nc, nc)
            XtVi = np.einsum("mnp,mnk->mpk", Xg, Vinv)        # (m, p, nc)
            XtViX += np.einsum("mpk,mkq->pq", XtVi, Xg)
            XtViy += np.einsum("mpk,mk->p", XtVi, yg)
            cache.append((nc, rows, Zg, yg, Xg, Vinv))
        alpha = np.linalg.solve(XtViX, XtViy)

        # --- E-step + M-step (D, sigma2) ---
        D_new = np.zeros((2, 2))
        s2_num = 0.0
        for nc, rows, Zg, yg, Xg, Vinv in cache:
            rg = yg - np.einsum("mnp,p->mn", Xg, alpha)       # (m, nc)
            ZtVi = np.einsum("mna,mnk->mak", Zg, Vinv)        # (m, 2, nc)
            b = np.einsum("ab,mbk,mk->ma", D, ZtVi, rg)       # BLUP (m, 2)
            ZtViZ = np.einsum("mak,mkb->mab", ZtVi, Zg)       # (m, 2, 2)
            S = D[None] - np.einsum("ab,mbc,cd->mad", D, ZtViZ, D)   # posterior cov (m,2,2)
            D_new += np.einsum("ma,mb->ab", b, b) + S.sum(0)
            ehat = rg - np.einsum("mna,ma->mn", Zg, b)        # (m, nc)
            ZtZ = np.einsum("mna,mnb->mab", Zg, Zg)           # (m, 2, 2)
            s2_num += float((ehat * ehat).sum() + np.einsum("mab,mab->", S, ZtZ))
        D_new /= n_groups
        sigma2_new = s2_num / n_obs

        delta = max(np.abs(D_new - D).max(), abs(sigma2_new - sigma2))
        D, sigma2 = D_new, max(sigma2_new, 1e-8)
        if delta < tol:
            converged = True
            break

    resid = y - X @ alpha
    return LMMNull(alpha=alpha, D=D, sigma2=float(sigma2), resid=resid,
                   converged=converged, n_iter=it + 1)


def gallop_precompute(null: LMMNull, X, tvec, group_idx, n_groups):
    """Per-subject GALLOP projections from the fitted null model.

    Returns (P, q, M, Ainv):
      P   (N,2,2)  Z_iᵀ V_i⁻¹ Z_i
      q   (N,2)    Z_iᵀ V_i⁻¹ r_i         (r_i = null residuals)
      M   (N,2,p)  Z_iᵀ V_i⁻¹ X_i         (covariate cross term)
      Ainv (p,p)   (Σ_i X_iᵀ V_i⁻¹ X_i)⁻¹
    The SNP design for subject i is G_i·Z_i, so C_s = Σ G_i² P_i, B_s = Σ G_i M_i,
    rhs_s = Σ G_i q_i, and [β_SNP, β_SNPxTIME] = (C_s − B_s Ainv B_sᵀ)⁻¹ rhs_s.
    """
    X = np.asarray(X, dtype=np.float64)
    tvec = np.asarray(tvec, dtype=np.float64)
    r = null.resid
    n, p = X.shape
    D, sigma2 = null.D, null.sigma2
    P = np.zeros((n_groups, 2, 2))
    q = np.zeros((n_groups, 2))
    M = np.zeros((n_groups, 2, p))
    A = np.zeros((p, p))
    for nc, rows, subj in _group_by_visitcount(group_idx, n_groups):
        Xg = X[rows]
        tg = tvec[rows]
        rg = r[rows]
        Zg = np.stack([np.ones_like(tg), tg], axis=-1)
        Vg = np.einsum("mna,ab,mkb->mnk", Zg, D, Zg) + sigma2 * np.eye(nc)[None]
        Vinv = np.linalg.inv(Vg)
        ZtVi = np.einsum("mna,mnk->mak", Zg, Vinv)            # (m,2,nc)
        P[subj] = np.einsum("mak,mkb->mab", ZtVi, Zg)
        q[subj] = np.einsum("mak,mk->ma", ZtVi, rg)
        M[subj] = np.einsum("mak,mkp->map", ZtVi, Xg)
        XtVi = np.einsum("mnp,mnk->mpk", Xg, Vinv)
        A += np.einsum("mpk,mkq->pq", XtVi, Xg)
    Ainv = np.linalg.inv(A)
    return P, q, M, Ainv


def conditional_residuals(null: LMMNull, X, tvec, y, group_idx, n_groups):
    """Per-observation conditional residuals ê_ij = y_ij − x_ijα − z_ij b̂_i.

    Removes both the fixed effects and the per-subject BLUP random effects, so ê
    estimates the within-subject noise — the input to the dispersion (variance) model
    that TrajGWAS's variability (τ) test is built on (PLAN §5).
    """
    X = np.asarray(X, dtype=np.float64)
    tvec = np.asarray(tvec, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r = y - X @ null.alpha
    D, sigma2 = null.D, null.sigma2
    ehat = np.empty_like(y)
    for nc, rows, subj in _group_by_visitcount(group_idx, n_groups):
        tg = tvec[rows]
        rg = r[rows]
        Zg = np.stack([np.ones_like(tg), tg], axis=-1)               # (m, nc, 2)
        Vg = np.einsum("mna,ab,mkb->mnk", Zg, D, Zg) + sigma2 * np.eye(nc)[None]
        Vinv = np.linalg.inv(Vg)
        b = np.einsum("ab,mkb,mnk,mn->ma", D, Zg, Vinv, rg)          # BLUP (m, 2)
        ehat[rows] = rg - np.einsum("mna,ma->mn", Zg, b)
    return ehat
