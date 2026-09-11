"""Question-fixed-effects regression with a spline in length, for the style module.

Why this rather than a simple paired contrast:

A binary "has this transform" dummy is nearly collinear with the transform's own token delta,
because within a transform the length change barely varies (emoji always adds ~3 tokens, padding
always adds ~150). So a single-level design identifies the style effect almost entirely off the
assumed functional form of the length term, which is exactly the assumption under test. Dose-
response ladders break that collinearity by creating within-transform length variation.

The length term is a restricted cubic spline, not linear. The response is known to be nonlinear
here: diminishing returns at long lengths, floor effects on very short answers, and a truncation
cliff at 512 tokens. Fitting it linearly would push that curvature into the transform coefficients.

Inference is by wild cluster bootstrap over questions. Cluster-robust standard errors are reported
too, but with a few dozen clusters they are known to be downward-biased, and the wild bootstrap is
the standard remedy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------------------
# restricted cubic spline
# --------------------------------------------------------------------------------------


def rcs_knots(x, n_knots: int = 4) -> np.ndarray:
    """Harrell's recommended knot quantiles, deduplicated."""
    x = np.asarray(x, dtype=float)
    table = {
        3: [0.10, 0.50, 0.90],
        4: [0.05, 0.35, 0.65, 0.95],
        5: [0.05, 0.275, 0.50, 0.725, 0.95],
        6: [0.05, 0.23, 0.41, 0.59, 0.77, 0.95],
    }
    qs = table.get(n_knots, list(np.linspace(0.05, 0.95, n_knots)))
    k = np.unique(np.quantile(x, qs))
    if k.size < 3:  # too little spread for a spline; caller falls back to linear
        k = np.unique(np.quantile(x, [0.1, 0.5, 0.9]))
    return k


def rcs_basis(x, knots) -> np.ndarray:
    """Restricted cubic spline basis: columns [x, s_1, ..., s_{k-2}].

    Linear beyond the outer knots, which keeps extrapolation sane at the ends of the length range.
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(knots, dtype=float)
    k = t.size
    if k < 3:
        return x.reshape(-1, 1)
    scale = (t[-1] - t[0]) ** 2
    cols = [x]
    cube = lambda z: np.clip(z, 0, None) ** 3
    for j in range(k - 2):
        term = (
            cube(x - t[j])
            - cube(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
            + cube(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])
        )
        cols.append(term / scale)
    return np.column_stack(cols)


# --------------------------------------------------------------------------------------
# fixed-effects OLS with cluster-robust inference
# --------------------------------------------------------------------------------------


def _demean_by_cluster(M: np.ndarray, codes: np.ndarray, G: int) -> np.ndarray:
    """Absorb cluster fixed effects by within-transformation."""
    M = np.atleast_2d(M.T).T.astype(float)
    out = M.copy()
    counts = np.bincount(codes, minlength=G).astype(float)
    for j in range(M.shape[1]):
        sums = np.bincount(codes, weights=M[:, j], minlength=G)
        out[:, j] -= (sums / np.maximum(counts, 1))[codes]
    return out


def _cluster_meat(X: np.ndarray, u: np.ndarray, codes: np.ndarray, G: int,
                  XtX_inv: np.ndarray, cr2: bool) -> np.ndarray:
    p = X.shape[1]
    meat = np.zeros((p, p))
    for g in range(G):
        rows = np.flatnonzero(codes == g)
        if rows.size == 0:
            continue
        Xg, ug = X[rows], u[rows]
        if cr2:
            H = Xg @ XtX_inv @ Xg.T
            I = np.eye(rows.size)
            # Bell-McCaffrey adjustment A_g = (I - H_gg)^{-1/2}, via symmetric eigendecomposition.
            w, V = np.linalg.eigh(I - H)
            # Absorbing cluster fixed effects at the clustering level leaves exact zero
            # eigenvalues; flooring them keeps the correction finite instead of exploding.
            w = np.clip(w, 1e-8, None)
            A = V @ np.diag(w**-0.5) @ V.T
            ug = A @ ug
        s = Xg.T @ ug
        meat += np.outer(s, s)
    return XtX_inv @ meat @ XtX_inv


@dataclass
class FEFit:
    names: list[str]
    coef: np.ndarray
    se: np.ndarray
    se_cr0: np.ndarray
    p_wild: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    n_obs: int
    n_clusters: int
    knots: np.ndarray
    length_cols: list[int]
    r2_within: float
    extra: dict = field(default_factory=dict)

    def get(self, name: str) -> dict:
        i = self.names.index(name)
        return {
            "term": name,
            "coef": float(self.coef[i]),
            "se": float(self.se[i]),
            "p": float(self.p_wild[i]),
            "ci_low": float(self.ci_low[i]),
            "ci_high": float(self.ci_high[i]),
        }

    def as_table(self) -> list[dict]:
        return [self.get(n) for n in self.names]


def fit_fe(
    y,
    X,
    clusters,
    names: list[str],
    *,
    length_cols: list[int] | None = None,
    knots: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> FEFit:
    """OLS with absorbed cluster fixed effects, CR2 standard errors, wild cluster bootstrap p/CI."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    labels = np.asarray(clusters)
    uniq, codes = np.unique(labels, return_inverse=True)
    G = uniq.size

    yd = _demean_by_cluster(y.reshape(-1, 1), codes, G).ravel()
    Xd = _demean_by_cluster(X, codes, G)

    XtX = Xd.T @ Xd
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (Xd.T @ yd)
    u = yd - Xd @ beta

    V_cr2 = _cluster_meat(Xd, u, codes, G, XtX_inv, cr2=True)
    V_cr0 = _cluster_meat(Xd, u, codes, G, XtX_inv, cr2=False)
    se = np.sqrt(np.clip(np.diag(V_cr2), 0, None))
    se0 = np.sqrt(np.clip(np.diag(V_cr0), 0, None))

    # Wild cluster bootstrap, restricted (null imposed) per coefficient.
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_boot, G))
    p = X.shape[1]
    p_wild = np.ones(p)
    ci_lo = np.empty(p)
    ci_hi = np.empty(p)
    for j in range(p):
        keep = [i for i in range(p) if i != j]
        if keep:
            Xr = Xd[:, keep]
            br = np.linalg.pinv(Xr.T @ Xr) @ (Xr.T @ yd)
            ur = yd - Xr @ br
            fitted_null = Xr @ br
        else:
            ur = yd - yd.mean()
            fitted_null = np.full_like(yd, yd.mean())
        t_obs = beta[j] / se[j] if se[j] > 0 else 0.0
        t_null = np.empty(n_boot)
        boot_beta = np.empty(n_boot)
        for b in range(n_boot):
            yb = fitted_null + ur * signs[b][codes]
            bb = XtX_inv @ (Xd.T @ yb)
            ub = yb - Xd @ bb
            Vb = _cluster_meat(Xd, ub, codes, G, XtX_inv, cr2=False)
            sb = np.sqrt(max(Vb[j, j], 0.0))
            t_null[b] = bb[j] / sb if sb > 0 else 0.0
            boot_beta[b] = bb[j]
        p_wild[j] = (1 + np.sum(np.abs(t_null) >= abs(t_obs))) / (n_boot + 1)
        # Percentile interval from the unrestricted spread of the bootstrap coefficient.
        spread = boot_beta - boot_beta.mean()
        ci_lo[j] = beta[j] + np.percentile(spread, 100 * alpha / 2)
        ci_hi[j] = beta[j] + np.percentile(spread, 100 * (1 - alpha / 2))

    ss_tot = float(((yd - yd.mean()) ** 2).sum())
    r2 = 1.0 - float((u**2).sum()) / ss_tot if ss_tot > 0 else 0.0

    return FEFit(
        names=list(names),
        coef=beta,
        se=se,
        se_cr0=se0,
        p_wild=p_wild,
        ci_low=ci_lo,
        ci_high=ci_hi,
        n_obs=y.size,
        n_clusters=G,
        knots=np.asarray(knots) if knots is not None else np.array([]),
        length_cols=length_cols or [],
        r2_within=r2,
    )


def length_curve(fit: FEFit, grid, knots) -> dict:
    """Predicted reward-vs-length curve and its local slope.

    A single global "reward per 100 tokens" is a lossy summary of this curve: it is a weighted
    average of local slopes whose weights are set by the length distribution we chose. The curve
    and the marked local slopes are the honest artifacts.
    """
    grid = np.asarray(grid, dtype=float)
    B = rcs_basis(grid, knots)
    coefs = fit.coef[fit.length_cols]
    if B.shape[1] != coefs.size:
        raise ValueError(f"basis has {B.shape[1]} columns but {coefs.size} length coefficients")
    f = B @ coefs
    f = f - f[0]
    h = max(1.0, float(np.ptp(grid)) / 200.0)
    Bp = rcs_basis(grid + h, knots)
    slope = ((Bp @ coefs) - (B @ coefs)) / h
    return {
        "tokens": grid.tolist(),
        "reward": f.tolist(),
        "slope_per_token": slope.tolist(),
        "slope_per_100_tokens": (slope * 100).tolist(),
    }
