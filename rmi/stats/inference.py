"""Inference for paired reward-model contrasts.

Three facts about this setting drive every choice here.

1. **The reward model is deterministic.** Scoring the same pair twice gives a bit-identical result
   (measured: 0.0 difference). So there is no measurement noise at all, and every confidence
   interval is a statement about heterogeneity across the items *we wrote*, not about sampling a
   population. That makes the clustering structure the entire inferential content.

2. **Items are reused across contrasts.** The same 30-100 questions appear in every style
   transform, and the same templates in every identity swap. Resampling rows would ignore that
   dependence and produce intervals that are far too narrow, so everything here resamples
   *clusters*, carrying all of a cluster's rows together.

3. **Contrasts are heavily positively correlated** for the same reason. Benjamini-Hochberg assumes
   independence or PRDS, which is not guaranteed here, so the default multiplicity correction is
   Westfall-Young step-down max-T: valid under arbitrary dependence, more powerful than BH under
   the positive correlation we actually have, and it subsumes the "we picked the biggest gap"
   selection problem with the same mechanism.

Deliberately *not* provided as a headline: Cohen's d_z. With a deterministic scorer, sd(d) contains
no measurement noise, so d_z measures how *consistent* an effect is across our hand-written items
rather than how *big* it is, and it diverges to infinity for a perfectly uniform effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import stats as sps


# --------------------------------------------------------------------------------------
# cluster machinery
# --------------------------------------------------------------------------------------


def _cluster_index(clusters) -> tuple[np.ndarray, np.ndarray]:
    """Map arbitrary cluster labels to 0..G-1. Returns (codes, unique_labels)."""
    labels = np.asarray(clusters)
    uniq, codes = np.unique(labels, return_inverse=True)
    return codes, uniq


def _cluster_sums(values: np.ndarray, codes: np.ndarray, n_clusters: int):
    sums = np.bincount(codes, weights=values, minlength=n_clusters)
    sizes = np.bincount(codes, minlength=n_clusters).astype(float)
    return sums, sizes


def cluster_se_of_mean(values: np.ndarray, codes: np.ndarray, n_clusters: int) -> float:
    """CR0 cluster-robust standard error of the mean.

    Treats each cluster as the independent unit. With G clusters this is
    sqrt( sum_g (S_g - n_g * xbar)^2 ) / n, i.e. the sandwich form for a simple mean.
    """
    n = values.size
    if n == 0:
        return float("nan")
    sums, sizes = _cluster_sums(values, codes, n_clusters)
    xbar = values.sum() / n
    resid = sums - sizes * xbar
    return float(np.sqrt((resid**2).sum()) / n)


# --------------------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------------------


def cluster_bootstrap(
    values,
    clusters,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    statistic=np.mean,
) -> np.ndarray:
    """Resample whole clusters with replacement, carrying every row of a selected cluster."""
    values = np.asarray(values, dtype=float)
    codes, uniq = _cluster_index(clusters)
    G = uniq.size
    rows_by_cluster = [np.flatnonzero(codes == g) for g in range(G)]
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, G, size=(n_boot, G))
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rows_by_cluster[g] for g in picks[b]])
        out[b] = statistic(values[idx])
    return out


def _bca_interval(
    boot: np.ndarray, observed: float, jackknife: np.ndarray, alpha: float
) -> tuple[float, float]:
    """Bias-corrected and accelerated interval. Falls back to percentile if degenerate."""
    lo_p, hi_p = 100 * alpha / 2, 100 * (1 - alpha / 2)
    prop = np.mean(boot < observed)
    if prop <= 0 or prop >= 1:
        return float(np.percentile(boot, lo_p)), float(np.percentile(boot, hi_p))
    z0 = sps.norm.ppf(prop)
    jbar = jackknife.mean()
    diff = jbar - jackknife
    denom = 6.0 * (np.sum(diff**2) ** 1.5)
    a = float(np.sum(diff**3) / denom) if denom > 0 else 0.0
    za = sps.norm.ppf(alpha / 2)
    zb = sps.norm.ppf(1 - alpha / 2)

    def adj(z):
        d = 1 - a * (z0 + z)
        if abs(d) < 1e-12:
            return np.nan
        return sps.norm.cdf(z0 + (z0 + z) / d)

    a1, a2 = adj(za), adj(zb)
    if not np.isfinite(a1) or not np.isfinite(a2):
        return float(np.percentile(boot, lo_p)), float(np.percentile(boot, hi_p))
    return float(np.percentile(boot, 100 * a1)), float(np.percentile(boot, 100 * a2))


def cluster_bootstrap_ci(
    values,
    clusters,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    statistic=np.mean,
    method: str = "bca",
) -> tuple[float, float]:
    """BCa cluster bootstrap CI. Jackknife for the acceleration leaves out whole clusters."""
    values = np.asarray(values, dtype=float)
    codes, uniq = _cluster_index(clusters)
    G = uniq.size
    observed = float(statistic(values))
    boot = cluster_bootstrap(
        values, clusters, n_boot=n_boot, seed=seed, statistic=statistic
    )
    if np.allclose(boot, boot[0]):  # degenerate, e.g. a single cluster
        return observed, observed
    if method == "percentile" or G < 3:
        return (
            float(np.percentile(boot, 100 * alpha / 2)),
            float(np.percentile(boot, 100 * (1 - alpha / 2))),
        )
    jack = np.array([float(statistic(values[codes != g])) for g in range(G)])
    return _bca_interval(boot, observed, jack, alpha)


def wild_cluster_bootstrap_p(
    values,
    clusters,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    two_sided: bool = True,
) -> float:
    """Wild cluster bootstrap p-value for H0: mean = 0, with Rademacher weights.

    The standard remedy when the number of clusters is small (identity has ~12-40 templates),
    where cluster-robust standard errors are severely downward-biased.
    """
    values = np.asarray(values, dtype=float)
    codes, uniq = _cluster_index(clusters)
    G = uniq.size
    t_obs = values.mean() / (cluster_se_of_mean(values, codes, G) or np.inf)
    rng = np.random.default_rng(seed)
    w = rng.choice([-1.0, 1.0], size=(n_boot, G))
    # Impose the null by recentring, then reweight each cluster's residuals.
    resid = values - values.mean()
    stats_null = np.empty(n_boot)
    for b in range(n_boot):
        y = resid * w[b][codes]
        m = y.mean()
        s = cluster_se_of_mean(y, codes, G)
        stats_null[b] = m / s if s > 0 else 0.0
    if two_sided:
        return float((1 + np.sum(np.abs(stats_null) >= abs(t_obs))) / (n_boot + 1))
    return float((1 + np.sum(stats_null >= t_obs)) / (n_boot + 1))


# --------------------------------------------------------------------------------------
# paired contrast summary
# --------------------------------------------------------------------------------------


@dataclass
class ContrastResult:
    name: str
    family: str
    n_items: int
    n_clusters: int
    mean_delta: float
    sd_delta: float
    ci_low: float
    ci_high: float
    win_rate: float
    win_ci_low: float
    win_ci_high: float
    p_sign: float
    p_wilcoxon: float
    cluster_se: float
    t_stat: float
    two_sided: bool
    n_truncated: int = 0
    p_adjusted: float | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def paired_contrast(
    deltas,
    clusters,
    *,
    name: str,
    family: str = "default",
    two_sided: bool = True,
    n_boot: int = 10_000,
    seed: int = 0,
    n_truncated: int = 0,
) -> ContrastResult:
    """Summarise one paired contrast.

    The sign test is primary for *direction* and the mean for *magnitude*. Wilcoxon is reported
    but not headlined: it assumes the difference distribution is symmetric, and these deltas are
    routinely bimodal because an effect fires on some items and not others, in which case Wilcoxon
    tests a pseudo-median that agrees with neither the mean nor the sign test.
    """
    d = np.asarray(deltas, dtype=float)
    codes, uniq = _cluster_index(clusters)
    G = uniq.size
    n = d.size
    se = cluster_se_of_mean(d, codes, G)
    lo, hi = cluster_bootstrap_ci(d, clusters, n_boot=n_boot, seed=seed)

    nz = d[d != 0]
    n_pos = int(np.sum(nz > 0))
    win = n_pos / nz.size if nz.size else 0.5
    # Win-rate CI is a cluster bootstrap, not Clopper-Pearson: wins cluster by question, so a
    # binomial interval would be too narrow for the same reason row bootstrap is.
    wlo, whi = cluster_bootstrap_ci(
        (d > 0).astype(float), clusters, n_boot=n_boot, seed=seed + 1
    )
    if nz.size:
        alt = "two-sided" if two_sided else "greater"
        p_sign = float(sps.binomtest(n_pos, nz.size, 0.5, alternative=alt).pvalue)
    else:
        p_sign = 1.0
    if nz.size == 0:  # every difference is exactly zero
        p_w = 1.0
    else:
        try:
            p_w = float(
                sps.wilcoxon(d, alternative="two-sided" if two_sided else "greater").pvalue
            )
        except ValueError:
            p_w = 1.0

    return ContrastResult(
        name=name,
        family=family,
        n_items=n,
        n_clusters=G,
        mean_delta=float(d.mean()),
        sd_delta=float(d.std(ddof=1)) if n > 1 else 0.0,
        ci_low=lo,
        ci_high=hi,
        win_rate=float(win),
        win_ci_low=wlo,
        win_ci_high=whi,
        p_sign=p_sign,
        p_wilcoxon=p_w,
        cluster_se=se,
        t_stat=float(d.mean() / se) if se > 0 else 0.0,
        two_sided=two_sided,
        n_truncated=n_truncated,
    )


# --------------------------------------------------------------------------------------
# multiplicity
# --------------------------------------------------------------------------------------


def bh_fdr(pvalues) -> np.ndarray:
    """Benjamini-Hochberg q-values. Kept as a simple fallback; Westfall-Young is the default."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0, 1)
    return out


def westfall_young_adjust(observed_t, null_t) -> np.ndarray:
    """Step-down max-T adjusted p-values.

    ``observed_t`` is (m,) and ``null_t`` is (B, m) drawn under a permutation scheme that
    regenerates *all* contrasts together, which is what preserves their correlation. Valid under
    arbitrary dependence between contrasts.
    """
    t = np.abs(np.asarray(observed_t, dtype=float))
    N = np.abs(np.asarray(null_t, dtype=float))
    m = t.size
    B = N.shape[0]
    order = np.argsort(-t)  # most significant first
    t_ord = t[order]
    N_ord = N[:, order]
    # successive maxima from the least significant end inwards
    M = np.maximum.accumulate(N_ord[:, ::-1], axis=1)[:, ::-1]
    p_raw = (1.0 + (M >= t_ord[None, :]).sum(axis=0)) / (B + 1.0)
    p_step = np.maximum.accumulate(p_raw)  # enforce monotonicity
    out = np.empty(m)
    out[order] = np.clip(p_step, 0, 1)
    return out


def signflip_null_t(
    delta_list: list[np.ndarray],
    cluster_list: list[np.ndarray],
    *,
    n_perm: int = 10_000,
    seed: int = 0,
) -> np.ndarray:
    """Null t-statistics for a family of paired contrasts, via cluster-level sign flipping.

    Under H0 each item's delta is symmetric about zero, so flipping the sign of *all* of a
    cluster's deltas is exchangeable and preserves within-cluster dependence. The same sign draws
    are reused across every contrast in the family, which is what preserves the between-contrast
    correlation that Westfall-Young needs.

    Everything reduces to per-cluster sums, so this is vectorised over permutations.
    """
    # Align all contrasts onto one shared cluster universe so a sign draw means the same thing
    # in every contrast.
    all_labels = np.unique(np.concatenate([np.asarray(c) for c in cluster_list]))
    lab_to_idx = {lab: i for i, lab in enumerate(all_labels)}
    G = len(all_labels)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, G))

    out = np.empty((n_perm, len(delta_list)))
    for j, (d, c) in enumerate(zip(delta_list, cluster_list)):
        d = np.asarray(d, dtype=float)
        codes = np.array([lab_to_idx[x] for x in np.asarray(c)])
        n = d.size
        sums = np.bincount(codes, weights=d, minlength=G)
        sizes = np.bincount(codes, minlength=G).astype(float)
        present = sizes > 0
        # flipped mean = sum_g s_g * S_g / n
        means = signs[:, present] @ sums[present] / n
        # flipped cluster-robust SE of the mean
        flipped_sums = signs[:, present] * sums[present][None, :]
        resid = flipped_sums - sizes[present][None, :] * means[:, None]
        se = np.sqrt((resid**2).sum(axis=1)) / n
        out[:, j] = np.where(se > 0, means / np.where(se > 0, se, 1.0), 0.0)
    return out


def adjust_family(results: list[ContrastResult], *, n_perm: int = 10_000, seed: int = 0,
                  deltas: dict[str, np.ndarray] | None = None,
                  clusters: dict[str, np.ndarray] | None = None) -> list[ContrastResult]:
    """Attach Westfall-Young adjusted p-values to one family, in place.

    Falls back to BH on the sign-test p-values when the raw deltas are unavailable.
    """
    if not results:
        return results
    if deltas and clusters and all(r.name in deltas for r in results):
        dl = [np.asarray(deltas[r.name], float) for r in results]
        cl = [np.asarray(clusters[r.name]) for r in results]
        null_t = signflip_null_t(dl, cl, n_perm=n_perm, seed=seed)
        adj = westfall_young_adjust([r.t_stat for r in results], null_t)
    else:
        adj = bh_fdr([r.p_sign for r in results])
    for r, a in zip(results, adj):
        r.p_adjusted = float(a)
    return results


# --------------------------------------------------------------------------------------
# identity omnibus
# --------------------------------------------------------------------------------------


def max_group_gap_permutation(
    scores,
    names,
    groups,
    templates,
    *,
    strata=None,
    n_perm: int = 10_000,
    seed: int = 0,
) -> dict:
    """Omnibus test: does *any* identity group differ, allowing for having picked the biggest gap?

    The statistic is the largest absolute difference between **group means** (averaged over
    templates), and the permutation shuffles the name->group map.

    This must operate on group means. Computing a max gap over *individual names* would make the
    statistic max(s) - min(s) over a fixed multiset of scores, which is invariant under
    permutation, so the permutation distribution collapses to a point mass and the test returns
    p = 1.0 always: a silent null that looks like a clean result. ``tests/`` pins this.

    ``strata`` (default: name token count) restricts the shuffle so a name may only take the group
    label of a similarly-tokenised name. Without it a rejection is consistent with names simply
    differing in token length rather than in perceived identity.
    """
    scores = np.asarray(scores, dtype=float)
    names = np.asarray(names)
    groups = np.asarray(groups)
    templates = np.asarray(templates)

    uniq_names, name_codes = np.unique(names, return_inverse=True)
    uniq_tmpl, tmpl_codes = np.unique(templates, return_inverse=True)
    uniq_groups = np.unique(groups)
    n_groups = uniq_groups.size
    if n_groups < 2:
        raise ValueError("need at least two identity groups")

    # One group label per name (names belong to exactly one group).
    name_group = np.empty(uniq_names.size, dtype=object)
    for i, nm in enumerate(uniq_names):
        name_group[i] = groups[names == nm][0]
    group_to_idx = {g: i for i, g in enumerate(uniq_groups)}
    name_group_idx = np.array([group_to_idx[g] for g in name_group])

    cell_count = np.zeros(uniq_names.size)
    cell_sum = np.zeros(uniq_names.size)
    # Template-centre first so template difficulty cannot leak into group means when the design
    # is not perfectly balanced.
    centred = scores.copy()
    for t in range(uniq_tmpl.size):
        m = tmpl_codes == t
        centred[m] -= centred[m].mean()
    np.add.at(cell_sum, name_codes, centred)
    np.add.at(cell_count, name_codes, 1.0)
    name_mean = cell_sum / np.maximum(cell_count, 1)

    def stat_from_assignment(assign: np.ndarray) -> float:
        gm = np.array(
            [name_mean[assign == g].mean() if np.any(assign == g) else np.nan
             for g in range(n_groups)]
        )
        return float(np.nanmax(gm) - np.nanmin(gm))

    observed = stat_from_assignment(name_group_idx)

    if strata is None:
        strata_arr = np.zeros(uniq_names.size, dtype=int)
    else:
        s_map = {}
        for i, nm in enumerate(uniq_names):
            s_map[i] = np.asarray(strata)[np.flatnonzero(names == nm)[0]]
        strata_arr = np.array([s_map[i] for i in range(uniq_names.size)])

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    strata_blocks = [np.flatnonzero(strata_arr == s) for s in np.unique(strata_arr)]
    for b in range(n_perm):
        assign = name_group_idx.copy()
        for blk in strata_blocks:
            assign[blk] = rng.permutation(assign[blk])
        null[b] = stat_from_assignment(assign)

    p = float((1 + np.sum(null >= observed)) / (n_perm + 1))
    group_means = {
        str(uniq_groups[g]): float(name_mean[name_group_idx == g].mean())
        for g in range(n_groups)
    }
    return {
        "statistic": observed,
        "p_value": p,
        "n_perm": n_perm,
        "group_means": group_means,
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "n_names": int(uniq_names.size),
        "n_templates": int(uniq_tmpl.size),
        "stratified": strata is not None,
    }


def within_template_max_gap_permutation(
    scores, levels, templates, *, n_perm: int = 10_000, seed: int = 0
) -> dict:
    """Omnibus test for a factor whose every level is its own group (descriptor swaps).

    ``max_group_gap_permutation`` relabels names to groups, which needs several names per group.
    When each level appears exactly once per template - religion, age, nationality, disability -
    that relabelling is a pure permutation of a fixed vector of level means, so the max-minus-min
    statistic is invariant and the test would return p = 1 by construction.

    The right exchangeability unit here is instead *within template*: under the null that the
    descriptor does not matter, the assignment of scores to descriptors inside a single template is
    arbitrary. Permuting independently within each template and then averaging across templates
    does move the level means, so the test has power.
    """
    scores = np.asarray(scores, dtype=float)
    levels = np.asarray(levels)
    templates = np.asarray(templates)
    uniq_levels, level_codes = np.unique(levels, return_inverse=True)
    uniq_tmpl, tmpl_codes = np.unique(templates, return_inverse=True)
    L, T = uniq_levels.size, uniq_tmpl.size
    if L < 2:
        raise ValueError("need at least two levels")

    centred = scores.copy()
    for t in range(T):
        m = tmpl_codes == t
        centred[m] -= centred[m].mean()

    def stat(codes: np.ndarray) -> float:
        sums = np.bincount(codes, weights=centred, minlength=L)
        cnts = np.bincount(codes, minlength=L)
        means = sums / np.maximum(cnts, 1)
        return float(means.max() - means.min())

    observed = stat(level_codes)
    rng = np.random.default_rng(seed)
    blocks = [np.flatnonzero(tmpl_codes == t) for t in range(T)]
    null = np.empty(n_perm)
    for b in range(n_perm):
        perm = level_codes.copy()
        for blk in blocks:
            perm[blk] = rng.permutation(perm[blk])
        null[b] = stat(perm)

    sums = np.bincount(level_codes, weights=centred, minlength=L)
    cnts = np.bincount(level_codes, minlength=L)
    means = sums / np.maximum(cnts, 1)
    return {
        "statistic": observed,
        "p_value": float((1 + np.sum(null >= observed)) / (n_perm + 1)),
        "n_perm": n_perm,
        "level_means": {str(uniq_levels[i]): float(means[i]) for i in range(L)},
        "null_p95": float(np.percentile(null, 95)),
        "n_levels": L,
        "n_templates": T,
    }
