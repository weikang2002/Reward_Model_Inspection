"""Tests for the inference layer.

The important ones here are not "does the code run" but "does the code refuse to lie": the
cluster bootstrap must actually widen intervals on dependent data, and the identity permutation
test must not silently return p = 1.
"""

import numpy as np
import pytest

from rmi.stats.inference import (
    bh_fdr,
    cluster_bootstrap_ci,
    cluster_se_of_mean,
    max_group_gap_permutation,
    paired_contrast,
    signflip_null_t,
    westfall_young_adjust,
    wild_cluster_bootstrap_p,
    _cluster_index,
)


def _t_stat(d, c):
    codes, uniq = _cluster_index(c)
    return d.mean() / cluster_se_of_mean(d, codes, uniq.size)


def _clustered_data(n_clusters=30, per_cluster=10, icc_shift=1.0, seed=0):
    """Data where most variance is *between* clusters, which is the case we must not get wrong."""
    rng = np.random.default_rng(seed)
    vals, clus = [], []
    for g in range(n_clusters):
        offset = rng.normal(0, icc_shift)
        vals.append(rng.normal(offset, 0.1, per_cluster))
        clus.append(np.full(per_cluster, g))
    return np.concatenate(vals), np.concatenate(clus)


def test_cluster_bootstrap_is_wider_than_row_bootstrap_on_dependent_data():
    vals, clus = _clustered_data()
    lo_c, hi_c = cluster_bootstrap_ci(vals, clus, n_boot=2000, seed=1)
    # Row bootstrap == every row its own cluster.
    lo_r, hi_r = cluster_bootstrap_ci(vals, np.arange(vals.size), n_boot=2000, seed=1)
    assert (hi_c - lo_c) > 2.0 * (hi_r - lo_r), (
        "cluster bootstrap must be much wider when variance is between clusters; "
        f"cluster width={hi_c - lo_c:.4f} row width={hi_r - lo_r:.4f}"
    )


def test_cluster_se_matches_row_se_when_every_row_is_its_own_cluster():
    rng = np.random.default_rng(3)
    x = rng.normal(size=200)
    codes, uniq = _cluster_index(np.arange(x.size))
    se = cluster_se_of_mean(x, codes, uniq.size)
    # Sandwich SE of a mean with singleton clusters is the population-style SEM (1/n normalisation)
    assert se == pytest.approx(x.std(ddof=0) / np.sqrt(x.size), rel=1e-10)


def test_cluster_bootstrap_covers_truth():
    """Nominal 95% interval should cover a known mean close to 95% of the time."""
    covered = 0
    trials = 120
    for s in range(trials):
        vals, clus = _clustered_data(n_clusters=40, per_cluster=5, icc_shift=1.0, seed=100 + s)
        lo, hi = cluster_bootstrap_ci(vals, clus, n_boot=600, seed=s)
        if lo <= 0.0 <= hi:
            covered += 1
    assert 0.86 <= covered / trials <= 1.0, f"coverage {covered / trials:.3f}"


# ---------------------------------------------------------------------------------------
# the identity omnibus: the degenerate-statistic trap
# ---------------------------------------------------------------------------------------


def _identity_frame(effect=0.0, n_templates=12, n_names_per_group=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    groups = ["A", "B"]
    for gi, g in enumerate(groups):
        for k in range(n_names_per_group):
            nm = f"{g}_name{k}"
            name_off = rng.normal(0, 0.05)
            for t in range(n_templates):
                tmpl_off = rng.normal(0, 1.0)  # big template effect, must be removed
                rows.append((tmpl_off + name_off + gi * effect, nm, g, f"tmpl{t}"))
    s, n, gr, tm = zip(*rows)
    return np.array(s), np.array(n), np.array(gr), np.array(tm)


def test_permutation_test_is_not_degenerate_under_a_real_effect():
    s, n, g, t = _identity_frame(effect=0.30, seed=1)
    res = max_group_gap_permutation(s, n, g, t, n_perm=2000, seed=0)
    assert res["p_value"] < 0.05, res
    assert res["statistic"] == pytest.approx(0.30, abs=0.08)


def test_permutation_test_is_null_when_there_is_no_effect():
    s, n, g, t = _identity_frame(effect=0.0, seed=2)
    res = max_group_gap_permutation(s, n, g, t, n_perm=2000, seed=0)
    assert res["p_value"] > 0.05, res


def test_name_level_max_gap_would_be_degenerate():
    """Pins the reason the statistic is computed on group means.

    A max-minus-min over individual name scores is invariant to relabelling, so the permutation
    p-value is exactly 1 by construction. If someone 'simplifies' the implementation to name
    level, this test documents what breaks.
    """
    s, n, g, t = _identity_frame(effect=0.9, seed=3)
    uniq = np.unique(n)
    name_mean = np.array([s[n == u].mean() for u in uniq])
    rng = np.random.default_rng(0)
    observed = name_mean.max() - name_mean.min()
    null = [
        (lambda p: p.max() - p.min())(rng.permutation(name_mean)) for _ in range(200)
    ]
    assert all(x == pytest.approx(observed) for x in null)


def test_permutation_removes_template_effects():
    """Huge template variance must not create a spurious group gap."""
    s, n, g, t = _identity_frame(effect=0.0, n_templates=20, seed=7)
    s = s + np.where(np.isin(t, ["tmpl0", "tmpl1"]), 50.0, 0.0)  # two wild templates
    res = max_group_gap_permutation(s, n, g, t, n_perm=1500, seed=0)
    assert res["p_value"] > 0.05
    assert abs(res["statistic"]) < 0.2


# ---------------------------------------------------------------------------------------
# multiplicity
# ---------------------------------------------------------------------------------------


def test_westfall_young_controls_fwer_under_the_global_null():
    """With 20 pure-null correlated contrasts, at most ~5% of runs should reject anything."""
    rejects = 0
    trials = 60
    for s in range(trials):
        rng = np.random.default_rng(s)
        shared = rng.normal(0, 1.0, size=(25, 1))  # induces correlation between contrasts
        deltas, clusters = [], []
        for j in range(20):
            d = (shared + rng.normal(0, 1.0, size=(25, 1))).ravel()
            deltas.append(d)
            clusters.append(np.arange(25))
        null_t = signflip_null_t(deltas, clusters, n_perm=400, seed=s)
        obs = [_t_stat(d, c) for d, c in zip(deltas, clusters)]
        adj = westfall_young_adjust(obs, null_t)
        if np.min(adj) < 0.05:
            rejects += 1
    assert rejects / trials <= 0.20, f"FWER looks broken: {rejects / trials:.3f}"


def test_westfall_young_detects_a_real_effect():
    rng = np.random.default_rng(0)
    deltas, clusters = [], []
    for j in range(10):
        shift = 1.5 if j == 0 else 0.0
        deltas.append(rng.normal(shift, 1.0, 40))
        clusters.append(np.arange(40))
    null_t = signflip_null_t(deltas, clusters, n_perm=2000, seed=0)
    obs = [_t_stat(d, c) for d, c in zip(deltas, clusters)]
    adj = westfall_young_adjust(obs, null_t)
    assert adj[0] < 0.01
    assert np.all(adj[1:] > 0.05)


def test_westfall_young_is_never_more_lenient_than_raw():
    rng = np.random.default_rng(1)
    deltas = [rng.normal(0.3, 1.0, 30) for _ in range(8)]
    clusters = [np.arange(30) for _ in range(8)]
    null_t = signflip_null_t(deltas, clusters, n_perm=1000, seed=0)
    obs = [_t_stat(d, c) for d, c in zip(deltas, clusters)]
    adj = westfall_young_adjust(obs, null_t)
    raw = [(1 + np.sum(np.abs(null_t[:, j]) >= abs(obs[j]))) / (null_t.shape[0] + 1)
           for j in range(8)]
    assert np.all(adj >= np.array(raw) - 1e-12)


def test_bh_fdr_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests

    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.6])
    assert np.allclose(bh_fdr(p), multipletests(p, method="fdr_bh")[1])


# ---------------------------------------------------------------------------------------
# paired contrast end to end
# ---------------------------------------------------------------------------------------


def test_paired_contrast_on_exact_null_finds_nothing():
    d = np.zeros(50)
    r = paired_contrast(d, np.arange(50), name="null", n_boot=500)
    assert r.mean_delta == 0.0
    assert r.p_sign == 1.0
    assert r.ci_low == 0.0 and r.ci_high == 0.0


def test_paired_contrast_recovers_a_planted_effect():
    rng = np.random.default_rng(0)
    d = rng.normal(0.5, 0.2, 60)
    r = paired_contrast(d, np.arange(60), name="planted", n_boot=2000)
    assert r.mean_delta == pytest.approx(0.5, abs=0.08)
    assert r.ci_low < 0.5 < r.ci_high
    assert r.p_sign < 1e-6
    assert r.win_rate > 0.95


def test_wild_cluster_bootstrap_null_is_uniformish():
    ps = []
    for s in range(80):
        vals, clus = _clustered_data(n_clusters=12, per_cluster=8, icc_shift=1.0, seed=500 + s)
        ps.append(wild_cluster_bootstrap_p(vals, clus, n_boot=300, seed=s))
    assert np.mean(np.array(ps) < 0.05) <= 0.15, f"rejection rate {np.mean(np.array(ps) < 0.05)}"
