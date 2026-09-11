"""Tests for the length-controlled style regression.

The headline test is ``test_pure_length_rule_yields_zero_style_effect``: if the scorer's only rule
is "longer is better", the model must report the length slope correctly *and* every style effect
as zero. If that fails, the whole style module is measuring length and calling it style.
"""

import numpy as np
import pytest

from rmi.stats.regression import fit_fe, length_curve, rcs_basis, rcs_knots


def test_rcs_basis_is_linear_beyond_the_outer_knots():
    knots = np.array([0.0, 10.0, 20.0, 30.0])
    x = np.array([30.0, 40.0, 50.0, 60.0])
    B = rcs_basis(x, knots)
    # A linear function has constant second differences of zero in every column.
    for j in range(B.shape[1]):
        d2 = np.diff(B[:, j], 2)
        assert np.allclose(d2, 0, atol=1e-8), f"column {j} is not linear past the last knot"


def test_rcs_basis_shape():
    knots = rcs_knots(np.linspace(0, 100, 200), n_knots=4)
    B = rcs_basis(np.linspace(0, 100, 50), knots)
    assert B.shape == (50, knots.size - 1)


def _style_design(n_questions=40, dose_effect=(0.0, 0.0), length_fn=None, seed=0, nonlinear=False):
    """Build a style-module-shaped dataset: every question appears in every arm."""
    rng = np.random.default_rng(seed)
    transforms = ["emoji", "hedging"]
    tokens_per_dose = {"emoji": 4.0, "hedging": 12.0}
    rows = []
    for q in range(n_questions):
        q_base = rng.normal(0, 2.0)  # question difficulty, absorbed by fixed effects
        base_len = rng.uniform(60, 200)
        # baseline arm
        rows.append((q, "baseline", 0, base_len, q_base))
        # pure length ladder
        for dose in (1, 2, 3, 4):
            rows.append((q, "length", dose, base_len + 40 * dose, q_base))
        # transform ladders
        for ti, t in enumerate(transforms):
            for dose in (1, 2, 4):
                L = base_len + tokens_per_dose[t] * dose
                rows.append((q, t, dose, L, q_base))
    qs, arms, doses, lens, bases = zip(*rows)
    qs = np.array(qs); arms = np.array(arms); doses = np.array(doses, float)
    lens = np.array(lens, float); bases = np.array(bases)
    if length_fn is None:
        length_fn = (lambda L: 0.004 * L) if not nonlinear else (lambda L: 1.5 * np.log1p(L / 50))
    y = bases + length_fn(lens)
    for ti, t in enumerate(transforms):
        y = y + np.where(arms == t, dose_effect[ti] * doses, 0.0)
    return qs, arms, doses, lens, y


def _fit(qs, arms, doses, lens, y, n_boot=300, seed=0):
    knots = rcs_knots(lens, n_knots=4)
    B = rcs_basis(lens, knots)
    cols = [B]
    names = [f"len{j}" for j in range(B.shape[1])]
    for t in ["emoji", "hedging"]:
        cols.append(np.where(arms == t, doses, 0.0).reshape(-1, 1))
        names.append(f"dose_{t}")
    X = np.column_stack(cols)
    length_cols = list(range(B.shape[1]))
    fit = fit_fe(y, X, qs, names, length_cols=length_cols, knots=knots,
                 n_boot=n_boot, seed=seed)
    return fit, knots


def test_pure_length_rule_yields_zero_style_effect():
    """Planted rule: reward = 0.004 * tokens, no style effect at all."""
    qs, arms, doses, lens, y = _style_design(dose_effect=(0.0, 0.0), seed=1)
    fit, knots = _fit(qs, arms, doses, lens, y)
    for t in ["emoji", "hedging"]:
        r = fit.get(f"dose_{t}")
        assert abs(r["coef"]) < 1e-6, f"{t} style effect should be zero, got {r['coef']}"
    curve = length_curve(fit, np.linspace(lens.min(), lens.max(), 20), knots)
    # Planted slope is 0.004/token == 0.4 per 100 tokens.
    assert np.allclose(curve["slope_per_100_tokens"], 0.4, atol=0.02)


def test_recovers_a_style_effect_on_top_of_a_nonlinear_length_curve():
    qs, arms, doses, lens, y = _style_design(
        dose_effect=(0.25, -0.10), nonlinear=True, seed=2
    )
    fit, knots = _fit(qs, arms, doses, lens, y, n_boot=400)
    assert fit.get("dose_emoji")["coef"] == pytest.approx(0.25, abs=0.02)
    assert fit.get("dose_hedging")["coef"] == pytest.approx(-0.10, abs=0.02)


def test_linear_length_model_would_misattribute_curvature_to_style():
    """Why the spline matters: fitting length linearly leaks curvature into the style terms."""
    qs, arms, doses, lens, y = _style_design(dose_effect=(0.0, 0.0), nonlinear=True, seed=3)
    # spline version
    fit_spline, _ = _fit(qs, arms, doses, lens, y, n_boot=100)
    # linear version
    X = np.column_stack([
        lens.reshape(-1, 1),
        np.where(arms == "emoji", doses, 0.0).reshape(-1, 1),
        np.where(arms == "hedging", doses, 0.0).reshape(-1, 1),
    ])
    fit_lin = fit_fe(y, X, qs, ["len", "dose_emoji", "dose_hedging"],
                     length_cols=[0], n_boot=100, seed=0)
    spline_err = abs(fit_spline.get("dose_hedging")["coef"])
    linear_err = abs(fit_lin.get("dose_hedging")["coef"])
    assert spline_err < linear_err, (
        f"spline should be closer to the true zero; spline={spline_err:.4f} linear={linear_err:.4f}"
    )
    assert spline_err < 0.01


def test_wild_bootstrap_rejects_a_real_effect_and_not_a_null_one():
    qs, arms, doses, lens, y = _style_design(dose_effect=(0.25, 0.0), seed=4)
    rng = np.random.default_rng(0)
    y = y + rng.normal(0, 0.3, y.size)  # add noise so the test is not degenerate
    fit, _ = _fit(qs, arms, doses, lens, y, n_boot=600)
    assert fit.get("dose_emoji")["p"] < 0.05
    assert fit.get("dose_hedging")["p"] > 0.05


def test_question_fixed_effects_absorb_question_difficulty():
    """Huge between-question variance must not disturb the within-question estimates."""
    qs, arms, doses, lens, y = _style_design(dose_effect=(0.25, 0.0), seed=5)
    y_shifted = y + np.where(qs % 2 == 0, 500.0, -500.0)
    fit_a, _ = _fit(qs, arms, doses, lens, y, n_boot=100)
    fit_b, _ = _fit(qs, arms, doses, lens, y_shifted, n_boot=100)
    assert fit_a.get("dose_emoji")["coef"] == pytest.approx(
        fit_b.get("dose_emoji")["coef"], abs=1e-8
    )
