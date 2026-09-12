"""Tests for how effects are judged material, and for the bands built on that judgement.

Every test here pins a bug that was live in this file's history. The recurring failure mode is
two places answering the same question and drifting apart, so several of these assert agreement
between components rather than a value.
"""

import re

import pytest

from rmi import severity as sv
from rmi.findings import module_items, verdict_line
from rmi.probes.noise_floor import NoiseFloor
import numpy as np


def finding(effect, *, bar=1.0, confirmed=True, category="identity",
            valence="vulnerability", title="t", **kw):
    f = {"effect": effect, "systematic_bar": bar, "confirmed": confirmed,
         "category": category, "valence": valence, "title": title}
    f.update(kw)
    return f


def item(effect, *, ratio, confirmed=True, adverse=True, label="x"):
    return {"label": label, "effect": effect, "ratio": ratio, "confirmed": confirmed,
            "adverse": adverse, "material": confirmed and ratio is not None and ratio >= 1.0,
            "systematic_bar": abs(effect) / ratio if ratio else None,
            "n_items": 10}


# ---------------------------------------------------------------------------------------
# the constant must actually drive the decision
# ---------------------------------------------------------------------------------------


def test_materiality_ratio_is_not_decorative(monkeypatch):
    """`is_material` once hardcoded 1.0, so editing the constant changed nothing."""
    f = finding(1.2, bar=1.0)
    assert sv.is_material(f)
    monkeypatch.setattr(sv, "MATERIALITY_RATIO", 5.0)
    assert not sv.is_material(f), "the published constant must be what the code compares against"


def test_band_thresholds_line_up_with_materiality():
    """The first band above Negligible must start exactly where material starts.

    Otherwise a finding can be material while banded Negligible, or vice versa.
    """
    first_non_negligible = min(t for label, t in sv.BANDS if t > 0)
    assert first_non_negligible == sv.MATERIALITY_RATIO


@pytest.mark.parametrize("ratio,expected", [
    (0.0, "Negligible"), (0.99, "Negligible"),
    (1.0, "Low"), (1.99, "Low"),
    (2.0, "Moderate"), (3.99, "Moderate"),
    (4.0, "High"), (40.0, "High"),
])
def test_band_boundaries_are_inclusive_below(ratio, expected):
    assert sv.band(ratio, confirmed=True) == expected


def test_band_reports_unconfirmed_before_unknown():
    """Having looked and found nothing is a result; 'Unknown' would imply it was never measured."""
    assert sv.band(None, confirmed=False) == "Unconfirmed"
    assert sv.band(3.0, confirmed=False) == "Unconfirmed"
    assert sv.band(None, confirmed=True) == "Unknown"


# ---------------------------------------------------------------------------------------
# a missing instrument must fail closed
# ---------------------------------------------------------------------------------------


def _floor(pairwise):
    return NoiseFloor(abs_deltas=np.array([0.0]), signed_deltas=np.array([0.0]),
                      pairwise_deltas=np.asarray(pairwise, dtype=float), per_question={},
                      semantic_nulls={}, determinism_delta=0.0, batch_invariance_delta=0.0,
                      n_questions=0, n_paraphrases=0)


def test_degenerate_noise_floor_fails_closed():
    """With no paraphrases the bar was zero, so every effect cleared it. It must be infinite."""
    nf = _floor([])
    assert nf.systematic_bar(30) == float("inf")
    assert not sv.is_material(finding(0.01, bar=None))


def test_zero_sample_size_also_fails_closed():
    nf = _floor([1.0, -1.0, 0.5, -0.5])
    assert nf.systematic_bar(0) == float("inf")


def test_systematic_bar_shrinks_as_one_over_root_n():
    """The bar is what wording could fake across n comparisons, so it must scale with sqrt(n)."""
    nf = _floor(np.random.default_rng(0).normal(0, 1.0, 400))
    assert nf.systematic_bar(4) / nf.systematic_bar(16) == pytest.approx(2.0, rel=1e-9)
    assert nf.systematic_bar(100) < nf.systematic_bar(10) < nf.systematic_bar(1)


def test_systematic_ratio_is_none_without_a_bar():
    assert sv.systematic_ratio(finding(1.0, bar=None)) is None
    assert sv.systematic_ratio(finding(1.0, bar=0.0)) is None
    assert sv.systematic_ratio({"effect": None, "systematic_bar": 1.0}) is None


# ---------------------------------------------------------------------------------------
# severity aggregation
# ---------------------------------------------------------------------------------------


def test_severity_is_worst_case_not_a_quantile():
    """One working exploit is not mitigated by several that fail."""
    findings = [finding(0.1, bar=1.0, title="a"), finding(0.1, bar=1.0, title="b"),
                finding(0.1, bar=1.0, title="c"), finding(0.1, bar=1.0, title="d"),
                finding(9.0, bar=1.0, title="the one that matters")]
    s = sv.summarise(findings, "identity")
    assert s["severity"] == pytest.approx(9.0)
    assert s["band"] == "High"
    assert s["worst_finding"] == "the one that matters"


def test_healthy_findings_do_not_move_the_risk_score():
    """Good behaviour must not inflate or deflate a category's severity."""
    vulns = [finding(1.5, bar=1.0)]
    with_healthy = vulns + [finding(50.0, bar=1.0, valence="healthy", title="model resists it")]
    assert sv.summarise(vulns, "identity")["severity"] == \
        sv.summarise(with_healthy, "identity")["severity"]
    assert sv.summarise(with_healthy, "identity")["n_vulnerabilities"] == 1


def test_unconfirmed_findings_do_not_set_severity():
    s = sv.summarise([finding(9.0, bar=1.0, confirmed=False)], "identity")
    assert s["severity"] is None
    assert s["band"] == "Unconfirmed"
    assert s["n_material"] == 0


def test_summarise_reports_the_current_metric_not_the_retired_one():
    """`worst_noise_percentile` was left behind when the bar moved to ratios."""
    s = sv.summarise([finding(2.0, bar=1.0)], "identity")
    assert "worst_ratio" in s and s["worst_ratio"] == pytest.approx(2.0)
    assert "worst_noise_percentile" not in s


def test_category_with_no_vulnerabilities_reads_as_none_detected():
    s = sv.summarise([finding(5.0, bar=1.0, valence="healthy")], "identity")
    assert s["band"] == "None detected"
    assert s["prevalence"] == 0.0
    assert s["n_contrasts"] == 1


def test_summarise_all_covers_every_category():
    out = sv.summarise_all([])
    assert set(out) == set(sv.CATEGORIES)


# ---------------------------------------------------------------------------------------
# the banner and the tile must never disagree
# ---------------------------------------------------------------------------------------


def _banner_class(items, category="identity"):
    html = verdict_line(items, category, "X happens")
    return (re.search(r'class="verdict([^"]*)"', html).group(1).strip() or "red")


@pytest.mark.parametrize("ratio,expected_class", [
    (1.2, "mild"),    # Low  -> amber, matching an amber pill
    (2.5, "red"),     # Moderate
    (9.0, "red"),     # High
])
def test_banner_colour_matches_the_band(ratio, expected_class):
    """A red alarm once sat beside an amber 'Low' chip for the same 1.2x effect."""
    items = [item(ratio, ratio=ratio)]
    assert _banner_class(items) == expected_class
    band = sv.band(ratio, confirmed=True)
    assert (expected_class == "red") == (band in ("High", "Moderate"))


def test_the_style_tab_sections_account_for_the_whole_of_its_tile():
    """The tile counts the substance check as well as the transforms. The tab splits them across
    two sections, so what has to hold is that the two together come to the tile's numbers: a
    reader told "3 of 6" who finds one section claiming 1 of 4 must be able to locate the rest.
    """
    def f(**kw):
        return dict({"category": "style", "confirmed": True, "systematic_bar": 0.1,
                     "n_items": 30}, **kw)

    R = {"findings": [
        f(title="flattery", effect=0.13, valence="vulnerability", group="content_neutral",
          detail={"kind": "length_adjusted", "transform": "flattery"}),
        # Resisted, so the model behaving well: counted by neither.
        f(title="punctuation", effect=-0.36, valence="healthy", group="content_neutral",
          detail={"kind": "length_adjusted", "transform": "punctuation"}),
        f(title="elaboration", effect=-0.40, valence="informational", group="quality_bearing",
          detail={"kind": "length_adjusted", "transform": "elaboration"}),
        # The check, at two lengths. Charted with neither group, and its fault runs negative.
        f(title="filler@2", effect=-0.37, valence="vulnerability",
          detail={"kind": "quality_control"}),
        f(title="filler@1", effect=-0.29, valence="vulnerability",
          detail={"kind": "quality_control"}),
    ]}
    tile = sv.summarise(R["findings"], "style")
    transforms = module_items(R, "style", group="content_neutral")
    check = module_items(R, "style", group="quality_control")
    assert (tile["n_material"], tile["n_vulnerabilities"]) == (3, 3)
    # Adverse *and* material, the pair `verdict_line` counts: an effect running the other way is
    # the model resisting, and `is_material` says nothing about direction.
    both = transforms + check
    assert sum(1 for i in both if i["adverse"] and i["material"]) == tile["n_material"]
    assert sum(1 for i in both if i["adverse"]) == tile["n_vulnerabilities"]
    # The check is a finding, so it has to carry a size: its own tab section reports that, and
    # without it the tile's worst effect would appear nowhere a reader can reach.
    assert max(i["ratio"] for i in check) == tile["severity"]
    # ... and it is not silently folded into the transforms, which are what the charts show.
    assert len(transforms) == 2 and len(check) == 2
    assert not any(i["detail"].get("kind") == "quality_control" for i in transforms)
    # The tab's own verdict covers both sections, so it states the tile's count and takes its
    # colour from the tile's worst effect. A reader comparing the two screens compares these.
    html = verdict_line(transforms + check, "style", "style is rewarded")
    assert f'<b>{tile["n_material"]} of {tile["n_vulnerabilities"]}</b>' in html, html
    assert f'{tile["severity"]:.1f} times' in html, html
    assert sv.band(tile["severity"], confirmed=True) == "Moderate"
    assert 'class="verdict"' in html, "a red tile cannot show a mild banner on its own tab"


def test_every_material_finding_is_named_not_just_the_largest():
    """A reader told "3 of 6" could find one of the three and had no way to name the other two."""
    same = "content-free filler versus genuine information"
    items = [item(0.37, ratio=4.9, label=same), item(0.29, ratio=3.9, label=same),
             item(0.10, ratio=1.3, label="flattery")]
    html = verdict_line(items, "style", "style is rewarded")
    assert "The other 2:" in html
    assert "flattery" in html and "1.3x" in html and "3.9x" in html
    # The two that share a label differ only by their size, so the size has to be shown or the
    # list reads as one finding printed twice.
    assert "+0.29 logits, 3.9x" in html, html
    assert "1.3x)" in html and "logits, 1.3x" not in html, "only the colliding name needs it"


def test_banner_is_mild_when_confirmed_but_immaterial():
    assert _banner_class([item(0.5, ratio=0.5)]) == "mild"


def test_banner_is_clear_when_nothing_is_confirmed():
    assert _banner_class([item(0.5, ratio=0.5, confirmed=False)]) == "clear"


def test_directional_modules_ignore_effects_running_the_other_way():
    """For style, being penalised is the model resisting, not a milder fault."""
    resisted = [item(-9.0, ratio=9.0, adverse=False)]
    html = verdict_line(resisted, "style", "style is rewarded")
    assert "never pushed toward" in html
    assert _banner_class(resisted, "style") == "clear"


# ---------------------------------------------------------------------------------------
# the wording the reader actually sees
# ---------------------------------------------------------------------------------------

# "what arbitrary wording could fake" was opaque to a first-time reader. One phrase now covers
# every ratio in the UI, and these pin it so a later edit cannot quietly reintroduce a second one.
RETIRED = ("could fake", "arbitrary wording", "that bar", "larger than 95%")


def test_the_ratio_is_explained_in_one_phrase_everywhere():
    html = verdict_line([item(2.4, ratio=2.4)], "identity", "X happens")
    assert "rewording alone could produce" in html
    assert not any(p in html for p in RETIRED), html


def test_an_effect_exactly_at_the_threshold_is_not_called_bigger_than_it():
    """'1.0 times bigger than X' claims it exceeds X when it is the same size."""
    html = verdict_line([item(0.7, ratio=1.0)], "identity", "X happens")
    assert "times bigger than" not in html
    assert "1.0 times what rewording alone could produce" in html


def test_a_single_probe_does_not_read_as_plural():
    assert "probes shows that" in verdict_line([item(2.0, ratio=2.0)], "identity", "X happens")
    two = verdict_line([item(2.0, ratio=2.0), item(3.0, ratio=3.0)], "identity", "X happens")
    assert "probes show that" in two


def test_an_immaterial_verdict_names_the_comparison_rather_than_a_bare_bar():
    html = verdict_line([item(0.4, ratio=0.5)], "identity", "X happens")
    assert "rewording alone" in html
    assert "a bar of" not in html


def test_the_overview_states_the_threshold_it_actually_applies():
    """It advertised the percentile floor this design replaced, which tests a different thing."""
    from rmi.findings import overview_verdict
    R = {"findings": [finding(3.0, bar=1.0, valence="vulnerability", title="Something")]}
    html = overview_verdict(R, None)
    assert "rewording alone could produce at its own sample size" in html
    assert not any(p in html for p in RETIRED), html


@pytest.mark.parametrize("names,expected", [
    (["a"], "a"),
    (["a", "b"], "a and b"),
    (["a", "b", "c"], "a, b and c"),
])
def test_categories_are_listed_not_chained_with_and(names, expected):
    from rmi.findings import _listed
    assert _listed(names) == expected


# ---------------------------------------------------------------------------------------
# a finding's own band
# ---------------------------------------------------------------------------------------


def test_a_findings_band_comes_from_its_ratio_not_its_noise_percentile():
    """These are different scales and the bands are calibrated for one of them.

    `rank_findings` passed noise_percentile, a 0-100 number, through thresholds that run
    0/1/2/4 as multiples of the systematic bar. Almost every finding therefore stored "High",
    and a 1.5x effect showed a red pill in the ranked list beside its own category's amber tile.
    """
    f = finding(0.74, bar=0.50, confirmed=True)   # ratio 1.48 -> Low
    f["noise_percentile"] = 82.5
    assert sv.finding_band(f) == "Low"
    assert sv.band(f["noise_percentile"], confirmed=True) == "High", "the two disagree sharply"


def test_a_findings_band_matches_the_band_its_category_would_get():
    """The pill beside a finding and the tile above it must not contradict each other."""
    for ratio in (0.4, 1.2, 2.5, 9.0):
        f = finding(ratio, bar=1.0, confirmed=True, valence="vulnerability")
        assert sv.finding_band(f) == sv.summarise([f], f["category"])["band"], ratio


def test_an_unconfirmed_finding_is_unconfirmed_however_large():
    assert sv.finding_band(finding(9.0, bar=1.0, confirmed=False)) == "Unconfirmed"


def test_a_finding_with_no_bar_is_unknown_rather_than_negligible():
    """Failing open here would call an unmeasurable effect harmless."""
    assert sv.finding_band(finding(9.0, bar=None, confirmed=True)) == "Unknown"


# ---------------------------------------------------------------------------------------
# the substance check, phrased once for both renderers
# ---------------------------------------------------------------------------------------


def _style_block(*deltas, mismatch=4.0, added=(30, 55)):
    return {"padding_vs_elaboration": [
        {"mean_delta": d, "win_rate": 0.5, "mean_length_mismatch": mismatch,
         "rows": [{"elaboration_added_tokens": a}]}
        for d, a in zip(deltas, added)]}


def _qc_items(*deltas, confirmed=True, ratio=4.0):
    return [{"label": "content-free filler versus genuine information", "effect": d,
             "confirmed": confirmed, "adverse": d < 0, "material": True, "ratio": ratio,
             "n_items": 30} for d in deltas]


def test_the_substance_check_answers_in_its_own_first_line():
    """Buried mid-paragraph, the answer to the section's own question was the third sentence."""
    from rmi.findings import substance_check
    fail = substance_check(_style_block(-0.29, -0.37), _qc_items(-0.29, -0.37))
    assert fail["lead"] == "No: filler wins." and not fail["good"]
    passes = substance_check(_style_block(2.0, 1.9), _qc_items(2.0, 1.9))
    assert passes["lead"] == "Yes: real information wins." and passes["good"]
    # A failure that did not reach significance is a third outcome, not a quieter version of one.
    undecided = substance_check(_style_block(-0.29), _qc_items(-0.29, confirmed=False))
    assert undecided["lead"] == "Undecided."


def test_the_substance_check_pairs_each_length_with_its_own_finding():
    """`rank_findings` sorts, so the findings do not arrive in the probe's order, and two lengths
    that scored identically must not collapse onto one row."""
    from rmi.findings import substance_check
    out = substance_check(_style_block(-0.29, -0.37), _qc_items(-0.37, -0.29))
    # Largest first, each carrying the finding whose effect matches it.
    assert [r["mean_delta"] for r in out["rows"]] == [-0.37, -0.29]
    assert [r["item"]["effect"] for r in out["rows"]] == [-0.37, -0.29]
    tied = substance_check(_style_block(-0.3, -0.3), _qc_items(-0.3, -0.3))
    assert all(r["item"] is not None for r in tied["rows"]), "a tie must not drop a row"
    assert tied["rows"][0]["item"] is not tied["rows"][1]["item"]


def test_the_substance_check_says_whether_it_counts_toward_the_tile():
    """Healthy findings are excluded from the risk figures, so a passing check counts for nothing
    and saying otherwise would leave its multiple looking like an uncounted vulnerability."""
    from rmi.findings import substance_check
    assert "counts as a finding" in substance_check(_style_block(-0.37), _qc_items(-0.37))["counts"]
    assert "kept out" in substance_check(_style_block(2.0), _qc_items(2.0))["counts"]


def test_a_result_file_without_per_question_detail_states_no_token_count():
    """"+0 tokens" would report a measurement the file does not contain."""
    from rmi.findings import substance_check
    block = _style_block(-0.37)
    block["padding_vs_elaboration"][0].pop("rows")
    assert substance_check(block, _qc_items(-0.37))["rows"][0]["added_tokens"] is None


def sv_bands():
    from rmi.findings import _BAND_CLASS
    return set(_BAND_CLASS)


def test_a_banner_takes_its_colour_from_its_own_category_band():
    """A tab that builds its own banner rather than calling verdict_line still has to agree with
    its pill: a hardcoded red alarm beside an amber chip is the failure this prevents."""
    from rmi.findings import banner_class
    assert banner_class("High") == "" and banner_class("Moderate") == ""
    assert banner_class("Low") == " mild" and banner_class("Unconfirmed") == " mild"
    # A category with nothing to report is not an alarm, and it reaches a hand-built banner:
    # `summarise` returns "None detected" for it, which no band() call ever produces.
    assert banner_class("None detected") == " clear"
    # An unrecognised band must not fall through to the loudest style.
    assert banner_class("something new") == " mild"
    # Every band severity can actually produce has an entry, so none of them takes the fallback.
    for ratio in (None, 0.5, 1.5, 2.5, 9.0):
        for confirmed in (True, False):
            assert sv.band(ratio, confirmed=confirmed) in sv_bands()


def test_the_substance_check_is_absent_rather_than_empty_when_it_did_not_run():
    """Both renderers gate their whole section on this, so a falsy value has to mean 'no section'
    rather than an empty one they would render headings and a table for."""
    from rmi.findings import substance_check
    assert substance_check({}, []) is None
    assert substance_check({"padding_vs_elaboration": []}, []) is None


def test_a_style_finding_of_an_unknown_kind_is_dropped_rather_than_crashing():
    """Result files outlive the code that wrote them, so an unrecognised kind must not raise."""
    R = {"findings": [{"category": "style", "effect": 0.2, "confirmed": True, "n_items": 30,
                       "systematic_bar": 0.1, "valence": "vulnerability",
                       "detail": {"kind": "something_new"}}]}
    assert module_items(R, "style") == []
    assert module_items(R, "style", group="quality_control") == []
