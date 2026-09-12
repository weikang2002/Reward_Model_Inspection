"""Tests for the presentation layer: the diff, the charts, and the standalone report.

These were the largest untested surface. None of it raises on a wrong number, so the failure mode
is a chart that quietly plots the wrong field, or a report that ships missing a section.
"""

import re
from html import escape

import pytest

from rmi import viz
from rmi import methodology
from rmi.findings import self_check
from rmi.report import build
from rmi.runner import run_scan
from rmi.scoring import StubScorer
from rmi.textdiff import word_diff


# ---------------------------------------------------------------------------------------
# word diff
# ---------------------------------------------------------------------------------------


def test_diff_marks_only_the_inserted_span():
    left, right = word_diff("a bad answer", "a bad answer plus payload")
    assert "<ins>" not in left and "<del>" not in left
    assert "<ins>plus payload</ins>" in right


def test_diff_marks_a_substitution_on_both_sides():
    left, right = word_diff("this a Christian professional", "this a Jewish professional")
    assert "<del>Christian</del>" in left
    assert "<ins>Jewish</ins>" in right


def test_diff_applies_the_insert_class_so_a_payload_is_not_coloured_as_an_improvement():
    _, right = word_diff("bad", "bad [SEP] good", ins_class="attack")
    assert '<ins class="attack">' in right


def test_diff_escapes_html_in_the_source_text():
    """Scored text is arbitrary; an unescaped tag would inject markup into the page."""
    left, right = word_diff("<script>alert(1)</script>", "safe")
    assert "<script>" not in left
    assert "&lt;script&gt;" in left
    assert "alert" in left


def test_identical_text_produces_no_marks():
    left, right = word_diff("same words here", "same words here")
    assert "<ins>" not in right and "<del>" not in left


# ---------------------------------------------------------------------------------------
# charts encode the field they claim to
# ---------------------------------------------------------------------------------------


def _items(*specs):
    return [{"label": lab, "effect": eff, "ratio": ratio, "confirmed": True,
             "adverse": eff > 0, "material": ratio >= 1.0, "systematic_bar": abs(eff) / ratio,
             "n_items": 10, "detail_text": ""}
            for lab, eff, ratio in specs]


def test_bias_bars_plots_the_ratio_not_the_raw_effect():
    """The scale moved from logits to multiples of the bar; plotting `effect` would look sane."""
    items = _items(("a", 4.0, 2.0), ("b", 1.0, 0.5))
    data = viz.bias_bars(items).data[0]
    assert sorted(data.x) == [0.5, 2.0], "x must be the ratio"
    assert set(data.y) == {"a", "b"}


def test_bias_bars_signed_mode_puts_a_resisted_effect_left_of_zero():
    fig = viz.bias_bars(_items(("penalised", -3.0, 3.0)), signed=True)
    assert fig.data[0].x[0] == pytest.approx(-3.0)


def test_bias_bars_unsigned_mode_uses_magnitude_only():
    fig = viz.bias_bars(_items(("gap", -3.0, 3.0)), signed=False)
    assert fig.data[0].x[0] == pytest.approx(3.0)


def test_bias_bars_survives_items_with_no_bar():
    """A probe with no sample size has no ratio; it must be dropped, not crash or plot as zero."""
    items = _items(("ok", 2.0, 2.0))
    items.append({"label": "no bar", "effect": 1.0, "ratio": None, "confirmed": True,
                  "adverse": True, "material": False, "systematic_bar": None, "n_items": None})
    fig = viz.bias_bars(items)
    assert list(fig.data[0].y) == ["ok"]


def test_bias_bars_with_nothing_to_plot_returns_an_empty_figure():
    fig = viz.bias_bars([])
    assert len(fig.data) == 0


def _bars(fig):
    """Every bar across every trace. Bars are grouped into one trace per valence."""
    return [(y, x) for t in fig.data for y, x in zip(t.y, t.x)]


def test_findings_vs_noise_ranks_by_ratio_and_keeps_the_largest():
    findings = [{"title": f"f{i}", "effect": float(i), "systematic_bar": 1.0,
                 "valence": "vulnerability"} for i in range(1, 25)]
    fig = viz.findings_vs_noise(findings)
    xs = [x for _, x in _bars(fig)]
    assert max(xs) == 24.0, "the largest finding must survive the cap"
    assert len(xs) == viz.MAX_ROWS, "the chart caps how many it shows"


def test_every_bar_is_labelled_with_its_category():
    """Titles alone made it impossible to tell a reward-hacking row from a style row."""
    findings = [
        {"title": "x", "effect": 3.0, "systematic_bar": 1.0, "valence": "vulnerability",
         "category": "reward_hacking",
         "detail": {"kind": "best_affix", "affix_id": "sep_double"}},
        {"title": "y", "effect": 2.0, "systematic_bar": 1.0, "valence": "informational",
         "category": "style", "detail": {"kind": "length_adjusted", "transform": "emoji"}},
    ]
    labels = [y for y, _ in _bars(viz.findings_vs_noise(findings))]
    assert any("sep_double" in lab for lab in labels), labels
    assert any("emoji" in lab for lab in labels), labels
    # Trailing, so right-aligned ticks put the categories in a column beside the bars.
    assert all(lab.endswith("</b>") for lab in labels), labels
    assert {lab.rsplit("<b>", 1)[1].rstrip("</b>") for lab in labels} == {"Reward hacking", "Style"}


def test_a_finding_with_an_unexpected_valence_is_still_drawn():
    """Grouping into traces by valence would otherwise drop anything off the known list."""
    findings = [{"title": "odd", "effect": 3.0, "systematic_bar": 1.0, "valence": "surprise",
                 "category": "style", "detail": {}},
                {"title": "normal", "effect": 2.0, "systematic_bar": 1.0,
                 "valence": "vulnerability", "category": "style", "detail": {}}]
    assert len(_bars(viz.findings_vs_noise(findings))) == 2


def test_two_findings_that_describe_the_same_thing_get_distinct_rows():
    """Bars sharing a y value land on one row, so a duplicate label hides a finding outright."""
    same = {"category": "style", "detail": {"kind": "quality_control"},
            "systematic_bar": 1.0, "valence": "healthy", "title": "elaboration beats filler"}
    fig = viz.findings_vs_noise([dict(same, effect=2.01), dict(same, effect=1.98)])
    labels = [y for y, _ in _bars(fig)]
    assert len(set(labels)) == 2, labels
    assert len(_bars(fig)) == 2


def test_the_chart_legend_names_what_each_colour_means():
    """The colours used to be decoded only by a caption underneath."""
    findings = [{"title": "a", "effect": 3.0, "systematic_bar": 1.0, "valence": v,
                 "category": "style", "detail": {}}
                for v in ("vulnerability", "healthy", "informational")]
    fig = viz.findings_vs_noise(findings)
    assert fig.layout.showlegend
    assert {t.name for t in fig.data} == {n for _, n in viz.VALENCE_LEGEND}


def test_the_sorted_order_survives_being_split_across_traces():
    findings = [{"title": f"f{i}", "effect": float(i), "systematic_bar": 1.0,
                 "category": "style", "detail": {},
                 "valence": "vulnerability" if i % 2 else "healthy"} for i in range(1, 9)]
    fig = viz.findings_vs_noise(findings)
    order = list(fig.layout.yaxis.categoryarray)
    by_label = dict(_bars(fig))
    assert [by_label[lab] for lab in order] == sorted(by_label.values())


def test_an_insistence_level_reads_as_english_not_as_an_identifier():
    """The label once interpolated the raw id, giving "when the user is expertise"."""
    f = {"title": "t", "effect": 2.0, "systematic_bar": 1.0, "valence": "vulnerability",
         "category": "sycophancy",
         "detail": {"kind": "insistence_slope", "level": "expertise"}}
    lab = [y for y, _ in _bars(viz.findings_vs_noise([f]))][0]
    assert "claims expertise" in lab
    assert "is expertise" not in lab


def test_exploit_ranking_orders_by_success_rate_not_lift():
    """Lift is largest where the base started lowest, which is not where the attack works."""
    exploits = [
        {"label": "big lift", "kind": "single affix", "lift": 9.0,
         "asr": {"p50": 0.02}, "examples": []},
        {"label": "actually works", "kind": "single affix", "lift": 1.0,
         "asr": {"p50": 0.30}, "examples": []},
    ]
    fig = viz.exploit_ranking(exploits, 0.0)
    # Horizontal bars are drawn bottom-up, so the last entry is the top of the chart.
    assert fig.data[0].y[-1] == "actually works"


# ---------------------------------------------------------------------------------------
# the standalone report
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    scan = run_scan("stub", depth="quick", calibrate=False, verbose=False, out_dir=out,
                    scorer=StubScorer(rule="length", coef=0.02))
    path = build(scan, out / "report.html")
    return scan, path.read_text()


def test_report_is_self_contained(rendered):
    """It must render offline, so plotly is inlined rather than fetched.

    Checks for tags that actually fetch at render time, not for the substring. The inlined bundle
    contains URLs in code paths that never run here, so a substring search reports a false alarm.
    """
    _, html = rendered
    assert "Plotly.newPlot" in html, "the charts must be inlined, not linked"
    tags = re.findall(r"<(?:script|link|img|iframe)\b[^>]*>", html, flags=re.I)
    external = [t for t in tags
                if re.search(r"(?:src|href)=[\"\']https?://", t, flags=re.I)]
    assert not external, external


def test_report_carries_every_module_section(rendered):
    _, html = rendered
    for heading in ("Where the problems are", "Bias · Identity", "Bias · Sycophancy",
                    "Bias · Style & length", "Reward hacking", "Appendix"):
        assert heading in html, heading


def test_report_opens_with_a_verdict(rendered):
    _, html = rendered
    assert re.search(r'class=[\'"]verdict', html), "the answer must come before the evidence"


def test_report_bands_agree_with_the_scan(rendered):
    """The report derives severity on load; it must not contradict the run it renders."""
    scan, html = rendered
    from rmi import severity as sv
    for cat, s in sv.summarise_all(scan["findings"]).items():
        assert s["band"] in html, f"{cat} band {s['band']} missing from the report"


def test_report_escapes_model_supplied_text(rendered):
    """Corpus text reaches the page; an unescaped angle bracket would break the document."""
    _, html = rendered
    assert "<script>alert" not in html


# ---------------------------------------------------------------------------------------
# the model card self-check
# ---------------------------------------------------------------------------------------


def test_self_check_reads_a_pass_in_the_direction_a_reader_expects():
    c = self_check({"sanity_check": {"passed": True, "margin": 1.56}})
    assert c["passed"] is True
    assert "supportive reply 1.56 logits above an abusive one" in c["headline"]


def test_self_check_states_a_failure_as_abuse_scoring_higher():
    """The margin is negative on a failure. Printing it raw would read as abuse scoring lower."""
    c = self_check({"sanity_check": {"passed": False, "margin": -0.61}})
    assert c["passed"] is False
    assert "abusive reply 0.61 logits above a supportive one" in c["headline"]
    assert "-0.61" not in c["headline"] and "−0.61" not in c["headline"]


def test_a_scorer_with_no_opinion_produces_no_self_check():
    """The stub's planted rule says nothing about the card, so neither Passes nor FAILS is honest."""
    assert self_check({"sanity_check": StubScorer().sanity_check()}) is None
    assert self_check({}) is None


def test_the_report_omits_the_self_check_rather_than_reporting_a_stub_failure(rendered):
    """`passed: None` is falsy, so the obvious rendering would print FAILS for a stub."""
    _, html = rendered
    assert "Self-check" not in html and "FAILS" not in html


def test_a_failing_self_check_reaches_the_report(rendered, tmp_path):
    scan, _ = rendered
    doctored = dict(scan, sanity_check={"source": "card", "helpful_score": -4.18,
                                        "rude_score": -3.57, "margin": -0.61, "passed": False})
    html = build(doctored, tmp_path / "fail.html").read_text()
    assert "Self-check failed" in html
    assert "abusive reply 0.61 logits above a supportive one" in html


def test_a_finding_with_no_structured_detail_falls_back_to_a_trimmed_sentence():
    """Any future finding kind still gets a usable tick rather than a 160-character sentence."""
    long = ("Junk appended to a good answer costs 3.72 logits, so the model does notice "
            "irrelevant text even when it is appended rather than prepended")
    f = {"title": long, "effect": 3.0, "systematic_bar": 1.0, "valence": "healthy",
         "category": "reward_hacking", "detail": {}}
    lab = [y for y, _ in _bars(viz.findings_vs_noise([f]))][0]
    assert lab == ("Junk appended to a good answer costs 3.72 logits  "
                   "<b>Reward hacking</b>"), lab


# ---------------------------------------------------------------------------------------
# the identity group chart
# ---------------------------------------------------------------------------------------


def _id_block(means, *, null_p95, p_value):
    return {"group_means": means,
            "half_split_control": {"p95_abs_diff": 0.2},
            "within_group_control": {"p95_abs_diff": 0.6},
            "omnibus_permutation": {"statistic": max(means.values()) - min(means.values()),
                                    "null_p95": null_p95, "p_value": p_value, "n_perm": 10000}}


def _band(fig):
    """The one shaded rect: the spread reshuffling produces by chance."""
    rects = [s for s in fig.layout.shapes if s.type == "rect"]
    assert len(rects) == 1, rects
    return rects[0].x0, rects[0].x1


def test_the_chance_band_is_exactly_the_width_of_the_null_spread():
    """The band's width is the statistic's null, so "do the groups fit" *is* the test."""
    blk = _id_block({"a": 0.24, "b": 0.10, "c": -0.11, "d": -0.23}, null_p95=0.29, p_value=0.0001)
    x0, x1 = _band(viz.identity_groups(blk))
    assert x1 - x0 == pytest.approx(0.29)


def test_the_chance_band_is_centred_on_the_observed_range():
    """Off-centre, "fits inside the band" would stop matching spread <= null."""
    blk = _id_block({"a": 1.24, "b": 1.10}, null_p95=0.29, p_value=0.3)
    x0, x1 = _band(viz.identity_groups(blk))
    assert (x0 + x1) / 2 == pytest.approx((1.24 + 1.10) / 2)


def test_a_spread_beyond_chance_marks_the_two_groups_that_produced_it():
    blk = _id_block({"a": 0.24, "b": 0.10, "c": -0.11, "d": -0.23}, null_p95=0.29, p_value=0.0001)
    colors = list(viz.identity_groups(blk).data[0].marker.color)
    assert colors.count(viz.STATUS["critical"]) == 2, colors
    lo_hi = [c for v, c in sorted(zip([-0.23, -0.11, 0.10, 0.24], colors))]
    assert lo_hi[0] == lo_hi[-1] == viz.STATUS["critical"], "the extremes are the marked pair"


def test_a_spread_inside_chance_is_not_coloured_as_a_finding():
    blk = _id_block({"a": 0.24, "b": 0.10, "c": -0.11, "d": -0.23}, null_p95=0.85, p_value=0.42)
    colors = list(viz.identity_groups(blk).data[0].marker.color)
    assert viz.STATUS["critical"] not in colors, colors


def test_a_significant_p_with_a_spread_inside_the_band_is_not_flagged_red():
    """Colour follows the picture. Red dots inside the band would contradict what is drawn."""
    blk = _id_block({"a": 0.1, "b": 0.0}, null_p95=0.5, p_value=0.001)
    assert viz.STATUS["critical"] not in list(viz.identity_groups(blk).data[0].marker.color)


def test_groups_are_ordered_by_score_not_by_name():
    blk = _id_block({"zeta": 0.4, "alpha": -0.4, "mid": 0.0}, null_p95=0.2, p_value=0.01)
    fig = viz.identity_groups(blk)
    assert list(fig.layout.yaxis.categoryarray) == ["alpha", "mid", "zeta"]


def test_the_summary_and_the_appendix_name_the_yardsticks_the_same_way(rendered):
    """The summary and the appendix cover the same two measures at two depths.

    They once read as duplicates. A shared noun is what makes the appendix read as more detail
    rather than as a second copy, so it is worth pinning against a rename touching only one.
    """
    _, html = rendered
    assert "The two yardsticks every number above is measured against" in html
    # The calibration half renders only when calibration ran, which this fixture skips.
    assert "Rewording noise, in detail" in html
    assert "instruments" not in html, "one noun for these two, not two nouns"


def test_the_calibration_half_of_the_appendix_uses_the_same_naming(tmp_path):
    scan = run_scan("stub", depth="quick", calibrate=False, verbose=False, out_dir=tmp_path,
                    scorer=StubScorer(rule="length", coef=0.02))
    scan["calibration"] = {"fitted": True, "temperature": 2.5, "note": "a held-out split",
                           "accuracy": 0.63, "n_pairs": 500, "accuracy_ci": [0.59, 0.67],
                           "median_abs_gap": 1.0, "bins": []}
    html = build(scan, tmp_path / "cal.html").read_text()
    assert "Agreement with human preferences, in detail" in html


# ---------------------------------------------------------------------------------------
# the methodology write-up
# ---------------------------------------------------------------------------------------


def test_the_report_carries_every_methodology_section(rendered):
    """It lived twice, as Markdown here and HTML there, and the copies drifted apart."""
    _, html = rendered
    for lead, body in methodology.SECTIONS:
        assert f"<b>{lead}.</b>" in html, lead
        assert escape(body[:60]) in html, lead
    for lead, _ in methodology.LIMITATIONS:
        assert escape(lead) in html, lead


def test_the_method_text_describes_the_unit_the_code_actually_uses():
    """Both copies still said 'a percentile of the paraphrase noise floor' long after that went."""
    prose = " ".join(b for _, b in methodology.SECTIONS)
    assert "rewording alone could produce" in prose
    assert "percentile of the paraphrase noise floor" not in prose


def test_the_method_text_needs_no_markup_to_render_in_either_place():
    """Plain bodies are what let one source feed a Markdown app and an HTML report."""
    for lead, body in methodology.SECTIONS + methodology.LIMITATIONS:
        for markup in ("**", "<b>", "<i>", "<p>"):
            assert markup not in body, (lead, markup)


def test_the_comparison_chart_marks_the_bar_the_rest_of_the_tool_uses():
    """It plotted a percentile against a 95th-percentile line, a rule no other tab applies."""
    rows = [{"title": "t", "a": 2.0, "b": 0.5, "valence": "vulnerability"}]
    fig = viz.compare_findings(rows, "A", "B")
    lines = [s for s in fig.layout.shapes if s.type == "line"]
    assert [s.x0 for s in lines] == [1], "the threshold is one times the bar"
    assert "rewording alone" in fig.layout.xaxis.title.text
    assert "percentile" not in fig.layout.xaxis.title.text


def test_the_comparison_chart_ranks_by_the_larger_of_the_two_models():
    """Sorting on model A alone buries a finding that only fires on model B."""
    rows = [{"title": "only B", "a": 0.1, "b": 9.0, "valence": "vulnerability"},
            {"title": "only A", "a": 3.0, "b": 0.1, "valence": "vulnerability"}]
    fig = viz.compare_findings(rows, "A", "B")
    assert list(fig.data[0].y)[0] == "only B"


def test_a_model_at_chance_leads_the_verdict_with_that():
    """Every bias number below describes something already not doing its job."""
    from rmi.findings import overview_verdict
    R = {"findings": [finding_row(3.0)]}
    cal = {"fitted": True, "accuracy": 0.506, "accuracy_ci": [0.462, 0.55]}
    html = overview_verdict(R, cal)
    assert "does not beat chance" in html
    assert "50.6%" in html and "46.2%" in html


def test_a_model_above_chance_does_not_get_the_chance_verdict():
    from rmi.findings import overview_verdict
    R = {"findings": [finding_row(3.0)]}
    cal = {"fitted": True, "accuracy": 0.63, "accuracy_ci": [0.59, 0.67]}
    assert "does not beat chance" not in overview_verdict(R, cal)


def finding_row(ratio, **kw):
    f = {"category": "style", "title": "Something happens", "effect": ratio, "confirmed": True,
         "systematic_bar": 1.0, "valence": "vulnerability", "n_items": 30}
    f.update(kw)
    return f


def test_a_model_with_only_small_effects_says_so_rather_than_alarming():
    from rmi.findings import overview_verdict
    R = {"findings": [finding_row(0.5), finding_row(0.7)]}
    html = overview_verdict(R, None)
    assert "Nothing found here is large enough to matter" in html
    assert "verdict mild" in html


def test_a_model_with_nothing_confirmed_says_nothing_was_confirmed():
    from rmi.findings import overview_verdict
    R = {"findings": [finding_row(9.0, confirmed=False)]}
    html = overview_verdict(R, None)
    assert "No confirmed vulnerabilities" in html
    assert "verdict clear" in html


def test_the_verdict_ignores_findings_that_show_the_model_behaving_well():
    """A healthy finding at 11x would otherwise headline a vulnerability report."""
    from rmi.findings import overview_verdict
    R = {"findings": [finding_row(11.0, valence="healthy")]}
    assert "No confirmed vulnerabilities" in overview_verdict(R, None)
