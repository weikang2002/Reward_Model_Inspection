"""Tests for the presentation layer: the diff, the charts, and the standalone report.

These were the largest untested surface. None of it raises on a wrong number, so the failure mode
is a chart that quietly plots the wrong field, or a report that ships missing a section.
"""

import re

import pytest

from rmi import viz
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


def test_findings_vs_noise_ranks_by_ratio_and_keeps_the_largest():
    findings = [{"title": f"f{i}", "effect": float(i), "systematic_bar": 1.0,
                 "valence": "vulnerability"} for i in range(1, 25)]
    fig = viz.findings_vs_noise(findings)
    xs = [t.x[0] for t in fig.data]
    assert max(xs) == 24.0, "the largest finding must survive the cap"
    assert len(xs) == 16, "the chart caps how many it shows"


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
