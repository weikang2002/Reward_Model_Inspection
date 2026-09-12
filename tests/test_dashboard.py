"""The dashboard, run headlessly against a stub scan.

`app.py` is half the product and had no automated cover at all: every regression in it was caught
by eye, or not at all. This is deliberately one broad test rather than many narrow ones, because
the cost is the app run, not the assertions.

It points the app at a temporary results directory so it never depends on what the developer
happens to have scanned. `app.py` re-executes on every AppTest run, so its `from rmi.runner import
RESULTS_DIR` picks up the patched value.
"""

import re
from pathlib import Path

import pytest

from rmi import runner, severity as sv
from rmi.findings import module_items, verdict_line
from rmi.scoring import StubScorer


@pytest.fixture(scope="module")
def dashboard(tmp_path_factory):
    from streamlit.testing.v1 import AppTest

    out = tmp_path_factory.mktemp("dash_results")
    scan = runner.run_scan("stub", depth="quick", calibrate=False, verbose=False, out_dir=out,
                           scorer=StubScorer(rule="length", coef=0.02))
    original, runner.RESULTS_DIR = runner.RESULTS_DIR, out
    try:
        # Relative paths resolve against this file, not the repo root.
        at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"),
                               default_timeout=500)
        at.run()
    finally:
        runner.RESULTS_DIR = original
    return scan, at


def test_the_dashboard_renders_a_saved_scan_without_error(dashboard):
    _, at = dashboard
    assert not at.exception, [e.value for e in at.exception]
    # It reached the tabs rather than stopping on the landing page.
    assert any("Which identities change the score" in m.value for m in at.markdown)


def test_the_export_button_is_live_on_the_first_load(dashboard):
    """The picked run is only loaded into session state after the sidebar has rendered, so keying
    the button off session state alone left it dim until the first rerun of a session."""
    _, at = dashboard
    export = next(b for b in at.sidebar.button if "Build standalone" in b.label)
    assert not export.disabled


def test_the_sidebar_leads_with_the_saved_runs(dashboard):
    """Opening a past scan is the common case; running one takes minutes and happens once."""
    _, at = dashboard
    headings = [m.value for m in at.sidebar.markdown if m.value.startswith("###")]
    assert headings[0].endswith("Open a past model scan to see results")
    assert "Run a model scan" in headings[1]


@pytest.mark.parametrize("category,phrase", [
    ("identity", "swapping a name or a descriptor changes the score"),
    ("sycophancy", "the model is rewarded for agreeing rather than correcting"),
    ("style", "surface style is rewarded for its own sake"),
])
def test_every_tab_verdict_states_its_own_tile_s_count(dashboard, category, phrase):
    """The overview tile and the tab that explains it must not answer differently. Style is the
    one that came apart: its tile counts the substance check, which the tab charts nowhere."""
    scan, at = dashboard
    items = (module_items(scan, "style", group="content_neutral")
             + module_items(scan, "style", group="quality_control")
             if category == "style" else module_items(scan, category))
    line = verdict_line(items, category, phrase)
    tile = sv.summarise(scan["findings"], category)
    # Only the material branch makes the tile's claim. "N of M are still statistically
    # real" below it counts confirmed-of-adverse, which is a different sentence.
    counted = re.search(r"<b>(\d+) of (\d+)</b> probes", line)
    if counted:
        assert (int(counted.group(1)), int(counted.group(2))) == (tile["n_material"],
                                                                  tile["n_vulnerabilities"])
    # Whatever it says, the app renders this exact line, so the two cannot drift.
    body = re.sub("<[^>]+>", "", line)[:60]
    assert any(body in re.sub("<[^>]+>", "", m.value) for m in at.markdown), body


def test_a_drill_down_names_the_selection_it_follows(dashboard):
    """Each drill-down inherits a selection from the panel above it. Without the selection in the
    heading it read as an independent section and nothing said which choice it was showing."""
    _, at = dashboard
    headings = [m.value for m in at.markdown if m.value.startswith("#####")]
    assert any(h.startswith("##### Read the exact text for ") for h in headings), headings
    assert any("when the user" in h for h in headings), headings
