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
    assert any("Which identities change the model score" in m.value for m in at.markdown)


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
    counted = re.search(r"(\d+) of (\d+) probes (?:show|shows) that", line)
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
    assert any(h.startswith("##### Read the text sent to model, for ")
               for h in headings), headings
    assert any("when the user" in h for h in headings), headings


@pytest.mark.parametrize("written", [
    # A brand-new file that sorts first, so the picker's default of the last entry is not it.
    "aaa__newest__quick__seed0.json",
    # Re-scanning a run that is already listed, but not the one on screen: the picker's options do
    # not change, so it keeps its old selection.
    "stub__quick__seed0.json",
])
def test_a_finished_scan_is_what_the_dashboard_shows(dashboard, tmp_path, written):
    """Finishing a scan reruns the app, and the picker then still named the run on screen before
    the scan. The loader saw the picker disagree with the fresh result and reloaded the picked file
    over it, so the new scan was never shown."""
    from streamlit.testing.v1 import AppTest

    scan, _ = dashboard
    src = Path(scan["meta"]["results_path"])
    (tmp_path / src.name).write_bytes(src.read_bytes())
    (tmp_path / "zzz__other__quick__seed0.json").write_bytes(src.read_bytes())

    def fake_run_scan(model_id, **_):
        path = tmp_path / written
        path.write_bytes(src.read_bytes())
        res = runner.load_results(path)
        res["meta"]["results_path"] = str(path)
        return res

    original = runner.RESULTS_DIR, runner.run_scan
    runner.RESULTS_DIR, runner.run_scan = tmp_path, fake_run_scan
    try:
        at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"),
                               default_timeout=500)
        at.run()
        assert at.session_state["results_name"] == "zzz__other__quick__seed0.json"
        next(b for b in at.sidebar.button if b.label == "Run scan").click()
        at.run()
    finally:
        runner.RESULTS_DIR, runner.run_scan = original
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["results_name"] == written
    picker = next(s for s in at.sidebar.selectbox if s.label == "Results file")
    assert picker.value == written.replace("__", " · ").replace(".json", "")


def test_comparing_runs_on_different_attack_grids_warns(dashboard, tmp_path):
    """The reduced grid reports smaller lifts and higher success rates for the same model. Side by
    side with a full-grid run that reads as a difference between models unless something says so."""
    import json

    from streamlit.testing.v1 import AppTest

    from rmi.report import build

    scan, _ = dashboard
    src = Path(scan["meta"]["results_path"])
    assert scan["reward_hacking"]["grid"] == "reduced"
    (tmp_path / "b_reduced__quick__seed0.json").write_bytes(src.read_bytes())
    old = json.loads(src.read_text())
    del old["reward_hacking"]["grid"]  # written before the reduced grid existed
    (tmp_path / "a_full__quick__seed0.json").write_text(json.dumps(old))

    original = runner.RESULTS_DIR
    runner.RESULTS_DIR = tmp_path
    try:
        at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"),
                               default_timeout=500)
        at.run()
    finally:
        runner.RESULTS_DIR = original
    assert not at.exception, [e.value for e in at.exception]
    warnings = [w.value for w in at.warning if "probed reward hacking differently" in w.value]
    assert warnings and "reduced attack grid" in warnings[0] and "full attack grid" in warnings[0]

    out = tmp_path / "report.html"
    build(runner.load_results(tmp_path / "b_reduced__quick__seed0.json"), out)
    page = out.read_text()
    assert "reduced attack grid" in page
    assert "both prefix and suffix" not in page


def test_the_dashboard_has_no_depth_control_and_scans_at_quick(dashboard, tmp_path):
    """Depth is fixed like the seed. A slider left in place would silently run a different preset
    from the one the app claims."""
    from streamlit.testing.v1 import AppTest

    scan, _ = dashboard
    src = Path(scan["meta"]["results_path"])
    (tmp_path / src.name).write_bytes(src.read_bytes())
    asked = {}

    def fake_run_scan(model_id, **kwargs):
        asked.update(kwargs)
        res = runner.load_results(tmp_path / src.name)
        res["meta"]["results_path"] = str(tmp_path / src.name)
        return res

    original = runner.RESULTS_DIR, runner.run_scan
    runner.RESULTS_DIR, runner.run_scan = tmp_path, fake_run_scan
    try:
        at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"),
                               default_timeout=500)
        at.run()
        assert not [s for s in at.sidebar.select_slider if s.label == "Depth"]
        next(b for b in at.sidebar.button if b.label == "Run scan").click()
        at.run()
    finally:
        runner.RESULTS_DIR, runner.run_scan = original
    assert not at.exception, [e.value for e in at.exception]
    assert asked["depth"] == "quick"


def test_reward_hacking_is_opt_in_and_a_scan_without_it_renders(tmp_path):
    """It scores more texts than every other probe together, so on a CPU host it starts unticked
    and says what it costs. A scan that skipped it must still render in both places."""
    from streamlit.testing.v1 import AppTest

    from rmi.report import build

    scan = runner.run_scan("stub", depth="quick", probes=("identity", "sycophancy", "style"),
                           calibrate=False, verbose=False, out_dir=tmp_path,
                           scorer=StubScorer(rule="length", coef=0.02))
    assert "reward_hacking" not in scan
    original = runner.RESULTS_DIR
    runner.RESULTS_DIR = tmp_path
    try:
        at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"),
                               default_timeout=500)
        at.run()
    finally:
        runner.RESULTS_DIR = original
    assert not at.exception, [e.value for e in at.exception]
    boxes = {c.label: c.value for c in at.sidebar.checkbox}
    assert boxes["Reward hacking"] is False
    assert all(boxes[label] for label in ("Identity", "Sycophancy", "Style and length"))
    assert any("Adds about 5–10 min" in c.value for c in at.sidebar.caption)

    out = tmp_path / "report.html"
    build(runner.load_results(Path(scan["meta"]["results_path"])), out)
    assert out.stat().st_size > 0
