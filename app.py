"""Reward Model Inspection dashboard.

    streamlit run app.py

Loads any HuggingFace AutoModelForSequenceClassification reward model, probes it for identity
bias, sycophancy, style and length preferences, and prefix/suffix injection attacks, then shows
what it found. Past runs load from results/*.json without needing the model in memory.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from rmi import severity as sev
from rmi.textdiff import word_diff
from rmi import viz
from rmi.findings import (module_items, overview_verdict, self_check, tone_check,
                          verdict_line)
from rmi.report import build as build_report
from rmi.scoring import ModelTooLargeError, RewardModel, check_download_size
from rmi.runner import (PROBE_GROUPS, PROBE_LABELS, PROBES, PRESETS, RESULTS_DIR,
                        list_results, load_results, probe_heading, run_scan)


@st.cache_data(show_spinner=False, ttl=3600)
def preflight_download(model_id: str):
    """Bytes a first scan would download, or None if already cached. Raises if over the limit."""
    return check_download_size(model_id)


@st.cache_data(show_spinner="Loading results")
def cached_results(path: str, mtime: float) -> dict:
    """Keyed on mtime so a re-run of the same file invalidates the cache."""
    return load_results(path)

st.set_page_config(page_title="Reward Model Inspection", page_icon="🔍", layout="wide")

PRESET_MODELS = [
    "OpenAssistant/reward-model-deberta-v3-large-v2",
    "OpenAssistant/reward-model-deberta-v3-base",
]

# Fixed rather than offered as a control. The seed changes only resampling draws and the injection
# dev/test split, so a box for it mostly invites re-rolling until a borderline finding turns
# significant, which is the one habit this project's statistics exist to prevent. A genuinely
# different split is still available headlessly via run_scan(seed=...).
SEED = 0

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1500px;}
  .tile {border:1px solid #e6e5e1; border-radius:10px; padding:14px 16px; background:#fcfcfb;
         height:100%;}
  .tile h4 {margin:0 0 6px 0; font-size:13px; letter-spacing:.04em; text-transform:uppercase;
            color:#52514e; font-weight:600;}
  .band {display:inline-block; padding:3px 10px; border-radius:999px; color:#fff;
         font-weight:600; font-size:13px;}
  .tile .sub {color:#52514e; font-size:12.5px; margin-top:8px; line-height:1.45;}
  .yard {border-left:3px solid #2a78d6; padding:10px 14px; background:#f4f8fe;
         border-radius:0 8px 8px 0; margin-bottom:6px;}
  .yard .big {font-size:23px; font-weight:700; display:block; margin-bottom:3px;}
  .yard {font-size:13px; line-height:1.5; color:#3c3b38;}
  .finding {border:1px solid #e6e5e1; border-left-width:4px; border-radius:8px;
            padding:10px 14px; margin-bottom:8px; background:#fcfcfb;}
  .txtbox {background:#f7f7f5; border:1px solid #e6e5e1; border-radius:8px; padding:10px 12px;
           font-size:13px; line-height:1.5; white-space:pre-wrap;}
  .verdict {border-left:4px solid #d03b3b; background:#fdf3f3; border-radius:0 8px 8px 0;
            padding:12px 16px; margin:2px 0 18px; font-size:15px; line-height:1.55;}
  .verdict.clear {border-left-color:#0ca30c; background:#f2faf2;}
  .verdict.mild {border-left-color:#fab219; background:#fdf6e8;}
  .verdict .hint {display:block; color:#52514e; font-size:12.5px; margin-top:6px;}
  ins {background:#d7f0d7; text-decoration:none;}
  ins.attack {background:#fbe3e0; border-bottom:2px solid #d03b3b; text-decoration:none;}
  .grouphead {font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:#52514e;
              font-weight:700; margin:-2px 0 2px;}
  /* Tighten the gap between the checkboxes only; a negative margin on the box itself pushed the
     last one through the bottom border. */
  [class*="st-key-probegroup"] {border-radius:9px !important;}
  [class*="st-key-probegroup"] [data-testid="stVerticalBlock"] {gap:.25rem !important;}
  [class*="st-key-probegroup_bias"] {border-left:4px solid #2a78d6 !important;}
  [class*="st-key-probegroup_hack"] {border-left:4px solid #d03b3b !important;}
  /* Same boxes for the two auxiliary blocks, left uncoloured: they are housekeeping, and a
     coloured edge would read as another probe category. */
  [class*="st-key-aux_"] {border-radius:9px !important; margin-bottom:.5rem;}
  [class*="st-key-drill"] details {border:1px solid #2a78d6 !important;
     border-left-width:5px !important; background:#ffffff !important; border-radius:9px;}
  [class*="st-key-drill"] details > summary {background:#ffffff !important;}
  [class*="st-key-drill"] summary {font-weight:600 !important; font-size:15px !important;}
  [class*="st-key-drill"] summary p {font-weight:600 !important; font-size:15px !important;}
  del {background:#fbdada; text-decoration:line-through;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# clearing saved results
# --------------------------------------------------------------------------------------

# A scan JSON runs to ~10 MB and its exported report to ~4.5 MB, and both go stale as soon as the
# analysis code changes, which is why they are worth being able to throw away from the UI. The
# score cache is deliberately not part of this; see the caption in the sidebar.
CLEARABLE = (".json", ".html")


def clearable_results(d: Path) -> list[Path]:
    """The files a clear would remove: plain files sitting directly in the results directory."""
    if not d.is_dir():
        return []
    return sorted(f for f in d.iterdir()
                  if f.is_file() and not f.is_symlink() and f.suffix in CLEARABLE)


def clear_results(d: Path) -> tuple[int, list[str]]:
    """Delete those files, returning how many went and the name of any that would not.

    Each path is resolved and checked against the resolved directory again immediately before the
    unlink, so no symlink or oddly built name can reach outside it, and subdirectories are never
    descended into. A file that refuses to go is reported rather than raised, because one
    permission error should not take the dashboard down with it.
    """
    root = d.resolve()
    removed, failed = 0, []
    for f in clearable_results(d):
        try:
            target = f.resolve()
            if target.parent != root or target.suffix not in CLEARABLE:
                failed.append(f"{f.name} (points outside {root.name}/)")
                continue
            target.unlink()
            removed += 1
        except OSError as exc:
            failed.append(f"{f.name} ({exc.strerror or exc})")
    return removed, failed


# --------------------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Run a scan")
    model_choice = st.selectbox("Reward model", PRESET_MODELS + ["Other (type below)"])
    model_id = (st.text_input("HuggingFace model id", value="")
                if model_choice.startswith("Other") else model_choice)

    # Checked here as well as in RewardModel, so an oversize model is refused before the user
    # commits to a scan rather than as an exception a minute later.
    too_large = None
    if model_id:
        try:
            pending = preflight_download(model_id)
        except ModelTooLargeError as exc:
            too_large = str(exc)
            st.error(too_large, icon=":material/block:")
        else:
            if pending:
                st.info(f"First run will download {pending / 1024**3:.1f} GB.",
                        icon=":material/download:")

    st.markdown("**Probe for**")
    chosen = []
    for slug, (group_title, members) in PROBE_GROUPS.items():
        with st.container(border=True, key=f"probegroup_{slug}"):
            st.markdown(f'<div class="grouphead">{group_title}</div>', unsafe_allow_html=True)
            for probe in members:
                if st.checkbox(PROBE_LABELS[probe], value=True, key=f"probe_{probe}"):
                    chosen.append(probe)
    calibrate = st.checkbox("Measure agreement with humans", value=True)
    st.caption(
        "Every scan also measures the paraphrase noise floor, the yardstick all findings are "
        "reported against, so it is not optional. The human-preference check needs a one-off "
        "dataset download."
    )
    depth = st.select_slider("Depth", options=list(PRESETS), value="standard",
                             format_func=str.capitalize)
    st.caption(
        "Quick skips the search for stacked attacks entirely, so it cannot find one. Deep searches "
        "wider and longer. Scores are cached, so re-runs are near-instant."
    )
    run_clicked = st.button("Run scan", type="primary", width="stretch",
                            disabled=not model_id or too_large is not None)

    st.divider()
    st.markdown("### Open a past run")
    files = list_results()
    labels = [f.name.replace("__", " · ").replace(".json", "") for f in files]
    picked = st.selectbox("Results file", labels, index=len(labels) - 1 if labels else None) \
        if files else None

    # Export and maintenance are one section: neither probes anything, both act on the files in
    # results/ rather than on a model, and grouping them keeps the two things that do run a scan
    # or read one at the top of the sidebar.
    st.divider()
    st.markdown("### Saved files")

    with st.container(border=True, key="aux_export"):
        st.markdown('<div class="grouphead">Export</div>', unsafe_allow_html=True)
        export_clicked = st.button("Build standalone HTML report", width="stretch",
                                   disabled="results" not in st.session_state)
        st.caption("One self-contained file with every chart inlined. Opens in any browser with "
                   "no Python and no network.")

    with st.container(border=True, key="aux_clear"):
        st.markdown('<div class="grouphead">Maintenance</div>', unsafe_allow_html=True)
        if done := st.session_state.pop("clear_done", None):
            st.success(done)
        stale = clearable_results(RESULTS_DIR)
        size_mb = sum(f.stat().st_size for f in stale) / 1e6
        if not stale:
            st.button("Clear saved results", width="stretch", disabled=True)
            st.caption("No saved results.")
        elif not st.session_state.get("confirm_clear"):
            # Armed first, deleted second: a single stray click would otherwise cost a scan that
            # takes minutes of GPU time to reproduce.
            if st.button("Clear saved results", width="stretch"):
                st.session_state["confirm_clear"] = True
                st.rerun()
            st.caption(f"{len(stale)} files · {size_mb:.1f} MB of results JSON and exported "
                       "reports.")
        else:
            st.warning(f"Delete {len(stale)} files, {size_mb:.1f} MB? This cannot be undone.")
            c1, c2 = st.columns(2)
            if c1.button("Delete permanently", type="primary", width="stretch"):
                removed, failed = clear_results(RESULTS_DIR)
                st.session_state.pop("confirm_clear", None)
                st.session_state.pop("results", None)
                st.session_state.pop("results_name", None)
                if failed:
                    st.error(f"Removed {removed} files. Could not remove " + ", ".join(failed))
                else:
                    # Rerun so the app falls back to its landing page rather than rendering a run
                    # whose file has just been deleted, and so the picker re-reads the directory.
                    st.session_state["clear_done"] = f"Removed {removed} files."
                    st.rerun()
            if c2.button("Cancel", width="stretch"):
                st.session_state.pop("confirm_clear", None)
                st.rerun()
        st.caption(
            "The score cache in `.rmi_cache/` is left alone. Scores are deterministic and keyed "
            "by model id, revision and text, so a cached score is never stale, only expensive to "
            "rebuild. Results JSON does go stale when the analysis code changes."
        )

if run_clicked:
    bar = st.progress(0.0, text="starting")

    def cb(frac, label):
        bar.progress(min(frac, 1.0), text=label)

    try:
        with st.spinner(f"Scanning {model_id}"):
            res = run_scan(model_id, depth=depth, probes=tuple(chosen), calibrate=calibrate,
                           seed=SEED, verbose=False, progress_cb=cb)
    except ModelTooLargeError as exc:
        bar.empty()
        st.error(str(exc), icon=":material/block:")
        st.stop()
    bar.empty()
    st.session_state["results"] = res
    st.session_state["results_name"] = Path(res["meta"]["results_path"]).name
    st.session_state["results_path"] = res["meta"]["results_path"]
    st.rerun()

R = st.session_state.get("results")
if picked and (R is None or st.session_state.get("results_name") != files[labels.index(picked)].name):
    if not run_clicked:
        _f = files[labels.index(picked)]
        R = cached_results(str(_f), _f.stat().st_mtime)
        st.session_state["results_path"] = str(_f)
        st.session_state["results"] = R
        st.session_state["results_name"] = files[labels.index(picked)].name

if R is None:
    st.title("Reward Model Inspection")
    st.markdown(
        "Reward models decide what a policy learns during RLHF. Whatever the reward model scores "
        "highly is what the trained model drifts toward, so a bias or an exploitable shortcut in "
        "the reward model is copied into the policy and amplified.\n\n"
        "Pick a model in the sidebar and run a scan."
    )
    st.stop()


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def labelled(options, labels: dict, *, sep: str = " — ", limit: int = 70):
    """Format a dropdown option as "id — short description".

    A bare id like t01 or q07 tells the reader nothing about what they are about to open.
    Returns the sorted options and the formatter, so both stay in step.
    """
    def fmt(key):
        text = str(labels.get(key, "")).strip()
        if len(text) > limit:
            text = text[: limit - 1].rstrip() + "…"
        return f"{key}{sep}{text}" if text else str(key)

    return sorted(options), fmt


def compare_verdict(A: dict, B: dict, a_name: str, b_name: str,
                    a_sev: dict, b_sev: dict) -> str:
    """What separates the two checkpoints, in one sentence.

    Agreement with human preferences leads. A model that does not track human judgment is not a
    milder version of one that does, and comparing their bias profiles first would bury that.
    """
    ca, cb = (A.get("calibration") or {}), (B.get("calibration") or {})
    acc_a = ca.get("accuracy") if ca.get("fitted") else None
    acc_b = cb.get("accuracy") if cb.get("fitted") else None
    chance = [n for n, c in ((a_name, ca), (b_name, cb))
              if c.get("fitted") and c["accuracy_ci"][0] <= 0.5]
    order = {"High": 3, "Moderate": 2, "Low": 1}
    worse = [c for c in a_sev
             if order.get(a_sev[c]["band"], 0) != order.get(b_sev.get(c, {}).get("band"), 0)]

    if chance:
        rest = [n for n in (a_name, b_name) if n not in chance]
        lead = (f'<b>{_esc_join(chance)} does not beat chance on human preferences.</b> ')
        if rest and acc_a is not None and acc_b is not None:
            better = a_name if (acc_a or 0) > (acc_b or 0) else b_name
            acc = max(acc_a, acc_b)
            lead += (f'{html.escape(better)} manages {acc:.1%}. That difference outranks any bias '
                     'comparison below: a model that is not tracking human judgment is not a '
                     'milder version of one that is.')
        return f'<div class="verdict">{lead}</div>'

    if acc_a is not None and acc_b is not None:
        better = a_name if acc_a > acc_b else b_name
        gap = abs(acc_a - acc_b)
        tail = (f' They are rated differently on {len(worse)} of {len(a_sev)} categories.'
                if worse else ' They land in the same severity band on every category.')
        return (f'<div class="verdict mild"><b>{html.escape(better)} agrees with human '
                f'preferences more often</b>, {max(acc_a, acc_b):.1%} against '
                f'{min(acc_a, acc_b):.1%}, a gap of {gap:.1%}.{tail}</div>')

    return ('<div class="verdict mild"><b>Neither run measured agreement with human '
            'preferences.</b> Re-run with that check enabled to compare the two on the number '
            'that matters most.</div>')


def _esc_join(names: list[str]) -> str:
    return " and ".join(html.escape(n) for n in names)


def severity_tiles(sev_map: dict, *, other: dict | None = None,
                   names: tuple[str, str] | None = None) -> None:
    """Severity per category, laid out in the same two groups as the sidebar and the tab bar.

    Equal column counts per group keep the tiles the same width, which matters more than filling
    the row: a lone double-width tile would read as more important rather than as a second kind.
    """
    for slug, (group_title, members) in PROBE_GROUPS.items():
        st.markdown(f'<div class="grouphead">{group_title}</div>', unsafe_allow_html=True)
        cols = st.columns(max(len(m) for _, m in PROBE_GROUPS.values()))
        for col, cat in zip(cols, members):
            sm = sev_map.get(cat)
            if not sm:
                continue
            with col:
                if other is None:
                    worst = (f"{sm['severity']:.1f}x" if sm.get("severity") is not None
                             else "n/a")
                    body = (f"<b>{sm.get('n_material', 0)} of "
                            f"{sm.get('n_vulnerabilities', 0)}</b> possible problems here are big "
                            f"enough to matter.<br>Worst confirmed effect: <b>{worst}</b> what "
                            "rewording alone could produce."
                            if sm.get("severity") is not None else
                            f"{sm.get('n_vulnerabilities', 0)} probes ran; none reached "
                            "significance.")
                    st.markdown(
                        f'<div class="tile"><h4>{cat}</h4>{band_pill(sm["band"])}'
                        f'<div class="sub">{body}</div></div>', unsafe_allow_html=True)
                else:
                    om = other.get(cat, {})
                    a, b = names or ("A", "B")
                    st.markdown(
                        f'<div class="tile"><h4>{cat}</h4>'
                        f'<div class="sub" style="margin:0 0 4px"><b>{html.escape(a)}</b></div>'
                        f'{band_pill(sm["band"])}'
                        f'<div class="sub" style="margin:9px 0 4px"><b>{html.escape(b)}</b></div>'
                        f'{band_pill(om.get("band", "n/a"))}</div>', unsafe_allow_html=True)


def band_pill(band: str) -> str:
    return (f'<span class="band" style="background:{viz.BAND_COLOR.get(band, "#8a8a85")}">'
            f'{html.escape(band)}</span>')


def side_by_side(title_a, text_a, score_a, title_b, text_b, score_b, *, diff=True,
                 ins_class: str = ""):
    la, lb = (word_diff(text_a, text_b, ins_class=ins_class) if diff
              else (html.escape(text_a), html.escape(text_b)))
    c1, c2 = st.columns(2)
    for col, t, body, s in ((c1, title_a, la, score_a), (c2, title_b, lb, score_b)):
        with col:
            st.markdown(f"**{t}** · score `{s:+.3f}`")
            st.markdown(f'<div class="txtbox">{body}</div>', unsafe_allow_html=True)


def _length_flip(sm: dict):
    """A transform whose sign reverses once length is controlled for, if there is one.

    Content-free transforms are preferred: those are the flips that change a *bias* verdict.
    Elaboration turning positive only means the model likes genuine information, which is not a
    finding about style.
    """
    raw = {}
    for c in sm["contrasts"]:
        if c["transform"] not in raw or c["dose"] > raw[c["transform"]]["dose"]:
            raw[c["transform"]] = c
    best = None
    for a in sm["adjusted"]:
        t = a["transform"]
        r, adj = raw.get(t, {}).get("mean_delta"), a.get("effect_at_max_dose")
        if r is None or adj is None or r * adj >= 0:
            continue
        rank = (a["group"] == "content_neutral", abs(adj))
        if best is None or rank > best[0]:
            best = (rank, t, r, adj)
    return best[1:] if best else None


noise = R.get("noise_floor")
cal = R.get("calibration")
# Derived on the fly so old result files render under the current thresholds.
severity = sev.summarise_all(R["findings"])
meta = R["meta"]

st.title("Reward Model Inspection")
st.caption(
    f"{meta['model_id']} · {meta['depth']} scan · seed {meta['seed']} · "
    f"{meta.get('runtime_seconds', 0):.0f}s · "
    f"{meta.get('provenance', {}).get('device', '?')} · {meta['started']}"
)

if export_clicked and R is not None:
    out = Path(st.session_state.get("results_path")
               or meta.get("results_path", "results/run.json")).with_suffix(".html")
    with st.spinner("Building report"):
        build_report(R, out)
    with st.sidebar:
        st.success(f"Wrote {out.name}")
        st.download_button("Download report", out.read_bytes(), file_name=out.name,
                           mime="text/html", width="stretch")

# Tab names carry the same grouping as the sidebar: the first three are bias probes, the fourth
# asks a different question entirely. Derived from PROBE_GROUPS so the two cannot drift apart.
tabs = st.tabs(["Overview"] + [probe_heading(p) for p in PROBES]
               + ["Compare models", "Appendix"])


def finding_key(f: dict) -> tuple:
    """Identifies the same probe across two runs, so findings can be lined up."""
    d = f.get("detail", {})
    return (d.get("module"), d.get("kind"), d.get("transform") or d.get("axis")
            or d.get("level") or d.get("subset") or d.get("where") or d.get("affix_id"))


# --------------------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------------------

with tabs[0]:
    st.markdown(overview_verdict(R, cal), unsafe_allow_html=True)

    # ---- where ------------------------------------------------------------------------------
    st.markdown("#### Where the problems are")
    severity_tiles(severity)
    st.caption(
        "Bands are the worst confirmed effect in a category, as a multiple of what rewording "
        "alone could produce across the same number of comparisons: below 1x Negligible, to 2x "
        "Low, to 4x Moderate, above that High. Findings showing the model behaving *well* are "
        "excluded. Open a category's tab for the evidence behind it."
    )

    # ---- everything, on one scale -----------------------------------------------------------
    if noise:
        st.markdown("#### Every finding, on one scale")
        st.plotly_chart(viz.findings_vs_noise(R["findings"]), width="stretch",
                        config={"displayModeBar": False})
        scored = [f for f in R["findings"]
                  if f.get("effect") is not None and f.get("systematic_bar")]
        shown = min(len(scored), viz.MAX_ROWS)
        st.caption(
            f"The {shown} largest of {len(scored)} findings, each labelled with the category it "
            "came from. Bars inside the shaded band are no larger than rewording alone would "
            "produce at the same sample size, so they are real but too small to steer a policy. "
            "Hover a bar for the effect in logits."
        )

    # ---- what the numbers above are measured against ------------------------------------
    st.markdown("#### The two yardsticks every number above is measured against")
    y1, y2 = st.columns(2)
    with y1:
        st.metric("Rewording noise", f"±{noise['pairwise_sd']:.2f} logits" if noise else "n/a",
                  help="Spread between two meaning-preserving rewrites of the same answer. "
                       "Measured rewrite against rewrite, so neither side is privileged.")
        if noise:
            st.caption(
                f"A single comparison moves up to {noise['per_comparison_bar']:.2f} on rewording "
                "alone, which swamps every bias effect here. But rewording pushes in an "
                "arbitrary direction, so it mostly cancels when averaged over many answers, "
                "while a bias does not. Every finding above is therefore measured against this "
                "spread shrunk to its own sample size."
            )
    with y2:
        if cal and cal.get("fitted"):
            st.metric("Agrees with human preferences", f"{cal['accuracy']:.1%}",
                      help="Measured on held-out Anthropic/hh-rlhf pairs. Chance is 50%.")
            st.caption(f"{cal['n_pairs']} held-out pairs, 95% CI {cal['accuracy_ci'][0]:.1%} to "
                       f"{cal['accuracy_ci'][1]:.1%}. Chance is 50%.")
        else:
            st.metric("Agrees with human preferences", "not measured")
            st.caption("Re-run with the human-preference check enabled to fill this in.")
    st.caption(
        "The Appendix has the distribution behind each of these, and the controls that check the "
        "yardstick itself."
    )

    # Below the yardsticks rather than beside them: nothing here is measured against it. See
    # findings.self_check for why it is on the front page at all.
    if check := self_check(R):
        box = st.success if check["passed"] else st.error
        box(check["headline"], icon=":material/check_circle:" if check["passed"]
            else ":material/report:")
        st.caption(check["detail"])

    with st.expander("Read every finding in words"):
        valence = st.radio("Show", ["vulnerability", "healthy", "informational", "all"],
                           horizontal=True, label_visibility="collapsed")
        for f in R["findings"]:
            if valence != "all" and f["valence"] != valence:
                continue
            color = viz.VALENCE_COLOR.get(f["valence"], "#8a8a85")
            bits = []
            if f.get("noise_percentile") is not None:
                bits.append(f"bigger than {f['noise_percentile']:.0f}% of rewordings")
            if f.get("preference_probability"):
                bits.append(f"shifts the model's preference to {f['preference_probability']:.0%}")
            if f.get("p_adjusted") is not None:
                bits.append(f"adjusted p = {f['p_adjusted']:.3g}")
            elif f.get("p_raw") is not None:
                bits.append(f"p = {f['p_raw']:.3g}")
            st.markdown(
                f'<div class="finding" style="border-left-color:{color}">'
                f'{band_pill(f["band"])} &nbsp;<b>{html.escape(f["title"])}</b>'
                f'<div class="sub" style="color:#52514e; font-size:12.5px; margin-top:5px">'
                f'{" · ".join(bits)}</div></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------------------

with tabs[1]:
    idr = R.get("identity")
    if not idr or not noise:
        st.info("The identity probe was not part of this run.")
    else:
        items = module_items(R, "identity")
        st.markdown(verdict_line(items, "identity",
                                 "swapping a name or a descriptor changes the score"), unsafe_allow_html=True)
        st.markdown("#### Which identities change the score")
        st.plotly_chart(viz.bias_bars(items), width="stretch",
                        config={"displayModeBar": False})
        st.caption(
            "Each bar is one identity axis: the largest gap it produces between groups, in "
            "otherwise identical templates, relative to what rewording alone could produce "
            "across the same number of templates. Identity bias is non-directional, so a shift either "
            "way counts and only the size is shown. Grey bars did not reach significance."
        )

        st.markdown("#### Inspect one axis")
        choices = {i["label"]: i for i in items}
        pick = st.selectbox("Which axis", list(choices),
                            index=list(choices).index(max(choices, key=lambda k: abs(choices[k]["effect"]))))
        chosen = choices[pick]
        kind = chosen["detail"].get("kind")

        if kind == "omnibus":
            blk = idr[f"names_{chosen['detail']['subset']}"]
            o = blk["omnibus_permutation"]
            st.plotly_chart(viz.identity_groups(blk), width="stretch",
                            config={"displayModeBar": False})
            hs = blk.get("half_split_control", {})
            order = sorted(blk["group_means"], key=blk["group_means"].get, reverse=True)
            fits = o["statistic"] <= o["null_p95"]
            st.markdown(
                f"Everything else in the template is identical. Only the name changes, and the "
                f"model still ranks them **{' > '.join(k.replace('_', ' ') for k in order)}**. "
                f"Top to bottom that is `{o['statistic']:.2f}` logits.\n\n"
                f"The grey band is how wide that spread gets when the names are reshuffled "
                f"between groups at random: `{o['null_p95']:.2f}` at most, 95% of the time. The "
                + (f"groups **fit inside it**, so this ordering is the kind of thing name choice "
                   f"produces on its own (p = {o['p_value']:.3g})."
                   if fits else
                   f"groups **do not fit inside it**. A spread this wide comes up in only "
                   f"**{o['p_value'] * 100:.2f}%** of reshuffles, so the ordering is not chance.")
                + (f" A second control agrees: two random halves of the *same* group land at most "
                   f"`{hs['p95_abs_diff']:.2f}` apart, which is this comparison run with no "
                   "identity difference in it at all." if hs.get("p95_abs_diff") else "")
            )
        else:
            e = idr["descriptors"][chosen["detail"]["axis"]]
            st.plotly_chart(viz.descriptor_chart(e), width="stretch",
                            config={"displayModeBar": False})
            st.markdown(
                f"**{html.escape(e['highest'])}** scores highest and "
                f"**{html.escape(e['lowest'])}** lowest, a gap of `{e['max_gap']:.2f}` logits. "
                f"Shuffling the descriptors within each template gives a gap that large "
                f"**{e['permutation']['p_value'] * 100:.2f}%** of the time. Based on "
                f"{e['n_templates']} templates, so treat it as indicative."
            )

        # Name swaps and descriptor swaps are scored in two separate sets of templates, so the
        # drill-down has to follow whichever axis is selected above.
        if kind == "omnibus":
            subset = chosen["detail"]["subset"]
            rows = [a for a in idr.get("arms", []) if a["distribution"] == subset]
            varies, extra_cols = "name", ["group"]
            what = "name"
        else:
            axis = chosen["detail"]["axis"]
            rows = [a for a in idr.get("descriptor_arms", []) if a["axis"] == axis]
            varies, extra_cols = "level", []
            what = f"{axis} descriptor"

        st.markdown("#### Read the exact text")
        with st.expander(f"See the scored text, swapping the {what}",
                         icon=":material/article:", expanded=True, key="drill_identity"):
            arms = pd.DataFrame(rows)
            if arms.empty:
                st.info("No scored text was stored for this axis.")
            else:
                domains = {r["template_id"]: r["domain"].replace("_", " ")
                           for r in arms.to_dict("records")}
                opts, fmt = labelled(arms["template_id"].unique(), domains)
                tmpl = st.selectbox("Template", opts, format_func=fmt,
                                    key=f"identity_template_{pick}")
                sub = arms[arms["template_id"] == tmpl].sort_values("score", ascending=False)
                top, bot = sub.iloc[0], sub.iloc[-1]

                def _tag(r):
                    if "group" in r.index:
                        return f"{r[varies]} ({r['group'].replace('_', ' ')})"
                    return str(r[varies])

                st.markdown(f"**Prompt:** {html.escape(top['question'])}")
                st.caption(f"The {what} appears in the prompt as well as the answer, so this is "
                           "the highest-scoring variant's prompt.")
                side_by_side(f"Highest: {_tag(top)}", top["text"], top["score"],
                             f"Lowest: {_tag(bot)}", bot["text"], bot["score"])
                st.caption(f"Only the {what} differs. Every variant scored in this template:")
                st.dataframe(sub[[varies] + extra_cols + ["score"]], width="stretch",
                             hide_index=True, height=260)

# --------------------------------------------------------------------------------------
# Sycophancy
# --------------------------------------------------------------------------------------

with tabs[2]:
    sy = R.get("sycophancy")
    if not sy or not noise or not sy.get("premium_by_level"):
        st.info("The sycophancy probe was not part of this run, or predates the current schema.")
    else:
        items = module_items(R, "sycophancy")
        m = sy["agreement_main_effect"]
        slope = max(sy["insistence_slopes"], key=lambda s: s["mean_delta"])
        by_level = {p["insistence"]: p for p in sy["premium_by_level"]}

        st.markdown(verdict_line(items, "sycophancy",
                                 "the model is rewarded for agreeing rather than correcting"), unsafe_allow_html=True)

        # ---- the one chart that answers the module ----------------------------------------
        st.markdown("#### The reward for agreeing, as the user pushes harder")
        st.plotly_chart(viz.sycophancy_ladder(sy, noise=noise), width="stretch",
                        config={"displayModeBar": False})
        st.caption(
            "Above the line the model prefers agreeing with the user; below it, correcting them. "
            "Each scenario is asked three ways, changing only how hard the user pushes, never the "
            "two answers being compared."
        )
        tone = tone_check(sy, noise)
        if tone["diverges"]:
            st.warning(tone["text"] + "  Version 1 reports the average of the two; the appendix "
                       "has the breakdown.")

        if slope["mean_delta"] > 0 and slope["p_sign"] < 0.05:
            st.warning(
                f"**Read the slope, not the average.** Across everything the model barely prefers "
                f"agreement ({m['mean_delta']:+.2f} logits, p = {m['p_sign']:.2f}), so a test that "
                f"reported only the average would find nothing here. But once the user "
                f"{viz.LADDER_LABEL[slope['level']]}, the line crosses zero: the model flips from "
                f"preferring corrections to preferring agreement, a change of "
                f"{slope['mean_delta']:+.2f} logits (p = {slope['p_sign']:.3f}). Caving under "
                "pressure is the behaviour that gets amplified in RLHF."
            )

        # ---- inspect one level ------------------------------------------------------------
        st.markdown("#### Every scenario, at one level of pressure")
        pick = st.selectbox("How hard the user pushes", viz.LADDER,
                            index=viz.LADDER.index(slope["level"]),
                            format_func=lambda k: f"The user {viz.LADDER_LABEL[k]}")
        lvl = by_level[pick]
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(viz.sycophancy_scenarios(sy["arms"], pick), width="stretch",
                            config={"displayModeBar": False})
        with c2:
            st.metric("Reward for agreeing", f"{lvl['mean_delta']:+.2f} logits",
                      delta=f"agreement wins on {lvl['win_rate']:.0%} of scenarios",
                      delta_color="off")
            st.markdown(
                f"95% interval `{lvl['ci_low']:+.2f}` to `{lvl['ci_high']:+.2f}`, "
                f"p = `{lvl['p_sign']:.3f}` across {lvl['n_items']} scenarios. "
                + ("The model leans toward agreeing at this level."
                   if lvl["mean_delta"] > 0 else
                   "The model still prefers correcting at this level.")
            )
            st.caption(
                "Each dot is one scenario, red where agreeing scored higher and green where "
                "correcting did. A mean can come from every scenario leaning the same way or from "
                "a few extremes, and those mean different things for a policy trained on it."
            )
            st.caption(
                f"Agreements run about {abs(m['mean_token_delta']):.0f} tokens shorter than "
                "corrections, because a correction has to explain itself. Adjusting for that "
                f"leaves the agreement effect at {sy['length_adjusted']['agrees']['coef']:+.2f} "
                "logits overall, so it is not a length artifact."
            )

        # ---- the exact text ----------------------------------------------------------------
        st.markdown("#### Read the exact text")
        with st.expander(f"See the responses when the user {viz.LADDER_LABEL[pick]}",
                         icon=":material/article:", expanded=True, key="drill_sycophancy"):
            arms = pd.DataFrame(sy["arms"])
            topics = dict(zip(arms["scenario_id"], arms["topic"]))
            s_opts, s_fmt = labelled(topics, topics)
            sid = st.selectbox("Scenario", s_opts, format_func=s_fmt)
            sub = arms[(arms["scenario_id"] == sid) & (arms["insistence"] == pick)]
            st.markdown("**The user says:**")
            st.markdown(f'<div class="txtbox">{html.escape(sub.iloc[0]["question"])}</div>',
                        unsafe_allow_html=True)
            # Each scenario is written more than one way and the reported premium averages them,
            # so quote that average rather than either pair's own delta, which would match nothing
            # else on the tab. Showing every pair forced a label ("version 1 of 2") that told the
            # reader nothing about what separated them.
            variants = sorted({c.split("_", 1)[1] for c in sub["cell"]
                               if c.startswith("agrees_")}, reverse=True)
            pairs = [(sub[sub["cell"] == f"agrees_{v}"].iloc[0],
                      sub[sub["cell"] == f"corrects_{v}"].iloc[0]) for v in variants]
            avg = sum(a["score"] - c["score"] for a, c in pairs) / len(pairs)
            st.markdown(
                f"On this scenario the model prefers **{'agreeing' if avg > 0 else 'correcting'}**"
                f", by `{abs(avg):.3f}` logits."
            )
            shown_a, shown_c = pairs[0]
            side_by_side("Agrees with the user", shown_a["text"], shown_a["score"],
                         "Corrects the user", shown_c["text"], shown_c["score"], diff=False)
            if len(pairs) > 1:
                st.caption(
                    f"One of the {len(pairs)} ways this scenario is worded. All of them are "
                    "scored, and the number above is their average, which is why it does not "
                    "equal the difference between the two scores shown."
                )


# --------------------------------------------------------------------------------------
# Style & length
# --------------------------------------------------------------------------------------

with tabs[3]:
    sm = R.get("style")
    if not sm or not noise:
        st.info("The style probe was not part of this run.")
    else:
        items = module_items(R, "style", group="content_neutral")
        bearing = module_items(R, "style", group="quality_bearing")
        st.markdown(verdict_line(
            items, "style", "surface style is rewarded for its own sake", extra="Effects are shown after removing what the extra length alone explains."),
            unsafe_allow_html=True)

        st.markdown("#### Which styles the model rewards")
        # One scale across both charts, so a bar in one is comparable to a bar in the other.
        shared_max = max([i["ratio"] for i in items + bearing if i.get("ratio")] + [1.0]) * 1.28
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Style that adds no information**")
            st.caption("Any reward at all here is unearned, so these are bias claims.")
            st.plotly_chart(
                viz.bias_bars(items, signed=True, xmax=shared_max,
                              bad_label="rewarded though it adds nothing",
                              good_label="penalised, the model resists it"),
                width="stretch", config={"displayModeBar": False})
        with c2:
            st.markdown("**Style that might genuinely improve the answer**")
            st.caption("Rewarding these would not be a fault, so they are preferences, not bias.")
            st.plotly_chart(
                viz.bias_bars(bearing, signed=True, fault=False, xmax=shared_max),
                width="stretch", config={"displayModeBar": False})
        st.caption(
            f"That is {len(items) + len(bearing)} of the {len(sm['transform_groups'])} transforms "
            "applied. The eleventh, **padding**, is not charted because it *is* the ruler: "
            "content-free filler at four intensities is what traces the reward-versus-length "
            "curve below, and every effect above is measured against that curve."
        )

        # ---- the quality control: a finding, not methodology, so it stays in the flow -------
        pv = sm.get("padding_vs_elaboration", [])
        if pv:
            b = pv[0]
            good = b["mean_delta"] > 0
            st.markdown("#### Does it read substance, or count tokens?")
            c1, c2 = st.columns([1, 2])
            c1.metric("Genuine information versus filler, at matched length",
                      f"{b['mean_delta']:+.2f} logits",
                      delta=f"wins on {b['win_rate']:.0%} of questions", delta_color="off")
            c2.markdown(
                "Two answers, matched to within "
                f"{b['mean_length_mismatch']:.0f} tokens of each other. One adds real "
                "information, the other adds content-free filler. "
                + ("The model prefers the real information, so it is reading substance rather "
                   "than counting tokens. This is the internal check that the length adjustment "
                   "below is not hiding a quality effect."
                   if good else
                   "The model cannot tell them apart, so it is rewarding length rather than "
                   "substance.")
            )

        st.markdown("#### Read the exact text")
        with st.expander("See a transformed answer beside the plain one",
                         icon=":material/article:", expanded=True, key="drill_style"):
            arms = pd.DataFrame(sm["arms"])
            questions = dict(zip(arms["question_id"], arms["question"]))
            q_opts, q_fmt = labelled(arms["question_id"].unique(), questions)
            qid = st.selectbox("Question", q_opts, format_func=q_fmt)
            groups = {k: v.replace("_", "-") for k, v in sm["transform_groups"].items()}
            t_opts, t_fmt = labelled(set(arms["transform"]) - {"baseline"}, groups)
            tname = st.selectbox("Transform", t_opts, format_func=t_fmt)
            sub = arms[(arms["question_id"] == qid) & (arms["transform"] == tname)]
            base = arms[(arms["question_id"] == qid)
                        & (arms["transform"] == "baseline")].iloc[0]
            dose = st.select_slider("Intensity", options=sorted(sub["dose"].unique()))
            row = sub[sub["dose"] == dose].iloc[0]
            st.markdown(f"**Prompt:** {html.escape(row['question'])}")
            side_by_side("Plain answer", base["text"], base["score"],
                         f"With {tname} at intensity {dose}", row["text"], row["score"])
            st.caption(f"{base['n_answer_tokens']} to {row['n_answer_tokens']} tokens · "
                       f"score change {row['score'] - base['score']:+.3f}")


        # ---- everything about length, consolidated and out of the way -----------------------
        with st.expander("How length was accounted for"):
            st.markdown(
                "Comparing a styled answer against the plain one measures two things at once: the "
                "style, and the extra length the style drags along. This model dislikes added "
                "length, so the naive comparison makes almost every transform look disliked. Each "
                "arrow starts at what the naive comparison says and ends at what is left once the "
                "length effect is taken out. Every arrow points the same way, and three of them "
                "cross zero, which means the naive comparison got their direction wrong."
            )
            st.plotly_chart(viz.style_length_decomposition(sm), width="stretch",
                            config={"displayModeBar": False})
            flip = _length_flip(sm)
            if flip:
                name, raw, adj = flip
                st.info(
                    f"**{name.capitalize()}** is the one that changes a verdict. It adds no "
                    f"information, so any reward for it is unearned. Compared naively against the "
                    f"plain answer it scores `{raw:+.2f}` and looks disliked; once the length it "
                    f"adds is accounted for it is `{adj:+.2f}`. Without the length control this "
                    "would have been recorded as the model resisting flattery, when it mildly "
                    "prefers it."
                )
            st.markdown("**The length curve itself**")
            st.plotly_chart(viz.style_length_curve(sm), width="stretch",
                            config={"displayModeBar": False})
            st.caption(
                "Traced by adding content-free filler at four intensities, which is what every "
                "effect above is measured against. A single \"reward per 100 tokens\" would be a "
                "lossy average of this curve, so local slopes are marked instead. This tokenizer "
                "also discards newlines, so bullets and headers reach the model only as `-` and "
                "`#` characters; layout as such is invisible to it."
            )


# --------------------------------------------------------------------------------------
# Injection
# --------------------------------------------------------------------------------------

with tabs[4]:
    ij = R.get("injection")
    if not ij or "top_exploits" not in ij.get("summary", {}):
        st.info("The injection probe was not part of this run, or predates the current schema.")
    else:
        srch = ij["summary"].get("search") or {}
        beam = ij.get("beam_search")
        exploits = ij["summary"]["top_exploits"]
        base_asr = (ij["summary"].get("baseline_asr") or {}).get("p50") or 0.0

        # ---- did it work? ------------------------------------------------------------------
        if exploits:
            best = exploits[0]
            st.markdown(
                f'<div class="verdict"><b>Yes, injection works on this model.</b> '
                f'<code>{html.escape(best["label"])}</code> lifts a deliberately bad answer by '
                f'<b>{best["lift"]:+.2f} logits</b> and makes it outscore a genuine answer to the '
                f'same prompt <b>{best["asr"]["p50"]:.0%}</b> of the time, against '
                f'{base_asr:.0%} unattacked. {len(exploits)} of '
                f'{srch.get("n_candidates", 0)} attacks tried succeed at all.'
                f'<span class="hint">Ranked on {srch.get("n_dev_questions", 0)} development '
                f'prompts, measured on {srch.get("n_test_questions", 0)} held-out prompts the '
                'search never saw.</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="verdict clear"><b>No injection attack succeeded.</b> None of the '
                f'{srch.get("n_candidates", 0)} attacks tried made a bad answer outscore a genuine '
                f'answer more often than the unmodified bad answer already did ({base_asr:.0%}).'
                '</div>', unsafe_allow_html=True)

        # ---- which ones worked -------------------------------------------------------------
        if exploits:
            st.markdown("#### Which attacks worked")
            st.plotly_chart(viz.exploit_ranking(exploits, base_asr), width="stretch",
                            config={"displayModeBar": False})
            st.caption(
                "An attack counts as working only if it beats a real answer more often than the "
                "unmodified bad answer already does. Lift alone is not enough: an affix can add "
                "several logits and still leave the answer far below anything a real model writes."
            )

            # ---- the examples ---------------------------------------------------------------
            st.markdown("#### What the attack actually looks like")
            choice = st.selectbox(
                "Attack", list(range(len(exploits))),
                format_func=lambda i: f"{exploits[i]['label']} — beats a real answer "
                                      f"{exploits[i]['asr']['p50']:.0%} of the time")
            chosen = exploits[choice]
            if not chosen["examples"]:
                st.info("No held-out example of this attack cleared a genuine answer.")
            for k, ex in enumerate(chosen["examples"], start=1):
                st.markdown(f"**Example {k} of {len(chosen['examples'])}** · the user asked: "
                            f"*{html.escape(ex['question'])}*")
                side_by_side(f"The bad answer on its own ({ex['base_type']})",
                             ex["base_text"], ex["base_score"],
                             "The same answer, attacked", ex["attacked_text"],
                             ex["attacked_score"], ins_class="attack")
                st.caption(
                    f"Highlighted in red is what the attack bolted on. It moved the score "
                    f"{ex['attacked_score'] - ex['base_score']:+.2f} logits, "
                    f"past the median genuine answer to this prompt at "
                    f"`{ex['median_genuine_score']:+.2f}`. For comparison, a real answer reads: "
                    f"*{html.escape(ex['genuine_answer'][:160])}…*"
                )

        # ---- supporting evidence, all below the fold ---------------------------------------
        st.markdown("#### Supporting evidence")

        with st.expander("How hard is the bar? Attack success against several definitions of "
                         "\"a real answer\""):
            st.plotly_chart(viz.attack_success(ij["summary"], beam), width="stretch",
                            config={"displayModeBar": False})
            st.caption(
                "The bar is set at several percentiles of genuine answers to the same prompt, "
                "because any single choice of bar could be tuned after the fact. Grey is the same "
                f"bad answer with nothing attached. Based on {ij['n_reference_ok']} prompts with "
                "enough genuine answers to support a percentile."
            )

        if beam:
            with st.expander("How the search found the stacked attack"):
                c1, c2, c3 = st.columns(3)
                c1.metric("On the prompts it was tuned on",
                          f"{beam['dev_mean_lift']:+.2f}"
                          if beam.get("dev_mean_lift") is not None else "n/a")
                c2.metric("On held-out prompts", f"{beam['heldout_mean_lift']:+.2f}",
                          delta=(f"{beam['heldout_mean_lift'] - beam['dev_mean_lift']:+.2f} lost "
                                 "to over-fitting")
                          if beam.get("dev_mean_lift") is not None else None,
                          delta_color="normal")
                c3.metric("Beats a median genuine answer",
                          f"{beam['heldout_asr']['p50']:.0%}"
                          if beam["heldout_asr"].get("p50") is not None else "n/a")
                st.dataframe(pd.DataFrame([
                    {"depth": h["depth"], "stack": " + ".join(a for a, _ in h["best_stack"]),
                     "candidates tried": h["n_candidates"],
                     "best lift on dev": round(h["best_dev_lift"], 3)}
                    for h in beam["history"]]), width="stretch", hide_index=True)
                st.caption(
                    "Every step was chosen on development prompts only. Reporting the dev number "
                    "would overstate the attack, which is what the gap above measures."
                )

        with st.expander("Every affix tried, ranked on development and measured on held-out"):
            st.plotly_chart(viz.injection_lifts(ij["summary"]), width="stretch",
                            config={"displayModeBar": False})
            attacks = [a for a in ij["summary"]["affixes"] if not a["is_control"]]
            worst_shrink = max((abs(a["shrinkage"]) for a in attacks), default=0.0)
            st.caption(
                "Grey bars are controls, not attacks: content-free text of matched length, and "
                "lookalike strings that cannot produce a special token. An affix has to beat its "
                f"control, not merely beat silence. Individual affixes lose at most "
                f"{worst_shrink:.2f} logits between development and held-out prompts, because each "
                "is a single pre-written candidate rather than something the search optimised."
            )
            st.dataframe(
                pd.DataFrame(ij["summary"]["affixes"])[
                    ["dev_rank", "affix_id", "family", "position", "is_control", "dev_mean_lift",
                     "mean_lift", "shrinkage", "lift_ci_low", "lift_ci_high", "adjusted_lift",
                     "mean_sep_injected", "noise_percentile"]],
                width="stretch", hide_index=True, height=340)

        kc = ij.get("key_contrasts", [])
        if kc:
            with st.expander("Is it really the attack, or just the text it carries?"):
                for c in kc:
                    exceeds = c.get("exceeds_noise_floor")
                    st.markdown(
                        f'<div class="finding" style="border-left-color:'
                        f'{viz.STATUS["critical"] if exceeds else viz.STATUS["warning"]}">'
                        f'<b>{html.escape(c["attack"])} versus {html.escape(c["control"])}: '
                        f'{c["mean_delta"]:+.2f} logits</b>'
                        f'<div class="sub">{html.escape(c["question"])}<br>'
                        f'Consistent on {c["win_rate"]:.0%} of items, p = {c["p_sign"]:.1e}, '
                        f'bigger than {c.get("noise_percentile", 0):.0f}% of rewordings.</div>'
                        "</div>", unsafe_allow_html=True)
                st.caption(
                    "Typing `[SEP]` into an answer really does insert the model's genuine "
                    "separator token, and the raw lift looks dramatic. But the control that "
                    "appends the same good answer with no separator recovers most of it. The "
                    "forged token is worth only the remainder."
                )

        cont = ij.get("contamination", {})
        if cont:
            with st.expander("Is padding free? What junk costs on a good answer"):
                cols = st.columns(len(cont))
                for col, (where, d) in zip(cols, cont.items()):
                    col.metric(f"Junk {where} to a good answer", f"{d['mean_delta']:+.2f}")
                first = next(iter(cont.values()))
                st.caption(
                    "A model that ignored contamination would sit near zero, which would mean a "
                    "policy could emit filler at no cost. "
                    + (f"This one drops {abs(first['mean_delta']):.2f} logits, so filler is not "
                       "free here." if first["mean_delta"] < -0.5 else "")
                )


# --------------------------------------------------------------------------------------
# Compare models
# --------------------------------------------------------------------------------------

with tabs[5]:
    others = [f for f in files if f.name != st.session_state.get("results_name")]
    if not others:
        st.info(
            "Only one results file exists. Run a scan on a second model, for example "
            "`OpenAssistant/reward-model-deberta-v3-base`, to compare them here."
        )
    else:
        other_label = st.selectbox(
            "Compare against",
            [f.name.replace("__", " · ").replace(".json", "") for f in others])
        of = others[[f.name.replace("__", " · ").replace(".json", "")
                     for f in others].index(other_label)]
        O = cached_results(str(of), of.stat().st_mtime)
        A_name = meta["model_id"].split("/")[-1]
        B_name = O["meta"]["model_id"].split("/")[-1]
        o_sev = sev.summarise_all(O["findings"])

        # ---- the answer first ---------------------------------------------------------------
        st.markdown(compare_verdict(R, O, A_name, B_name, severity, o_sev),
                    unsafe_allow_html=True)
        st.caption(
            "Raw scores are not comparable between checkpoints: the two have different scales and "
            "different calibration. Everything here is a percentile of each model's *own* "
            "rewording noise, or a probability. Both are measured per model and so mean the same "
            "thing on either side."
        )

        # ---- where they differ ----------------------------------------------------------------
        amap = {finding_key(f): f for f in R["findings"]}
        bmap = {finding_key(f): f for f in O["findings"]}
        shared = [k for k in amap if k in bmap
                  and amap[k].get("noise_percentile") is not None
                  and bmap[k].get("noise_percentile") is not None]
        if shared:
            rows = [{"title": amap[k]["title"].split(":")[0][:70],
                     "a": amap[k]["noise_percentile"], "b": bmap[k]["noise_percentile"],
                     "valence": amap[k]["valence"]} for k in shared]
            rows = sorted(rows, key=lambda r: -abs(r["a"] - r["b"]))[:14]
            st.markdown("#### Where they differ most")
            st.plotly_chart(viz.compare_findings(rows, A_name, B_name), width="stretch",
                            config={"displayModeBar": False})
            st.caption(
                "Sorted by how far apart the two models are. A bar past the dotted line means the "
                "effect is larger than 95% of that model's own meaning-preserving rewordings."
            )

        # ---- severity, grouped the same way as everywhere else --------------------------------
        st.markdown("#### Severity side by side")
        severity_tiles(severity, other=o_sev, names=(A_name, B_name))

        # ---- head to head ---------------------------------------------------------------------
        st.markdown("#### The two yardsticks, and the self-check")
        c1, c2 = st.columns(2)
        for col, res, name in ((c1, R, A_name), (c2, O, B_name)):
            with col:
                st.markdown(f"**{name}**")
                cc = res.get("calibration") or {}
                sk = res.get("sanity_check") or {}
                st.metric("Agreement with human preferences",
                          f"{cc['accuracy']:.1%}" if cc.get("fitted") else "n/a",
                          help="Chance is 50%.")
                st.metric("Rewording noise, median",
                          f"{res['noise_floor']['median_abs_delta']:.2f} logits")
                # "normal" keeps a negative margin red, which is what a failure should look like.
                passed = sk.get("passed")
                st.metric("Prefers support over abuse",
                          "n/a" if passed is None else ("Passes" if passed else "FAILS"),
                          delta=f"{sk.get('margin', 0):+.2f} logits", delta_color="normal",
                          help="The worked example from the OpenAssistant model card. A model "
                               "that scores the abusive reply higher would pay a policy to be "
                               "abusive.")

        # ---- the detail, below the fold --------------------------------------------------------
        sa = {a["transform"]: a for a in (R.get("style") or {}).get("adjusted", [])}
        sb = {a["transform"]: a for a in (O.get("style") or {}).get("adjusted", [])}
        common = sorted(set(sa) & set(sb))
        if common:
            neutral = [t for t in common if sa[t]["group"] == "content_neutral"]
            a_rew = [t for t in neutral if sa[t]["effect_at_max_dose"] > 0]
            b_rew = [t for t in neutral if sb[t]["effect_at_max_dose"] > 0]
            st.markdown("#### Does the bigger model fix the smaller one's style habits?")
            st.markdown(
                f"Of {len(neutral)} transforms that add no information, **{A_name}** rewards "
                f"{len(a_rew)} and **{B_name}** rewards {len(b_rew)}. Rewarding any of them is "
                "unearned, so fewer is better."
            )
            with st.expander("Every transform, both models"):
                st.dataframe(pd.DataFrame([{
                    "transform": t, "group": sa[t]["group"],
                    f"{A_name} (logits at max)": sa[t]["effect_at_max_dose"],
                    f"{B_name} (logits at max)": sb[t]["effect_at_max_dose"],
                } for t in common]), width="stretch", hide_index=True)
        if not shared:
            st.info("The two runs have no probes in common to line up.")


# --------------------------------------------------------------------------------------
# Appendix
# --------------------------------------------------------------------------------------

with tabs[6]:
    st.markdown(
        "Everything behind the numbers: the run itself, the two yardsticks they are measured "
        "against, every stimulus that was scored, every statistic, and how each was computed."
    )

    # ---- what people actually come here for ---------------------------------------------------
    st.markdown("#### Take the data")
    c1, c2 = st.columns([1, 2])
    with c1:
        rp = st.session_state.get("results_path") or meta.get("results_path")
        if rp and Path(rp).exists():
            st.download_button("Download full results JSON", Path(rp).read_bytes(),
                               file_name=Path(rp).name, mime="application/json",
                               width="stretch")
        prov = meta.get("provenance", {})
        st.caption(
            f"{meta['model_id']} · {meta['depth']} scan · seed {meta['seed']} · "
            f"{prov.get('device', '?')} · {meta.get('runtime_seconds', 0):.0f}s · "
            f"torch {prov.get('torch', '?')}, transformers {prov.get('transformers', '?')}"
        )
    with c2:
        st.caption(
            "Every number on every tab is in that file, including the per-item rows behind each "
            "contrast. Two runs at the same seed produce byte-identical output, so a diff between "
            "two files is a real change in the model or the code, never resampling noise."
        )

    # ---- the instruments ----------------------------------------------------------------------
    st.markdown("#### The two yardsticks, in detail")
    st.caption(
        "The overview reports one headline number from each of these. Here is the distribution "
        "that number came from, and the checks that say the yardstick itself can be trusted: an "
        "instrument that drifts between runs, or that moves on a change of quotation marks, "
        "cannot be the scale for anything else."
    )
    i1, i2 = st.columns(2)
    with i1:
        st.markdown("**Rewording noise**")
        if noise:
            st.plotly_chart(viz.noise_floor_hist(noise), width="stretch",
                            config={"displayModeBar": False})
            def _tiny(v: float) -> str:
                # "0e+00" is technically right and reads like a glitch.
                return "exactly 0" if v == 0 else f"{v:.0e}"

            m1, m2 = st.columns(2)
            m1.metric("Repeat scoring differs by", _tiny(noise["determinism_delta"]))
            m2.metric("Alone vs in a batch", _tiny(noise["batch_invariance_delta"]))
            with st.expander("Negative controls: byte changes that should score identically"):
                st.dataframe(pd.DataFrame(noise["semantic_nulls"]).T.reset_index()
                             .rename(columns={"index": "perturbation"}),
                             width="stretch", hide_index=True)
                st.caption(
                    "These change bytes but not meaning, so they should be zero. Whatever they "
                    "are is the floor below the floor."
                )
    with i2:
        st.markdown("**Agreement with human preferences**")
        if cal and cal.get("fitted"):
            st.plotly_chart(viz.calibration_reliability(cal), width="stretch",
                            config={"displayModeBar": False})
            m1, m2 = st.columns(2)
            m1.metric("Fitted temperature", f"{cal['temperature']:.2f}")
            m2.metric("Median genuine gap", f"{cal['median_abs_gap']:.2f} logits")
            st.caption(
                f"Fitted on {cal['note']}. A score difference of *d* logits means the model "
                f"prefers that variant sigmoid(d / {cal['temperature']:.2f}) of the time."
            )
        else:
            st.info("Calibration was not run, so effects are reported as raw logits.")

    st.markdown("#### Every number")
    st.caption("Every number behind the charts. The dashboard shows the effects; these are the "
               "intervals, win rates, token deltas and p-values behind them.")
    sm_, sy_, idr_ = R.get("style"), R.get("sycophancy"), R.get("identity")
    if sm_:
        with st.expander("Style — every transform at every intensity"):
            st.dataframe(pd.DataFrame(sm_["contrasts"])[
                ["transform", "dose", "group", "mean_delta", "ci_low", "ci_high", "win_rate",
                 "mean_added_tokens", "p_sign", "p_adjusted", "noise_percentile"]],
                width="stretch", hide_index=True, height=380)
        with st.expander("Style — length-adjusted regression terms"):
            st.dataframe(pd.DataFrame(sm_["regression"]["terms"]), width="stretch",
                         hide_index=True)
            st.caption(
                f"Question fixed effects absorbed, {sm_['regression']['n_clusters']} clusters, "
                f"within-R² {sm_['regression']['r2_within']:.3f}. Length enters as a restricted "
                "cubic spline; p-values are wild cluster bootstrap."
            )
        if sm_.get("padding_vs_elaboration"):
            with st.expander("Style — genuine elaboration versus filler, question by question"):
                st.dataframe(pd.DataFrame(sm_["padding_vs_elaboration"][0]["rows"]),
                             width="stretch", hide_index=True, height=300)
    if sy_:
        with st.expander("Sycophancy — agreement premium split by wording, not used in version 1"):
            st.dataframe(pd.DataFrame(sy_["premiums"])[
                ["insistence", "tone", "mean_delta", "ci_low", "ci_high", "win_rate",
                 "p_sign", "p_adjusted", "mean_token_delta"]], width="stretch", hide_index=True)
    if idr_:
        with st.expander("Identity — pairwise group gaps"):
            for subset, label in (("ood", "Decision contexts"), ("id", "Assistant dialogue")):
                blk = idr_.get(f"names_{subset}")
                if not blk:
                    continue
                st.markdown(f"**{label}**")
                cols = ["group_a", "group_b", "gap", "ci_low", "ci_high", "p_wild",
                        "noise_percentile"]
                df_ = pd.DataFrame(blk["pairwise_gaps"])
                if "q_value" in df_:
                    cols.insert(-1, "q_value")
                st.dataframe(df_[cols], width="stretch", hide_index=True)
            st.caption(
                "Intervals resample whole templates; p-values are a wild cluster bootstrap, which "
                "is the standard correction when there are only a few dozen clusters."
            )

    st.markdown("#### Every stimulus")
    # Gated the same way as the overview block, so a stub scan does not advertise texts it
    # never scored.
    if (sc := R.get("sanity_check")) and sc.get("passed") is not None:
        with st.expander("The self-check shown on the overview \u2014 model card worked example"):
            st.caption(f"Source: {sc['source']}. Both replies answer the same question, so the "
                       "two scores are directly comparable.")
            st.markdown(f'<div class="txtbox"><b>Question.</b> '
                        f'{html.escape(RewardModel.CARD_QUESTION)}</div>',
                        unsafe_allow_html=True)
            k1, k2 = st.columns(2)
            k1.metric("Supportive reply", f"{sc['helpful_score']:.2f}")
            k1.markdown(f'<div class="txtbox">{html.escape(RewardModel.CARD_HELPFUL)}</div>',
                        unsafe_allow_html=True)
            k2.metric("Abusive reply", f"{sc['rude_score']:.2f}")
            k2.markdown(f'<div class="txtbox">{html.escape(RewardModel.CARD_RUDE)}</div>',
                        unsafe_allow_html=True)
    base = Path("rmi/corpora")
    for name, label in [("style_corpus.json", "Factual corpus, transforms and paraphrases"),
                        ("sycophancy.json", "Sycophancy scenarios"),
                        ("identity.json", "Identity templates and name lists"),
                        ("affixes.json", "Attack affixes and base responses")]:
        path = base / name
        if path.exists():
            data = json.loads(path.read_text())
            with st.expander(f"{label} — {name}"):
                if name == "identity.json":
                    st.markdown(f"*{data['citation']}*")
                st.json(data, expanded=False)


    # ---- how, for the reader who wants to check the method ------------------------------------
    st.markdown("#### How each number was produced")
    with st.expander("Statistical methodology"):
        st.markdown("""
    **Scoring.** The reward model emits one scalar per (prompt, answer) pair. Scores are not
    comparable across different prompts, so every contrast here is paired within a prompt. Scoring is
    deterministic: repeated runs give bit-identical results, so the only randomness in the project is
    which items were written.

    **The reward unit problem.** A raw logit means nothing on its own. Dividing by a hand-authored
    "good minus poor answer" gap would be worse than nothing, because the size of that unit is set
    entirely by how bad the poor answers are written to be. Instead effects are reported three ways:
    as raw logits, as a percentile of the **paraphrase noise floor**, and as a **calibrated preference
    probability**. The last uses the fact that these models are trained with a pairwise ranking loss,
    so a within-prompt score difference already estimates a log-odds; a single temperature is fitted
    against held-out human preference data to make that reading honest.

    **Clustering.** The same questions and templates are reused across many contrasts, which makes the
    observations dependent. Every confidence interval resamples whole **clusters** (questions,
    templates, scenarios) rather than rows; resampling rows would produce intervals several times too
    narrow. Bias-corrected and accelerated intervals are used where there are enough clusters, and the
    wild cluster bootstrap where there are few.

    **Multiplicity.** Dozens of contrasts are tested. Within each module, adjusted p-values come from
    **Westfall-Young step-down max-T permutation**, which is valid under arbitrary dependence between
    contrasts and is more powerful than Benjamini-Hochberg under the positive correlation that shared
    items create. The permutation flips the sign of all of a cluster's deltas at once, and the same
    draws are reused across contrasts so their correlation is preserved.

    **Effect sizes.** Mean difference with a cluster bootstrap interval, plus the win rate and the
    spread across items. Cohen's *d* is deliberately **not** reported: because the scorer is
    deterministic, its denominator contains no measurement noise, so it measures how *consistent* an
    effect is across hand-written items rather than how *large* it is, and it diverges to infinity for
    a perfectly uniform effect.

    **Directionality.** Identity is two-sided, since a shift either way is bias. Sycophancy is
    one-sided. Style transforms that add no information are one-sided, because any reward for them is
    unearned; transforms that might genuinely improve an answer are two-sided and reported as
    preferences rather than as bias.

    **Selection.** Reporting the largest of several gaps is biased upward. The identity module handles
    this with an omnibus permutation test on group means; the injection module handles it by doing all
    affix ranking and search on development prompts and reporting only held-out numbers.
    """)
        st.markdown("**Severity thresholds**")
        st.json(meta.get("severity_thresholds", {}), expanded=False)
        st.caption(
            "Severity is the worst confirmed effect in a category, as a percentile of the "
            "paraphrase noise floor. Prevalence is the share of that category's probes that are "
            "both statistically confirmed and above the materiality threshold. Two numbers rather "
            "than one, because a mean over contrasts can be diluted by adding null contrasts "
            "without anything about the model changing."
        )

    st.markdown("#### Limitations")
    st.markdown("""
- **These stimuli are hand-written and not blind-authored.** A pair that differs in more than the
  intended variable produces a confident false finding. The paraphrase floor and the within-group
  name controls exist to bound how much that can matter, but they do not eliminate it.
- **This is the pilot corpus size.** Descriptor axes in particular rest on only a few templates and
  should be read as indicative.
- **Decision-context templates are far outside these models' training distribution**, which was
  web question answering, summarisation and assistant dialogue. That is why assistant-dialogue
  templates are reported separately.
- **Per-prompt percentile references are built from a modest number of genuine answers**, so the
  attack-success curves are coarse at the upper percentiles.
""")
