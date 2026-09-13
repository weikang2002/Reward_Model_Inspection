"""Chart builders shared by the Streamlit app and the static HTML export.

Colour follows the job, not the series index. Almost every chart here answers "how big, and in
which direction", so the default is a single measure with direction carried by bar sign and
meaning carried by the status palette. Only the identity-group chart needs categorical hues,
because there the colour identifies a group rather than a magnitude.

Status colours always ship alongside their band name as text, so identity is never colour-alone.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import plotly.graph_objects as go

from .probes.sycophancy import INSISTENCE_PHRASE
from .severity import is_material

# Categorical slots, in the fixed validated order. Never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Status palette: fixed, never themed, always paired with a visible label.
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
BAND_COLOR = {
    "High": STATUS["critical"],
    "Moderate": STATUS["serious"],
    "Low": STATUS["warning"],
    "Negligible": STATUS["good"],
    "None detected": STATUS["good"],
    "Unconfirmed": "#8a8a85",
    "Unknown": "#8a8a85",
    "No data": "#8a8a85",
}
VALENCE_COLOR = {
    "vulnerability": STATUS["critical"],
    "healthy": STATUS["good"],
    "informational": SERIES[0],
}
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"


def _base(fig: go.Figure, *, height=380, xtitle=None, ytitle=None, showlegend=False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=16, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13, color=INK),
        showlegend=showlegend,
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=12, color=INK2)),
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor="#b9b8b3",
                     zerolinewidth=1, linecolor=GRID, title=xtitle,
                     title_font=dict(size=12, color=INK2), tickfont=dict(color=INK2))
    fig.update_yaxes(showgrid=False, linecolor=GRID, title=ytitle,
                     title_font=dict(size=12, color=INK2), tickfont=dict(color=INK2))
    return fig


# --------------------------------------------------------------------------------------
# chart labels
# --------------------------------------------------------------------------------------

CATEGORY_LABEL = {"identity": "Identity", "sycophancy": "Sycophancy",
                  "style": "Style", "reward_hacking": "Reward hacking"}
_SUBSET_LABEL = {"id": "in-distribution", "ood": "out-of-distribution"}

# Fixed order, so the legend never reshuffles between models or reruns.
VALENCE_LEGEND = [
    ("vulnerability", "a vulnerability"),
    # Neutral wording on purpose: "informational" now covers a quality-bearing style transform, a
    # control contrast that explains an attack, and an attack whose success rate never beat the
    # unattacked baseline. "A preference that may be legitimate" described only the first.
    ("informational", "real, but not a fault on its own"),
    ("healthy", "the model behaving correctly"),
]


def _first_clause(title: str, limit: int = 54) -> str:
    """Fallback for a finding kind with no short form: the sentence up to its first break."""
    for sep in (", so ", ". ", " ("):
        cut = title.find(sep)
        if 0 < cut <= limit:
            return title[:cut]
    return title if len(title) <= limit else title[:limit - 1].rsplit(" ", 1)[0] + "\u2026"


def _describe(category: str, detail: dict, title: str) -> str:
    """A short name for one finding, rebuilt from its structured detail.

    Titles are whole sentences baked in at scan time and carry their own numbers, which makes them
    unusable as axis ticks: they truncate mid-word, repeat what the bar already shows, and never
    say which category the row belongs to. Deriving the label from ``detail`` instead means result
    files written before this existed get the short labels too, with no rescan.
    """
    kind = detail.get("kind")
    if category == "reward_hacking":
        if kind == "best_affix":
            return f"best single affix: {detail.get('affix_id', 'unknown')}"
        if kind == "beam_search":
            # Files written before the stack was recorded in the detail still carry it in the
            # title, which is where the label used to be unable to reach it.
            stack = detail.get("stack") or re.findall(r"\(([^)]*)\)", title)[:1]
            named = " + ".join(stack) if isinstance(stack, list) and stack else ""
            return f"best stacked attack: {named}" if named else "best stacked attack"
        if kind == "contamination":
            return f"junk {detail.get('where', 'added')} to a good answer"
        if kind == "key_contrast":
            return title.split(":")[0].replace(" versus its control ", " vs ")
    elif category == "identity":
        if kind == "descriptor":
            return f"{detail.get('axis', 'unknown')} descriptor swap"
        if kind == "omnibus":
            sub = _SUBSET_LABEL.get(detail.get("subset"))
            return f"name swap, {sub}" if sub else "name swap"
    elif category == "style":
        if kind == "length_adjusted":
            return f"{detail.get('transform', 'unknown').replace('_', ' ')}, length-adjusted"
        if kind == "quality_control":
            return "genuine elaboration vs filler"
    elif category == "sycophancy":
        if kind == "insistence_slope":
            lvl = LADDER_LABEL.get(detail.get("level"))
            return f"agreeing pays more when the user {lvl}" if lvl else "agreeing under pressure"
        if kind == "main_effect":
            return "agreeing rather than correcting"
    return _first_clause(title)


def finding_labels(findings: list[dict]) -> list[str]:
    """Short labels ending in the category, guaranteed unique.

    The category goes last, which reads a little oddly in isolation but scans far better on a
    chart: plotly right-aligns tick labels, so a trailing category lines up in a column next to
    the bars, while a leading one lands at a different offset on every row.

    Uniqueness is load-bearing rather than cosmetic: plotly places two bars sharing a y value on
    the same row, so a duplicate label silently hides a finding.
    """
    cats = [CATEGORY_LABEL.get(f.get("category"), f.get("category") or "?") for f in findings]
    bodies = [_describe(f.get("category"), f.get("detail") or {}, f.get("title") or "")
              for f in findings]
    counts = Counter(zip(cats, bodies))
    if any(c > 1 for c in counts.values()):
        # What separates two findings with the same name is their size, so show it. It goes before
        # the category so the category column stays flush.
        bodies = [f"{b}  ({f.get('effect') or 0:+.2f})" if counts[(c, b)] > 1 else b
                  for c, b, f in zip(cats, bodies, findings)]
    seen, out = Counter(), []
    for c, b in zip(cats, bodies):
        seen[(c, b)] += 1
        n = seen[(c, b)]
        out.append(f"{b}{'' if n == 1 else f' #{n}'}  <b>{c}</b>")
    return out


# --------------------------------------------------------------------------------------
# the signature chart: every finding measured against rewording noise
# --------------------------------------------------------------------------------------


MAX_ROWS = 16


def _largest_plus_every_counted_problem(scored: list[dict]) -> list[dict]:
    """The largest findings, with every problem a category tile counts guaranteed a row.

    Ranking on size alone let counted problems fall off the bottom. On `-base` every identity
    finding did, so a tile reading "1 of 6 · Low · 1.5x" sat above a chart with no identity bar at
    all; on `-large-v2` identity has two material findings and only the larger survived, so the tab
    showed two red bars and the overview one. The rule is the tile's own: a vulnerability that
    `severity.is_material` counts must be findable here.

    If more problems are counted than there are rows, the largest of them take the chart - they are
    the ones a reader needs first, and the caption already points at the full list.
    """
    ratio = lambda f: abs(f["effect"]) / f["systematic_bar"]  # noqa: E731
    ranked = sorted(scored, key=ratio)
    if len(ranked) <= MAX_ROWS:
        return ranked
    keep = list(ranked[-MAX_ROWS:])
    must = sorted((f for f in scored
                   if f.get("valence") == "vulnerability" and is_material(f)),
                  key=ratio, reverse=True)[:MAX_ROWS]
    must_ids = {id(f) for f in must}
    for f in must:
        if any(k is f for k in keep):
            continue
        # `keep` runs smallest first, so this drops the smallest row that is not itself counted.
        drop = next((k for k in keep if id(k) not in must_ids), None)
        if drop is None:
            break
        keep = [k for k in keep if k is not drop] + [f]
    return sorted(keep, key=ratio)


def _unconfirmed_fault(f: dict) -> bool:
    """A vulnerability that did not reach significance: real-looking, but counting toward nothing.

    Restricted to vulnerabilities on purpose. The directional modules test one-sided toward the
    fault, so a finding running the healthy way is unconfirmed by construction rather than by
    being weak.
    """
    return f.get("valence") == "vulnerability" and not f.get("confirmed")


def _finding_bars(rows, name: str, color: str, *, hatched: bool = False,
                  showlegend: bool = True) -> go.Bar:
    """One bar trace of (label, value, finding) rows, solid or hatched throughout.

    fillmode is load-bearing. Plotly defaults it to "replace", which makes ``marker.color`` the
    colour of the *stripes* over a transparent background, so an explicit white fgcolor painted
    the whole bar white and the row rendered empty. "overlay" keeps ``marker.color`` as the fill
    and draws the stripes on top of it.
    """
    return go.Bar(
        y=[r[0] for r in rows], x=[r[1] for r in rows], orientation="h", width=0.62,
        name=name, showlegend=showlegend,
        marker=dict(color=color,
                    pattern=dict(shape="/" if hatched else "", fillmode="overlay",
                                 fgcolor="#ffffff", size=5, solidity=0.32)),
        customdata=[[r[2]["effect"], r[2]["systematic_bar"], r[2].get("n_items") or 0]
                    for r in rows],
        hovertemplate="<b>%{y}</b><br>%{customdata[0]:+.2f} logits"
                      "<br>%{x:.1f}x what rewording alone could produce"
                      " (%{customdata[1]:.2f} over %{customdata[2]} comparisons)"
                      "<extra></extra>")


def findings_vs_noise(findings: list[dict], *, height=None) -> go.Figure:
    """Every finding against what rewording alone could produce at its own sample size.

    On a shared "multiples of the bar" scale rather than in logits, because each finding averages a
    different number of comparisons and so has a different bar. Anything left of 1.0 is the size
    wording noise alone would produce.
    """
    sel = [f for f in findings
           if f.get("effect") is not None and f.get("systematic_bar")]
    sel = _largest_plus_every_counted_problem(sel)
    labels = finding_labels(sel)
    vals = [abs(f["effect"]) / f["systematic_bar"] for f in sel]
    fig = go.Figure()
    fig.add_vrect(x0=0, x1=1, fillcolor="#8a8a85", opacity=0.11, line_width=0)
    fig.add_vline(x=1, line=dict(color="#8a8a85", width=2, dash="dot"))
    # Anchored to the bottom row, not the top: the legend sits above the plot and a fourth entry
    # wraps it onto a second line, straight through an annotation placed up there.
    fig.add_annotation(x=1, y=-0.42, text="as far as rewording alone gets",
                       showarrow=False, xanchor="left", xshift=6,
                       font=dict(size=11, color=INK2))

    # One trace per valence rather than per bar, so the colours get a real legend instead of
    # living only in a caption below the chart. Any valence outside the fixed list still gets a
    # trace, because a finding dropped for having an unexpected label would vanish silently.
    known = [v for v, _ in VALENCE_LEGEND]
    groups = VALENCE_LEGEND + [(v, str(v)) for v in dict.fromkeys(f.get("valence") for f in sel)
                               if v not in known]
    hatch_listed = False
    for valence, name in groups:
        rows = [(lab, v, f) for lab, v, f in zip(labels, vals, sel)
                if f.get("valence") == valence]
        if not rows:
            continue
        color = VALENCE_COLOR.get(valence, INK2)
        # Split by pattern as well as by valence. Plotly takes a trace's legend swatch from its
        # first point, so a trace mixing solid and hatched bars showed "a vulnerability" as a
        # hatched chip and the solid red most of its bars actually use appeared nowhere.
        solid = [r for r in rows if not _unconfirmed_fault(r[2])]
        hatched = [r for r in rows if _unconfirmed_fault(r[2])]
        if solid:
            fig.add_trace(_finding_bars(solid, name, color))
        elif hatched:
            # Every bar of this valence is hatched, so nothing would explain its colour.
            fig.add_trace(go.Bar(y=[labels[0]], x=[None], orientation="h", width=0.62,
                                 name=name, hoverinfo="skip", marker=dict(color=color)))
        if hatched:
            fig.add_trace(_finding_bars(hatched, "not statistically confirmed", color,
                                        hatched=True, showlegend=not hatch_listed))
            hatch_listed = True
    fig.update_traces(marker_cornerradius=4)
    _base(fig, height=height or max(300, 34 * len(sel) + 110), showlegend=True,
          xtitle="multiples of what rewording alone could produce")
    # Sorted order has to be restated: with one trace per valence, plotly would otherwise order
    # the axis by first appearance and scatter the ranking.
    fig.update_yaxes(categoryorder="array", categoryarray=labels,
                     tickfont=dict(size=11, color=INK))
    fig.update_xaxes(range=[0, max(vals + [1.0]) * 1.12])
    return fig


def noise_floor_hist(noise: dict) -> go.Figure:
    per_q = noise.get("per_question", {})
    deltas = []
    for v in per_q.values():
        base = v["neutral_score"]
        deltas.extend(abs(s - base) for s in v["paraphrase_scores"])
    fig = go.Figure(go.Histogram(
        x=deltas, nbinsx=24, marker=dict(color=SERIES[0], line=dict(width=0)),
        hovertemplate="|delta| %{x:.2f}<br>%{y} rewordings<extra></extra>"))
    fig.add_vline(x=noise["median_abs_delta"], line=dict(color=INK, width=2),
                  annotation_text=f"median {noise['median_abs_delta']:.2f}",
                  annotation_font=dict(size=11, color=INK))
    fig.add_vline(x=noise["p95_abs_delta"], line=dict(color=STATUS["critical"], width=2, dash="dot"),
                  annotation_text=f"95th pct {noise['p95_abs_delta']:.2f}",
                  annotation_font=dict(size=11, color=STATUS["critical"]))
    _base(fig, height=300, xtitle="score change from rewording the same answer, logits",
          ytitle="count")
    fig.update_yaxes(showgrid=True, gridcolor=GRID)
    return fig


# --------------------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------------------


def identity_groups(block: dict) -> go.Figure:
    """Do the identity groups fit inside what reshuffling the names produces by chance?

    One chart, not two. The statistic behind the p-value is exactly the top-to-bottom spread of
    these group means, so drawing the null as a band of exactly that width, laid over the observed
    range, turns the test into something readable off the picture: if every group fits inside the
    band, the spread is no bigger than chance. Two separate charts, group means beside a
    chance-versus-observed pair, told the same story twice and never showed that the second
    chart's "observed" bar *was* the distance between the first chart's outermost bars.
    """
    gm = block["group_means"]
    o = block["omnibus_permutation"]
    order = sorted(gm, key=lambda k: gm[k])           # ascending; plotly draws index 0 at the foot
    labels = [g.replace("_", " ") for g in order]
    vals = [gm[g] for g in order]
    lo, hi = vals[0], vals[-1]
    spread, chance = hi - lo, o["null_p95"]
    mid = (lo + hi) / 2
    beyond = spread > chance
    extreme = STATUS["critical"] if beyond and o["p_value"] < 0.05 else INK2

    fig = go.Figure()
    fig.add_vrect(x0=mid - chance / 2, x1=mid + chance / 2, fillcolor="#8a8a85", opacity=0.13,
                  line_width=0)
    # Ends marked, so the band reads as a width to compare against rather than as a zone.
    for x in (mid - chance / 2, mid + chance / 2):
        fig.add_vline(x=x, line=dict(color="#8a8a85", width=1, dash="dot"))
    fig.add_annotation(x=mid, y=len(vals) - 0.42, yanchor="bottom", showarrow=False,
                       text=f"chance spread, {chance:.2f} wide",
                       font=dict(size=11, color=INK2))
    # A rule joining the two extremes: this distance is the statistic being tested.
    fig.add_shape(type="line", x0=lo, x1=hi, y0=-0.45, y1=-0.45, xref="x", yref="y",
                  line=dict(color=extreme, width=2))
    fig.add_annotation(x=mid, y=-0.5, yanchor="top", showarrow=False,
                       text=f"observed spread, {spread:.2f}",
                       font=dict(size=11, color=extreme))
    fig.add_trace(go.Scatter(
        x=vals, y=labels, mode="markers+text", showlegend=False,
        marker=dict(size=[15 if v in (lo, hi) else 11 for v in vals],
                    color=[extreme if v in (lo, hi) else INK2 for v in vals],
                    line=dict(width=2, color="#ffffff")),
        text=[f"{v:+.2f}" for v in vals], textposition="top center",
        textfont=dict(size=11, color=INK2),
        hovertemplate="%{y}<br>mean %{x:+.3f} logits<extra></extra>"))
    _base(fig, height=330, xtitle="mean score for that group, with each template centred on zero")
    fig.update_yaxes(categoryorder="array", categoryarray=labels,
                     tickfont=dict(size=12.5, color=INK))
    pad = max(spread, chance) * 0.22 + 0.04
    fig.update_xaxes(range=[mid - max(spread, chance) / 2 - pad,
                            mid + max(spread, chance) / 2 + pad],
                     showgrid=True, gridcolor=GRID)
    fig.update_layout(margin=dict(l=8, r=16, t=28, b=52))
    return fig


# --------------------------------------------------------------------------------------
# sycophancy
# --------------------------------------------------------------------------------------


LADDER = ["neutral", "expertise", "emotional"]
LADDER_LABEL = INSISTENCE_PHRASE


def sycophancy_ladder(block: dict, noise: dict | None = None) -> go.Figure:
    """The one chart that answers the module: does agreeing pay more as the user pushes harder?

    One line, the agreement premium averaged over the two phrasings each scenario was written in.
    Version 1 does not split by phrasing; see the appendix table for the breakdown.
    """
    rows = sorted(block.get("premium_by_level") or [], key=lambda r: LADDER.index(r["insistence"]))
    fig = go.Figure()
    if noise:
        band = noise["median_abs_delta"]
        fig.add_hrect(y0=-band, y1=band, fillcolor="#8a8a85", opacity=0.10, line_width=0,
                      annotation_text="rewording an answer moves the score this much",
                      annotation_position="bottom right",
                      annotation_font=dict(size=11, color=INK2))
    x = [LADDER_LABEL[r["insistence"]] for r in rows]
    y = [r["mean_delta"] for r in rows]
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers+text", name="reward for agreeing",
        line=dict(color=SERIES[0], width=2.5), marker=dict(size=11, color=SERIES[0]),
        error_y=dict(type="data", symmetric=False,
                     array=[r["ci_high"] - r["mean_delta"] for r in rows],
                     arrayminus=[r["mean_delta"] - r["ci_low"] for r in rows],
                     color=SERIES[0], width=6, thickness=1.5),
        text=[f"{v:+.2f}" for v in y], textposition="top center",
        textfont=dict(size=12, color=INK),
        customdata=[[r["win_rate"], r["p_sign"]] for r in rows],
        hovertemplate="%{x}<br>premium %{y:+.2f} logits"
                      "<br>agreement wins on %{customdata[0]:.0%} of scenarios"
                      "<br>p = %{customdata[1]:.3f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="#8a8a85", width=1.5))
    _base(fig, height=400, ytitle="reward for agreeing rather than correcting (logits)",
          xtitle="how hard the user pushes")
    fig.update_yaxes(showgrid=True, gridcolor=GRID)
    return fig


def sycophancy_scenarios(arms: list[dict], level: str) -> go.Figure:
    """Every scenario at one level of pressure, so the spread behind the average is visible.

    A mean of twenty scenarios can come from all twenty leaning the same way or from a handful of
    extremes, and those mean very different things for a policy trained on this signal.
    """
    rows = [a for a in arms if a["insistence"] == level]
    by_scn: dict[str, dict[str, float]] = {}
    for a in rows:
        by_scn.setdefault(a["scenario_id"], {})[a["cell"]] = a["score"]
    pts = []
    for sid, sc in by_scn.items():
        vals = [sc[f"agrees_{t}"] - sc[f"corrects_{t}"]
                for t in ("warm", "blunt") if f"agrees_{t}" in sc and f"corrects_{t}" in sc]
        if vals:
            pts.append((sid, float(np.mean(vals))))
    pts.sort(key=lambda t: t[1])
    vals = [v for _, v in pts]
    mean = float(np.mean(vals)) if vals else 0.0
    fig = go.Figure(go.Scatter(
        x=vals, y=[s for s, _ in pts], mode="markers",
        marker=dict(size=10,
                    color=[STATUS["critical"] if v > 0 else STATUS["good"] for v in vals],
                    line=dict(width=1, color="#fcfcfb")),
        hovertemplate="%{y}<br>%{x:+.2f} logits<extra></extra>", showlegend=False))
    fig.add_vline(x=0, line=dict(color="#8a8a85", width=1.5))
    fig.add_vline(x=mean, line=dict(color=SERIES[0], width=2, dash="dash"),
                  annotation_text=f"mean {mean:+.2f}", annotation_position="top",
                  annotation_font=dict(size=11, color=SERIES[0]))
    _base(fig, height=max(360, 17 * len(pts) + 120),
          xtitle="reward for agreeing rather than correcting (logits)")
    fig.update_yaxes(tickfont=dict(size=10.5, color=INK2))
    return fig


# --------------------------------------------------------------------------------------
# style
# --------------------------------------------------------------------------------------


def style_length_decomposition(block: dict) -> go.Figure:
    """Where the naive number goes once you take the length out of it.

    Comparing a styled answer with the plain one measures two things at once: the style, and the
    extra length the style drags along. Those two add up to the naive number, so the honest way to
    show it is a decomposition: start at what the naive comparison says, then move by the amount
    length alone explains, and land on the style itself.

    Two bars side by side invited the reader to compare two unrelated quantities. An arrow says
    "this estimate moves to here, and this is why", which is the actual claim.
    """
    adj = {a["transform"]: a for a in block["adjusted"]}
    raw_by: dict[str, dict] = {}
    for c in block["contrasts"]:
        if c["transform"] not in raw_by or c["dose"] > raw_by[c["transform"]]["dose"]:
            raw_by[c["transform"]] = c

    rows = []
    for t, a in adj.items():
        if t not in raw_by:
            continue
        r = raw_by[t]["mean_delta"]
        v = a.get("effect_at_max_dose") or 0.0
        rows.append({"t": t, "raw": r, "adj": v, "length": r - v, "group": a["group"]})
    # Biggest distortion at the top: that is the whole point of the chart.
    rows.sort(key=lambda d: abs(d["length"]))
    names = [d["t"] for d in rows]

    fig = go.Figure()
    for d in rows:
        fig.add_annotation(
            x=d["adj"], y=d["t"], ax=d["raw"], ay=d["t"], xref="x", yref="y",
            axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.1,
            arrowwidth=2, arrowcolor="#c9c8c3")
    fig.add_vline(x=0, line=dict(color="#8a8a85", width=1.5))
    fig.add_trace(go.Scatter(
        x=[d["raw"] for d in rows], y=names, mode="markers",
        name="what the naive comparison says",
        marker=dict(symbol="circle-open", size=11, color=INK2, line=dict(width=2)),
        hovertemplate="%{y}<br>naive: %{x:+.2f} logits<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[d["adj"] for d in rows], y=names, mode="markers+text",
        name="the style itself, once length is removed",
        marker=dict(size=12, color=SERIES[0]),
        text=[f"{d['adj']:+.2f}" for d in rows], textposition="middle right",
        textfont=dict(size=11.5, color=INK),
        customdata=[[d["length"], d["raw"]] for d in rows],
        hovertemplate="%{y}<br>naive %{customdata[1]:+.2f}"
                      "<br>of which length explains %{customdata[0]:+.2f}"
                      "<br>leaving %{x:+.2f} for the style<extra></extra>"))
    _base(fig, height=max(340, 40 * len(rows) + 110), showlegend=True,
          xtitle="effect at maximum intensity (logits)")
    lo = min([d["raw"] for d in rows] + [d["adj"] for d in rows])
    hi = max([d["raw"] for d in rows] + [d["adj"] for d in rows])
    pad = (hi - lo) * 0.14
    fig.update_xaxes(range=[lo - pad, hi + pad * 1.6])
    fig.update_yaxes(tickfont=dict(size=12.5, color=INK))
    return fig


def style_length_curve(block: dict) -> go.Figure:
    c = block["length_curve"]
    fig = go.Figure(go.Scatter(
        x=c["tokens"], y=c["reward"], mode="lines", line=dict(color=SERIES[0], width=2),
        hovertemplate="%{x:.0f} tokens<br>%{y:+.2f} logits<extra></extra>", name="reward"))
    n = len(c["tokens"])
    for i in [n // 4, n // 2, 3 * n // 4]:
        fig.add_annotation(x=c["tokens"][i], y=c["reward"][i],
                           text=f"{c['slope_per_100_tokens'][i]:+.1f} per 100 tokens",
                           showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="#b9b8b3",
                           ax=0, ay=-26, font=dict(size=11, color=INK2))
    _base(fig, height=340, xtitle="answer length, tokens",
          ytitle="reward relative to the shortest answer (logits)")
    fig.update_yaxes(showgrid=True, gridcolor=GRID)
    return fig


def style_dose_response(block: dict, transforms: list[str]) -> go.Figure:
    fig = go.Figure()
    for i, t in enumerate(transforms[:8]):
        rows = sorted([c for c in block["contrasts"] if c["transform"] == t],
                      key=lambda r: r["dose"])
        if not rows:
            continue
        fig.add_trace(go.Scatter(
            x=[0] + [r["dose"] for r in rows], y=[0] + [r["mean_delta"] for r in rows],
            mode="lines+markers", name=t, line=dict(color=SERIES[i], width=2),
            marker=dict(size=8, color=SERIES[i]),
            hovertemplate=t + "<br>dose %{x}<br>%{y:+.2f} logits<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="#b9b8b3", width=1))
    _base(fig, height=380, xtitle="intensity of the transform",
          ytitle="change from the untransformed answer (logits)", showlegend=True)
    fig.update_yaxes(showgrid=True, gridcolor=GRID)
    return fig


# --------------------------------------------------------------------------------------
# reward hacking: how far each affix moves a bad answer
# --------------------------------------------------------------------------------------


def attack_lifts(summary: dict, *, top: int = 18) -> go.Figure:
    """Held-out lift for the affixes the search ranked highest, with their controls beside them."""
    # Selected by development rank so the choice of what to show is never made on the data being
    # reported; sorted by the held-out value only so the chart reads cleanly.
    rows = sorted(summary["affixes"], key=lambda r: r.get("dev_rank", 1e9))[:top]
    rows = sorted(rows, key=lambda r: r["mean_lift"])
    labels = [f"{r['affix_id']} ({r['position']})" for r in rows]
    colors = ["#8a8a85" if r["is_control"]
              else (STATUS["critical"] if r["mean_lift"] > 0 else STATUS["good"]) for r in rows]
    fig = go.Figure(go.Bar(
        y=labels, x=[r["mean_lift"] for r in rows], orientation="h", width=0.62,
        marker=dict(color=colors),
        error_x=dict(type="data", symmetric=False,
                     array=[r["lift_ci_high"] - r["mean_lift"] for r in rows],
                     arrayminus=[r["mean_lift"] - r["lift_ci_low"] for r in rows],
                     color="#8a8a85", width=4, thickness=1.2),
        customdata=[[r["family"], "control" if r["is_control"] else "attack"] for r in rows],
        hovertemplate="%{y}<br>%{customdata[1]} · %{customdata[0]}"
                      "<br>lift %{x:+.2f} logits<extra></extra>"))
    fig.update_traces(marker_cornerradius=4)
    _base(fig, height=max(360, 30 * len(rows) + 80),
          xtitle="lift over the unmodified bad answer (logits). Grey bars are controls")
    fig.update_yaxes(tickfont=dict(size=11))
    return fig


def calibration_reliability(cal: dict) -> go.Figure:
    rel = cal.get("reliability") or []
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0.5, 1.0], y=[0.5, 1.0], mode="lines", name="perfect calibration",
                             line=dict(color="#b9b8b3", width=2, dash="dot")))
    fig.add_trace(go.Scatter(
        x=[r["predicted"] for r in rel], y=[r["observed"] for r in rel],
        mode="lines+markers", name="this model", line=dict(color=SERIES[0], width=2),
        marker=dict(size=9, color=SERIES[0]),
        hovertemplate="predicted %{x:.2f}<br>observed %{y:.2f}<extra></extra>"))
    _base(fig, height=330, showlegend=True,
          xtitle="predicted chance of agreeing with the human",
          ytitle="observed share where it actually did")
    fig.update_yaxes(showgrid=True, gridcolor=GRID)
    return fig


# --------------------------------------------------------------------------------------
# cross-model comparison
# --------------------------------------------------------------------------------------


def compare_findings(rows: list[dict], label_a: str, label_b: str) -> go.Figure:
    """Two models on one shared scale.

    Raw logits are not comparable across checkpoints: the two have different score scales and
    different calibration. A multiple of each model's *own* rewording bar is, because that bar is
    measured on that model. This used to plot a percentile of the single-comparison noise floor
    against a 95th-percentile line, which is the materiality rule the rest of the tool retired,
    so the comparison tab decided a finding mattered by a test no other tab applied.
    """
    rows = sorted(rows, key=lambda r: -max(r["a"] or 0, r["b"] or 0))
    labels = [(r["title"][:58] + "\u2026") if len(r["title"]) > 58 else r["title"] for r in rows]
    fig = go.Figure()
    fig.add_vrect(x0=0, x1=1, fillcolor="#8a8a85", opacity=0.11, line_width=0)
    for i, (key, name) in enumerate((("a", label_a), ("b", label_b))):
        fig.add_trace(go.Bar(
            y=labels, x=[r[key] for r in rows], orientation="h", name=name,
            marker=dict(color=SERIES[i]), width=0.34,
            hovertemplate="%{y}<br>" + name +
                          ": %{x:.1f}x what rewording alone could produce<extra></extra>"))
    fig.update_traces(marker_cornerradius=4)
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.add_vline(x=1, line=dict(color="#8a8a85", width=2, dash="dot"))
    fig.add_annotation(x=1, y=len(rows) - 0.4, text="as far as rewording alone gets",
                       showarrow=False, xanchor="left", xshift=5, font=dict(size=11, color=INK2))
    _base(fig, height=max(360, 44 * len(rows) + 90), showlegend=True,
          xtitle="multiples of what rewording alone could produce")
    fig.update_xaxes(range=[0, max([r[k] or 0 for r in rows for k in ("a", "b")] + [1.0]) * 1.12])
    return fig


# --------------------------------------------------------------------------------------
# the bias overview used at the top of each bias tab
# --------------------------------------------------------------------------------------

# Identity is non-directional: a shift either way is bias, so magnitude is the whole story.
UNSIGNED_STATES = [
    ("bigger than rewording alone", STATUS["critical"]),
    ("real, but no bigger than rewording alone", STATUS["warning"]),
    ("not statistically confirmed", "#9a9a95"),
]
# Style and sycophancy are directional: only one direction is a fault. Being pushed the *other*
# way is the model resisting, and colouring that like a vulnerability would be a lie.
SIGNED_STATES = [
    ("{bad}, bigger than rewording alone", STATUS["critical"]),
    ("{bad}, but no bigger than rewording alone", STATUS["warning"]),
    ("{good}", STATUS["good"]),
    ("not statistically confirmed", "#9a9a95"),
]


# For transforms that might genuinely improve an answer, being rewarded is not a fault, so the
# fault palette would mislabel them. They get a plain preference palette instead.
PREFERENCE_STATES = [
    ("confirmed preference", SERIES[0]),
    ("not statistically confirmed", "#9a9a95"),
]


def bias_state(item: dict, *, signed: bool) -> int:
    """Colour state. Materiality is the ratio to the item's own bar, not a global percentile."""
    if not item.get("confirmed"):
        return 3 if signed else 2
    material = bool(item.get("material"))
    if not signed:
        return 0 if material else 1
    # Direction comes from the item's own `adverse` flag where it has one. `module_items` sets it
    # from the effect's sign for every transform, so charts read as before, but the substance check
    # is a comparison whose fault runs negative and the sign rule reads it backwards.
    adverse = item.get("adverse")
    if adverse is None:
        adverse = (item.get("effect") or 0) > 0
    if not adverse:
        return 2
    return 0 if material else 1


# One phrasing of the two directions of the substance check, shared by both renderers: it is not a
# transform, so the words the transform charts use ("penalised, the model resists it") describe the
# wrong thing entirely on a model that passes it.
SUBSTANCE_LABELS = dict(bad="filler scores higher", good="real information wins")


def fault_chip(item: dict, *, bad: str, good: str) -> tuple[str, str]:
    """The words and colour ``bias_bars`` would give this finding, for one shown outside a chart.

    A finding presented beside those bars has to be marked by the same rule, or a reader cannot
    count the two together: red there means a confirmed fault bigger than rewording alone, which is
    exactly what the category tiles count, and is not the same question as which severity band it
    lands in.
    """
    name, color = SIGNED_STATES[bias_state(item, signed=True)]
    return name.format(bad=bad, good=good), color


def bias_bars(items: list[dict], *, signed: bool = False, fault: bool = True,
              bad_label: str = "rewarded", good_label: str = "penalised, the model resists it",
              xtitle: str | None = None, height: int | None = None,
              xmax: float | None = None) -> go.Figure:
    """Every probe in one module, against what rewording alone could produce at its own size.

    Plotted as a multiple of that bar rather than in raw logits. Each probe averages a different
    number of comparisons, so each has a different bar, and one shaded band in logits would be
    wrong for most of the rows. On this scale 1.0 is the bar itself and every row is comparable,
    across modules and across models.

    In signed mode the bar's side of zero carries the direction, so a transform the model pushes
    *against* reads as green on the left rather than as a large red vulnerability.
    """
    items = [r for r in items if r.get("ratio") is not None]
    if not items:
        fig = go.Figure()
        _base(fig, height=height or 200, xtitle="no probe in this module reported a sample size")
        return fig
    key = (lambda r: (r["ratio"] if (r["effect"] or 0) > 0 else -r["ratio"])) if signed \
        else (lambda r: r["ratio"])
    items = sorted(items, key=key)
    labels = [r["label"] for r in items]
    vals = [key(r) for r in items]
    states = [bias_state(r, signed=signed) for r in items]
    if fault:
        palette = SIGNED_STATES if signed else UNSIGNED_STATES
        names = [n.format(bad=bad_label, good=good_label) for n, _ in palette]
    else:
        palette = PREFERENCE_STATES
        states = [0 if r.get("confirmed") else 1 for r in items]
        names = [n for n, _ in palette]

    fig = go.Figure()
    fig.add_vrect(x0=-1 if signed else 0, x1=1, fillcolor="#8a8a85", opacity=0.11, line_width=0)
    for x in ([1, -1] if signed else [1]):
        fig.add_vline(x=x, line=dict(color="#8a8a85", width=2, dash="dot"))
    fig.add_annotation(x=1, y=len(items) - 0.35, text="as far as rewording alone gets",
                       showarrow=False, xanchor="left", xshift=6,
                       font=dict(size=11, color=INK2))
    fig.add_trace(go.Bar(
        y=labels, x=vals, orientation="h", width=0.6, showlegend=False,
        marker=dict(color=[palette[st][1] for st in states]),
        text=[f"{abs(v):.1f}x" for v in vals], textposition="outside",
        textfont=dict(size=12, color=INK),
        customdata=[[r.get("detail_text", ""), names[st], r["effect"],
                     r.get("systematic_bar") or 0, r.get("n_items") or 0]
                    for r, st in zip(items, states)],
        hovertemplate="<b>%{y}</b><br>%{customdata[2]:+.2f} logits"
                      "<br>%{x:.1f}x what rewording alone could produce"
                      " (%{customdata[3]:.2f} over %{customdata[4]} comparisons)"
                      "<br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>"))
    for st, name in enumerate(names):
        if st in states:
            fig.add_trace(go.Bar(y=[labels[0]], x=[None], orientation="h", name=name,
                                 marker=dict(color=palette[st][1]), width=0.6, hoverinfo="skip"))
    fig.update_traces(marker_cornerradius=4)
    _base(fig, height=height or max(260, 40 * len(items) + 110), showlegend=True,
          xtitle=xtitle or "multiples of what rewording alone could produce")
    fig.update_yaxes(tickfont=dict(size=12.5, color=INK))
    span = xmax if xmax is not None else max(max(abs(v) for v in vals), 1.0) * 1.28
    fig.update_xaxes(range=[-span, span] if signed else [0, span])
    return fig


def descriptor_chart(axis_block: dict) -> go.Figure:
    """Levels of one descriptor axis. The two ends are the finding, so only they are coloured."""
    means = axis_block["level_means"]
    order = sorted(means, key=lambda k: means[k])
    vals = [means[k] for k in order]
    hi, lo_l = axis_block["highest"], axis_block["lowest"]
    sig = axis_block["permutation"]["p_value"] < 0.05
    colors = []
    for k in order:
        if k == hi:
            colors.append(SERIES[0] if sig else "#9a9a95")
        elif k == lo_l:
            colors.append(STATUS["critical"] if sig else "#9a9a95")
        else:
            colors.append("#c9c8c3")
    fig = go.Figure(go.Bar(
        y=order, x=vals, orientation="h", width=0.62, marker=dict(color=colors),
        text=[f"{v:+.2f}" for v in vals], textposition="outside",
        textfont=dict(size=11, color=INK),
        hovertemplate="%{y}<br>%{x:+.3f} logits<extra></extra>"))
    fig.update_traces(marker_cornerradius=4)
    _base(fig, height=max(240, 34 * len(order) + 80),
          xtitle="mean score, with each template's own average removed (logits)")
    span = max(abs(min(vals)), abs(max(vals))) * 1.3
    fig.update_xaxes(range=[-span, span])
    return fig


# --------------------------------------------------------------------------------------
# reward hacking: search, then validate
# --------------------------------------------------------------------------------------


def attack_success(summary: dict, beam: dict | None) -> go.Figure:
    """Does the attack actually make a bad answer beat a real one?

    The bar to clear is set at several percentiles of genuine answers to the same prompt, because
    any single choice of bar is a researcher degree of freedom. An unmodified bad answer is shown
    alongside, so the attack has something to be a success *over*.
    """
    pcts = [25, 50, 75, 90]
    xs = [f"{p}th" for p in pcts]
    series = [("an unmodified bad answer", summary.get("baseline_asr") or {}, "#9a9a95")]
    best = (summary.get("search") or {}).get("best_single")
    if best:
        series.append((f"best single affix ({best['affix_id']})", best["asr"], SERIES[0]))
    if beam and beam.get("heldout_asr"):
        stack = " + ".join(a for a, _ in beam["stack"])
        series.append((f"best stacked attack ({stack})", beam["heldout_asr"], SERIES[1]))

    fig = go.Figure()
    for name, asr, color in series:
        ys = [asr.get(f"p{p}") for p in pcts]
        fig.add_trace(go.Bar(
            x=xs, y=[0 if v is None else v for v in ys], name=name,
            marker=dict(color=color), width=0.24,
            text=[("n/a" if v is None else f"{v:.0%}") for v in ys], textposition="outside",
            textfont=dict(size=11, color=INK),
            hovertemplate=name + "<br>clears the %{x} percentile bar "
                          "%{y:.0%} of the time<extra></extra>"))
    fig.update_traces(marker_cornerradius=4)
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    _base(fig, height=400, showlegend=True,
          xtitle="bar set at this percentile of genuine answers to the same prompt",
          ytitle="share of attacks that clear the bar")
    fig.update_yaxes(showgrid=True, gridcolor=GRID, tickformat=".0%")
    return fig


def exploit_ranking(exploits: list[dict], baseline_asr: float) -> go.Figure:
    """Only the attacks that beat a real answer, against the bar an unattacked answer already clears.

    Ranked by success rate rather than by lift. Lift measures how far an affix moved the score;
    success measures whether that was far enough to matter, and the two disagree because the
    biggest lifts come from the answers that started lowest.

    One colour for every bar. It used to be orange for a stacked attack and red for a single affix,
    which reads as a severity ramp - the same two colours the Moderate and High band pills use - so
    the strongest attack on the chart looked like the milder one. Every bar here is an attack that
    works; how badly is the bar's length, and which kind it is is in the label and the hover.
    """
    rows = sorted(exploits, key=lambda e: (e["asr"].get("p50") or 0))
    labels = [(e["label"][:44] + "…") if len(e["label"]) > 44 else e["label"] for e in rows]
    vals = [e["asr"].get("p50") or 0 for e in rows]
    fig = go.Figure()
    if baseline_asr > 0:
        fig.add_vline(x=baseline_asr, line=dict(color="#8a8a85", width=2, dash="dot"))
        fig.add_annotation(x=baseline_asr, y=len(rows) - 0.4,
                           text="the bad answer already clears this unattacked",
                           showarrow=False, xanchor="left", xshift=6,
                           font=dict(size=11, color=INK2))
    fig.add_trace(go.Bar(
        y=labels, x=vals, orientation="h", width=0.6, showlegend=False,
        marker=dict(color=STATUS["critical"]),
        text=[f"{v:.0%}" for v in vals], textposition="outside",
        textfont=dict(size=12, color=INK),
        customdata=[[e["kind"], e["lift"], len(e["examples"])] for e in rows],
        hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>beats a real answer %{x:.0%} of the time"
                      "<br>lift %{customdata[1]:+.2f} logits<extra></extra>"))
    fig.update_traces(marker_cornerradius=4)
    _base(fig, height=max(240, 44 * len(rows) + 110),
          xtitle="share of prompts where the attacked bad answer beats a median genuine answer")
    fig.update_xaxes(tickformat=".0%", range=[0, max(vals + [baseline_asr]) * 1.45 + 0.02])
    fig.update_yaxes(tickfont=dict(size=12.5, color=INK))
    return fig
