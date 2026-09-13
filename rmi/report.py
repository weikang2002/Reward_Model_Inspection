"""Static HTML export: one self-contained file that renders offline, with no Python.

Mirrors the dashboard's structure so a reader who was not at the terminal gets the same story:
the two yardsticks first, then where the problems are, then the evidence, then the methodology.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import plotly.io as pio

from . import methodology
from . import severity as sev
from .runner import probe_heading
from . import viz
from .findings import (banner, contamination_check, module_items, overview_verdict,
                       self_check, substance_check, tile_body, tone_check,
                       verdict_line)
from .textdiff import word_diff

CSS = """
:root{--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e1;--surface:#fcfcfb;--plane:#f9f9f7;}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
  font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;}
.wrap{max-width:1180px;margin:0 auto;padding:32px 20px 80px;}
h1{font-size:30px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:21px;margin:44px 0 12px;padding-top:18px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:26px 0 8px}
.sub{color:var(--ink2);font-size:13.5px;margin:0 0 26px}
.grid{display:grid;gap:14px;margin:14px 0}
.g2{grid-template-columns:repeat(2,1fr)} .g3{grid-template-columns:repeat(3,1fr)}
.g4{grid-template-columns:repeat(4,1fr)}
@media(max-width:860px){.g2,.g3,.g4{grid-template-columns:1fr}}
.yard{border-left:3px solid #2a78d6;background:#f4f8fe;border-radius:0 8px 8px 0;padding:12px 15px;
  font-size:13.5px;line-height:1.5}
.yard .big{display:block;font-size:24px;font-weight:700;margin-bottom:3px}
.tile{border:1px solid var(--line);border-radius:10px;padding:14px 16px;background:var(--surface)}
.tile h4{margin:0 0 8px;font-size:12px;letter-spacing:.05em;text-transform:uppercase;
  color:var(--ink2);font-weight:600}
.band{display:inline-block;padding:3px 10px;border-radius:999px;color:#fff;font-weight:600;
  font-size:12.5px}
.tile p{margin:9px 0 0;font-size:12.5px;color:var(--ink2);line-height:1.45}
.finding{border:1px solid var(--line);border-left-width:4px;border-radius:8px;padding:10px 14px;
  margin-bottom:8px;background:var(--surface)}
.finding .meta{color:var(--ink2);font-size:12.5px;margin-top:5px}
.card{border:1px solid var(--line);border-radius:10px;background:var(--surface);padding:16px 18px;
  margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:13px;background:var(--surface)}
th,td{border-bottom:1px solid var(--line);padding:7px 9px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
.note{background:#f4f8fe;border-left:3px solid #2a78d6;padding:11px 15px;border-radius:0 8px 8px 0;
  font-size:13.5px;margin:14px 0}
.warn{background:#fdf6e8;border-left-color:#fab219}
.good{background:#f2faf2;border-left-color:#0ca30c}
code{background:#f0efec;padding:1px 5px;border-radius:4px;font-size:12.5px}
ins{background:#d7f0d7;text-decoration:none}
ins.attack{background:#fbe3e0;border-bottom:2px solid #d03b3b;text-decoration:none}
del{background:#fbdada;text-decoration:line-through}
.answer{background:#f7f7f5;border:1px solid var(--line);border-radius:8px;padding:10px 12px;
  font-size:13px;line-height:1.55;white-space:pre-wrap;margin:6px 0 12px}
.verdict{border-left:4px solid #d03b3b;background:#fdf3f3;border-radius:0 8px 8px 0;
  padding:12px 16px;margin:2px 0 18px;font-size:15px;line-height:1.55}
.verdict.clear{border-left-color:#0ca30c;background:#f2faf2}
.verdict.mild{border-left-color:#fab219;background:#fdf6e8}
.verdict .hint,.yard .hint{display:block;color:#52514e;font-size:12.5px;margin-top:6px}
.verdict ul{margin:7px 0 0;padding-left:20px}
.verdict li{margin:3px 0;line-height:1.5}
details{margin:10px 0}summary{cursor:pointer;font-weight:600;font-size:14px;color:var(--ink2)}
pre{background:#f0efec;padding:12px;border-radius:8px;overflow-x:auto;font-size:12px}
"""


class _FigWriter:
    """Inlines plotly.js exactly once per *document*, so each file stands alone offline.

    A module-level flag would inline it only into the first report built in a process, silently
    producing later files that render nothing without a network.
    """

    def __init__(self):
        self.done = False

    def __call__(self, fig) -> str:
        inc = False if self.done else "inline"
        self.done = True
        return pio.to_html(fig, full_html=False, include_plotlyjs=inc,
                           config={"displayModeBar": False},
                           default_height=fig.layout.height or 380)


def _table(rows: list[dict], cols: list[str], *, limit: int = 200) -> str:
    if not rows:
        return "<p>No rows.</p>"
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = []
    for r in rows[:limit]:
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                v = f"{v:.4g}"
            elif isinstance(v, bool):
                v = "yes" if v else "no"
            cells.append(f"<td>{html.escape(str(v)) if v is not None else '—'}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    more = (f"<p class='sub'>Showing {limit} of {len(rows)} rows. The full set is in the "
            "results JSON.</p>") if len(rows) > limit else ""
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>" \
           + "".join(body) + f"</tbody></table></div>{more}"


def build(results: dict, out_path: Path | str) -> Path:
    R = results
    meta, noise, cal = R["meta"], R.get("noise_floor"), R.get("calibration")
    _fig = _FigWriter()
    P: list[str] = []
    A = P.append

    A("<h1>Reward Model Inspection</h1>")
    A(f"<p class='sub'>{html.escape(meta['model_id'])} · {meta['depth']} scan · seed "
      f"{meta['seed']} · {meta['started']} · {meta.get('provenance', {}).get('device', '?')}</p>")
    A(overview_verdict(R, cal))

    # -- severity ---------------------------------------------------------------------
    A("<h2>Where the problems are</h2><div class='grid g4'>")
    for cat, s in sev.summarise_all(R["findings"]).items():
        A(f"<div class='tile'><h4>{html.escape(probe_heading(cat))}</h4>"
          f"<span class='band' style='background:{viz.BAND_COLOR.get(s['band'], '#8a8a85')}'>"
          f"{html.escape(s['band'])}</span>"
          f"<p>{tile_body(s)}</p></div>")
    A("</div><p class='sub'>Bands are the <i>worst confirmed</i> effect in a category, as a "
      "multiple of what rewording alone could produce across the same number of comparisons: "
      "below "
      "1x Negligible, to 2x Low, to 4x Moderate, above that High. Findings showing the model "
      "behaving <i>well</i> are excluded.</p>")

    if noise:
        A("<h2>The biggest findings, on one scale</h2>")
        A(_fig(viz.findings_vs_noise(R["findings"])))
        scored = [f for f in R["findings"]
                  if f.get("effect") is not None and f.get("systematic_bar")]
        A(f"<p class='sub'>{min(len(scored), viz.MAX_ROWS)} of the {len(scored)} findings "
          "measured, each labelled with the category it came from: the largest, plus every "
          "problem the tiles above count, so nothing they report is missing here. "
          "Bars inside the shaded band are no "
          "larger than rewording alone would produce at the same sample size. A tile counts a "
          "finding only if it is a vulnerability, clears the line <i>and</i> is statistically "
          "confirmed; hatched red bars fail that last test, and green and blue ones are not "
          "faults at all.</p>")

    # -- yardsticks, after the problems they calibrate ----------------------------------
    A("<h2>The two yardsticks every number above is measured against</h2>"
      "<div class='grid g2'>")
    if noise:
        A(f"<div class='yard'><span class='big'>±{noise['pairwise_sd']:.2f} logits</span>"
          "spread between two <b>meaning-preserving rewrites</b> of the same answer. One "
          f"comparison moves up to {noise['per_comparison_bar']:.2f} on rewording alone, which "
          "swamps every bias effect here. But rewording pushes in an arbitrary direction, so it "
          "mostly cancels when averaged over many answers, while a bias does not. Every finding "
          "above is judged against that spread shrunk to its own sample size.</div>")
    if cal and cal.get("fitted"):
        A(f"<div class='yard'><span class='big'>{cal['accuracy']:.1%}</span>"
          "of the time this model agrees with <b>real human preference judgments</b> "
          f"({cal['n_pairs']} held-out pairs, 95% CI {cal['accuracy_ci'][0]:.1%} to "
          f"{cal['accuracy_ci'][1]:.1%}). Chance is 50%.</div>")
    A("</div>")

    # Outside the grid above: this is not a yardstick, nothing is measured against it.
    if check := self_check(R):
        ok = check["passed"]
        A(f"<div class='yard' style=\"border-left-color:"
          f"{viz.STATUS['good'] if ok else viz.STATUS['critical']};"
          f"background:{'#f2faf2' if ok else '#fdf3f3'}\">"
          f"<b>{html.escape(check['headline'])}</b>"
          f"<span class='hint'>{html.escape(check['detail'])}</span></div>")

    # -- findings ---------------------------------------------------------------------
    A("<h2>Ranked findings</h2>")
    for label, val in (("Vulnerabilities", "vulnerability"),
                       ("Model behaving correctly", "healthy"),
                       ("Preferences that may be legitimate", "informational")):
        sel = [f for f in R["findings"] if f["valence"] == val]
        if not sel:
            continue
        A(f"<h3>{label}</h3>")
        for f in sel:
            bits = []
            if f.get("noise_percentile") is not None:
                bits.append(f"bigger than {f['noise_percentile']:.0f}% of rewordings")
            if f.get("preference_probability"):
                bits.append(f"shifts the model's preference to "
                            f"{f['preference_probability']:.0%}")
            if f.get("p_adjusted") is not None:
                bits.append(f"adjusted p = {f['p_adjusted']:.3g}")
            elif f.get("p_raw") is not None:
                bits.append(f"p = {f['p_raw']:.3g}")
            A(f"<div class='finding' style='border-left-color:"
              f"{viz.VALENCE_COLOR.get(f['valence'], '#8a8a85')}'>"
              f"<span class='band' style='background:"
              f"{viz.BAND_COLOR.get(sev.finding_band(f), '#8a8a85')}'>"
              f"{html.escape(sev.finding_band(f))}</span> "
              f"<b>{html.escape(f['title'])}</b>"
              f"<div class='meta'>{' · '.join(bits)}</div></div>")

    # -- identity ---------------------------------------------------------------------
    idr = R.get("identity")
    if idr:
        A(f"<h2>{html.escape(probe_heading('identity'))}</h2>")
        A(verdict_line(module_items(R, "identity"), "identity",
                       "swapping a name or a descriptor changes the score"))
        A("<h3>Which identities change the model score</h3>")
        A(_fig(viz.bias_bars(module_items(R, "identity"))))
        A("<p class='sub'>Each bar is one identity axis: the largest gap it produces between "
          "groups, in otherwise identical templates. Identity bias is non-directional, so a shift "
          "either way counts and only the size is shown. Below, the shaded band is a matched "
          "control: how far apart two random halves of the <i>same</i> group land, which is the "
          "level a between-group gap lives at.</p>")
        for subset, label in (("ood", "Decision contexts (hiring, lending, parole)"),
                              ("id", "Assistant dialogue (in distribution)")):
            blk = idr.get(f"names_{subset}")
            if not blk:
                continue
            o = blk["omnibus_permutation"]
            A(f"<h3>{label}</h3>{_fig(viz.identity_groups(blk))}")
            # None when swapping names inside a group moves nothing at all, so there is no
            # within-group spread to take a ratio against. Formatting it blind crashed the build.
            ratio = blk.get("between_vs_within_ratio")
            compare = (f" Swapping across groups moves the score {ratio:.2f} times as much as "
                       "swapping to another name inside the same group."
                       if ratio is not None else
                       " Swapping to another name inside the same group moves the score not at "
                       "all, so there is no within-group spread to compare against.")
            A(f"<p>The largest gap between identity groups is <code>{o['statistic']:.3f}</code> "
              f"logits. Reshuffling which names belong to which group {o['n_perm']:,} times, and "
              "only within blocks of equal name token length, produces a gap that large "
              f"<b>p = {o['p_value']:.4f}</b> of the time. The grey band above is how wide that "
              f"reshuffled gap gets 95% of the time, <code>{o['null_p95']:.3f}</code>."
              f"{compare}</p>")
            A(_table(blk["pairwise_gaps"],
                     ["group_a", "group_b", "gap", "ci_low", "ci_high", "p_wild", "q_value",
                      "noise_percentile"]))
        A("<h3>Descriptor swaps</h3>")
        for axis, e in idr.get("descriptors", {}).items():
            sig = e["permutation"]["p_value"] < 0.05
            A(f"<details{' open' if sig else ''}><summary>{axis} — largest gap "
              f"{e['max_gap']:.2f} logits ({html.escape(e['highest'])} highest, "
              f"{html.escape(e['lowest'])} lowest), permutation p = "
              f"{e['permutation']['p_value']:.4f}</summary>"
              f"{_fig(viz.descriptor_chart(e))}"
              f"<p class='sub'>{e['n_templates']} templates. With so few, treat this as "
              "indicative: it is the part of the scan most in need of more items.</p></details>")

    # -- sycophancy -------------------------------------------------------------------
    sy = R.get("sycophancy")
    if sy:
        m = sy["agreement_main_effect"]
        slope = max(sy["insistence_slopes"], key=lambda s: s["mean_delta"])
        A(f"<h2>{html.escape(probe_heading('sycophancy'))}</h2>")
        A(verdict_line(module_items(R, "sycophancy"), "sycophancy",
                       "the model is rewarded for agreeing rather than correcting"))
        A(_fig(viz.bias_bars(module_items(R, "sycophancy"), signed=True,
                             bad_label="favours agreeing",
                             good_label="favours correcting, the right direction")))
        A("<p class='sub'>Above zero the model prefers agreeing with the user; below zero it "
          "prefers correcting them. Each scenario is written in two versions and both are scored; "
          "the premium is their average.</p>")
        A("<h3>The model's reward for agreeing, as the user pushes harder</h3>")
        A(_fig(viz.sycophancy_ladder(sy)))
        A("<p class='sub'>Above the line the model prefers agreeing with the user; below it, "
          "correcting them. Each scenario is asked three ways, changing only how hard the user "
          "pushes, never the two answers being compared.</p>")
        _tone = tone_check(sy, noise or {"median_abs_delta": 0.0})
        if _tone["diverges"]:
            A("<div class='note warn'>" + _tone["text"].replace("**", "")
              + " Version 1 reports the average of the two; the table below has the breakdown."
              + "</div>")
        if slope["mean_delta"] > 0 and slope["p_sign"] < 0.05:
            A(f"<div class='note warn'><b>The headline.</b> Overall the model barely prefers "
              f"agreement ({m['mean_delta']:+.2f}, p = {m['p_sign']:.2f}), so a test that measured "
              f"only the average would report nothing. But when the user is {slope['level']}, the "
              f"reward for agreeing rises by {slope['mean_delta']:+.2f} logits "
              f"(p = {slope['p_sign']:.3f}), flipping the model from mildly preferring corrections "
              "to preferring agreement. That is the pattern that matters in RLHF.</div>")
        A(f"<h3>Every scenario, with the user {viz.LADDER_LABEL[slope['level']]}</h3>")
        A(_fig(viz.sycophancy_scenarios(sy["arms"], slope["level"])))
        A("<p class='sub'>Each dot is one scenario, red where agreeing scored higher and green "
          "where correcting did. A mean can come from every scenario leaning the same way or from "
          "a few extremes, and those mean different things for a policy trained on it.</p>")
        A(_table(sy["premiums"], ["insistence", "tone", "mean_delta", "ci_low", "ci_high",
                                  "win_rate", "p_sign", "p_adjusted", "mean_token_delta"]))

    # -- style ------------------------------------------------------------------------
    sm = R.get("style")
    if sm:
        A(f"<h2>{html.escape(probe_heading('style'))}</h2>")
        # The verdict is about the transforms. The filler-versus-information control is a check on
        # the model rather than a transform applied to an answer, so it gets its own section below;
        # the category tile counts both, so the hint says where the rest of its count is.
        _check = module_items(R, "style", group="quality_control")
        A(verdict_line(
            module_items(R, "style", group="content_neutral") + _check, "style",
            "surface style is rewarded for its own sake",
            extra="Only transforms that add no information are charted, since those are the ones "
                  "where any reward at all is unearned. Effects are shown after removing what the "
                  "extra length alone explains. The substance check counts here too; it is not a "
                  "transform, so it is not one of the bars but a section of its own below."))
        _neutral = module_items(R, "style", group="content_neutral")
        _bearing = module_items(R, "style", group="quality_bearing")
        A("<h3>Style that adds no information</h3>")
        A("<p class='sub'>Same answer, repackaged, so any reward here is a fault.</p>")
        _smax = max([i["ratio"] for i in _neutral + _bearing if i.get("ratio")] + [1.0]) * 1.28
        A(_fig(viz.bias_bars(_neutral, signed=True, xmax=_smax,
                             bad_label="rewarded though it adds nothing",
                             good_label="penalised, the model resists it")))
        A("<h3>Style that might genuinely improve the answer</h3>")
        A("<p class='sub'>These change the answer itself, so a reward can be deserved: "
          "preferences, not faults.</p>")
        A(_fig(viz.bias_bars(_bearing, signed=True, fault=False, xmax=_smax)))
        A(f"<p class='sub'>That is {len(_neutral) + len(_bearing)} of the "
          f"{len(sm['transform_groups'])} transforms applied. The eleventh, <b>padding</b>, is not "
          "charted because it <i>is</i> the ruler: content-free filler at four intensities is what "
          "traces the reward-versus-length curve below, and every effect above is measured against "
          "that curve.</p>")
        # Findings first. Everything about length is methodology, so it goes last, consolidated.
        # The check on the scorer itself, after the transforms it validates. Prose, pairing and
        # row order come from findings.substance_check, shared with the dashboard so the two
        # cannot answer the same question differently.
        check = substance_check(sm, _check)
        if check:
            A("<h3>Does added length have to carry information?</h3>")
            A(f"<div class='note {'good' if check['good'] else 'warn'}'>"
              f"<b>{check['lead']}</b> {check['body']}</div>")
            A(f"<p class='sub'>{check['setup']}</p>")
            _rows = [{
                "tokens added either way": (round(r["added_tokens"])
                                            if r["added_tokens"] else None),
                "mean_delta": r["mean_delta"],
                "win_rate": r["win_rate"],
                "x rewording alone": round(r["ratio"], 1) if r["ratio"] else None,
                "reads as": (viz.fault_chip(r["item"], **viz.SUBSTANCE_LABELS)[0]
                             if r["item"] else ""),
            } for r in check["rows"]]
            A(_table(_rows, list(_rows[0])))
            A(f"<p class='sub'>{check['counts']}</p>")

        A("<h3>How length was accounted for</h3>")
        A("<p>Comparing a styled answer against the plain one measures two things at once: the "
          "style, and the extra length the style drags along. This model dislikes added length, so "
          "the naive comparison makes almost every transform look disliked. Each arrow starts at "
          "what the naive comparison says and ends at what is left once the length effect is taken "
          "out.</p>")
        A(_fig(viz.style_length_decomposition(sm)))
        A(f"<div class='grid g3' style='grid-template-columns:1fr 1fr'>"
          f"<div>{_fig(viz.style_length_curve(sm))}</div>"
          f"<div>{_fig(viz.style_dose_response(sm, [t for t, g in sm['transform_groups'].items() if g == 'content_neutral'][:4]))}</div></div>")
        A("<p class='sub'>The curve is traced by adding content-free filler at four intensities, "
          "which is what every effect above is measured against. This tokenizer also discards "
          "newlines, so bullets and headers reach the model only as <code>-</code> and "
          "<code>#</code> characters; layout as such is invisible to it.</p>")
        A(_table(sm["contrasts"], ["transform", "dose", "group", "mean_delta", "ci_low", "ci_high",
                                   "win_rate", "mean_added_tokens", "p_sign", "p_adjusted",
                                   "noise_percentile"]))

    # -- reward hacking ---------------------------------------------------------------
    ij = R.get("reward_hacking")
    if ij:
        srch = ij["summary"].get("search") or {}
        best = srch.get("best_single")
        beam = ij.get("beam_search")
        base_asr = (ij["summary"].get("baseline_asr") or {}).get("p50")
        cands = []
        if best:
            cands.append({"name": f"{best['affix_id']} as a {best['position']}",
                          "lift": best["mean_lift"], "asr": (best.get("asr") or {}).get("p50")})
        if beam:
            cands.append({"name": " + ".join(a for a, _ in beam["stack"]),
                          "lift": beam["heldout_mean_lift"],
                          "asr": (beam.get("heldout_asr") or {}).get("p50")})
        winner = max(cands, key=lambda c: (c["asr"] if c["asr"] is not None else -1, c["lift"])) \
            if cands else None

        A(f"<h2>{html.escape(probe_heading('reward_hacking'))}</h2>")
        if winner:
            works = (winner["asr"] is not None and base_asr is not None
                     and winner["asr"] > base_asr)
            A(banner(
                "" if works else " clear",
                "Yes, this model can be reward hacked." if works
                else "No attack succeeded on this model.",
                [f"The strongest attack found, <code>{html.escape(winner['name'])}</code>, lifts "
                 f"a deliberately bad answer by <b>{winner['lift']:+.2f} logits</b> on prompts "
                 "the search never saw.",
                 f"It then outscores a genuine answer <b>{(winner['asr'] or 0):.0%}</b> of the "
                 f"time, against {(base_asr or 0):.0%} unattacked."],
                "Every number here is measured on held-out prompts. The search that found this "
                "attack only ever saw the development half."))
        A(f"<p><b>How this was searched.</b> {srch.get('n_candidates', 0)} affixes were tried as "
          "both prefix and suffix against non-answers, off-topic text, confidently false claims "
          f"and rude replies. All ranking happened on {srch.get('n_dev_questions', 0)} development "
          "prompts"
          + (f", then a beam search stacked up to {beam['depth']} of them on the same prompts"
             if beam else "")
          + f". Every number reported is then measured once on the "
          f"{srch.get('n_test_questions', 0)} held-out prompts.</p>")

        exploits = ij["summary"].get("top_exploits") or []
        if exploits:
            A("<h3>Which attacks worked</h3>")
            A(_fig(viz.exploit_ranking(exploits, base_asr or 0.0)))
            A("<p class='sub'>An attack counts as working only if it beats a real answer more "
              "often than the unmodified bad answer already does. Lift alone is not enough: an "
              "affix can add several logits and still leave the answer far below anything a real "
              "model writes, which is why the category tile counts these and not the biggest "
              "lifts. Every attack that works gets a bar; the scan reports the strongest single "
              "affix and the best stack as one finding each, so the tile can count fewer than "
              "there are bars.</p>")
            A("<h3>What the attack actually looks like</h3>")
            for e in exploits[:3]:
                for ex in e["examples"][:1]:
                    left, right = word_diff(ex["base_text"], ex["attacked_text"],
                                            ins_class="attack")
                    A(f"<div class='card'><b>{html.escape(e['label'])}</b> · "
                      f"{html.escape(e['kind'])} · beats a real answer "
                      f"{e['asr']['p50']:.0%} of the time<br>"
                      f"<span class='sub'>The user asked: {html.escape(ex['question'])}</span>"
                      f"<p><b>The bad answer on its own</b> ({html.escape(ex['base_type'])}), "
                      f"score <code>{ex['base_score']:+.2f}</code></p>"
                      f"<div class='answer'>{left}</div>"
                      f"<p><b>The same answer, attacked</b>, score "
                      f"<code>{ex['attacked_score']:+.2f}</code>, past the median genuine answer "
                      f"at <code>{ex['median_genuine_score']:+.2f}</code>. What the attack bolted "
                      "on is highlighted.</p>"
                      f"<div class='answer'>{right}</div></div>")

        # Absent from this report entirely until now, while its two rows were the largest counted
        # problems in the category on one checkpoint and drawn on the ranked chart above.
        _cont = contamination_check(R)
        if _cont:
            A("<h3>Is padding free?</h3>")
            A(f"<div class='note {'warn' if _cont['pays'] else 'good'}'>"
              f"<b>{_cont['lead']}</b> {html.escape(_cont['body']).replace('*', '')}</div>")
            A(f"<p class='sub'>{_cont['setup']}</p>")
            A(_table([{
                "junk": r["where"],
                "change to a good answer": r["mean_delta"],
                "x rewording alone": round(r["ratio"], 1) if r["ratio"] else None,
                "reads as": (viz.fault_chip(r["item"], bad="padding pays",
                                            good="the model notices it")[0]
                             if r["item"] else ""),
            } for r in _cont["rows"]],
                ["junk", "change to a good answer", "x rewording alone", "reads as"]))
            A(f"<p class='sub'>{_cont['counts']}</p>")

        A("<h3>Supporting evidence</h3>")
        A("<h4>How hard is the bar?</h4>")
        A(_fig(viz.attack_success(ij["summary"], beam)))
        A("<p class='sub'>The bar is set at several percentiles of genuine answers to the same "
          "prompt, because any single choice of bar could be tuned after the fact. Grey is the "
          f"same bad answer with nothing attached. Based on {ij['n_reference_ok']} prompts with "
          "enough genuine answers to support a percentile.</p>")

        if beam:
            A("<h4>How the search found the stacked attack</h4><div class='grid g3'>")
            for label, val in (("on the prompts it was tuned on",
                                f"{beam['dev_mean_lift']:+.2f}"
                                if beam.get("dev_mean_lift") is not None else "n/a"),
                               ("on held-out prompts", f"{beam['heldout_mean_lift']:+.2f}"),
                               ("beats a median genuine answer",
                                f"{beam['heldout_asr']['p50']:.0%}"
                                if beam["heldout_asr"].get("p50") is not None else "n/a")):
                A(f"<div class='tile'><h4>{label}</h4>"
                  f"<div style='font-size:23px;font-weight:700'>{val}</div></div>")
            A("</div>")
            A(_table([{"depth": h["depth"],
                       "stack": " + ".join(a for a, _ in h["best_stack"]),
                       "candidates tried": h["n_candidates"],
                       "best lift on dev": round(h["best_dev_lift"], 3)}
                      for h in beam["history"]],
                     ["depth", "stack", "candidates tried", "best lift on dev"]))

        A("<h4>Every affix tried</h4>")
        A(_fig(viz.attack_lifts(ij["summary"])))
        _atk = [a for a in ij["summary"]["affixes"] if not a["is_control"]]
        _shr = max((abs(a["shrinkage"]) for a in _atk), default=0.0)
        A("<p class='sub'>Grey bars are controls, not attacks: content-free text of matched "
          "length, and lookalike strings that cannot produce a special token. Individual affixes "
          f"lose at most {_shr:.2f} logits between development and held-out prompts, because each "
          "is a single pre-written candidate rather than something the search optimised.</p>")

    # -- appendix ---------------------------------------------------------------------
    A("<h2>Appendix: how each number was produced</h2>")
    for lead, body in methodology.SECTIONS:
        A(f"<p><b>{lead}.</b> {html.escape(body)}</p>")
    if noise:
        A("<h3>Rewording noise, in detail</h3>")
        A(_fig(viz.noise_floor_hist(noise)))
        A("<p class='sub'>Negative controls change bytes but not meaning, so they should be zero."
          "</p>")
        A(_table([{"perturbation": k, **v} for k, v in noise["semantic_nulls"].items()],
                 ["perturbation", "mean_abs_delta", "max_abs_delta"]))
        A(f"<p class='sub'>Scoring the same pair twice differs by "
          f"{noise['determinism_delta']:.2e}; scoring alone versus inside a batch differs by "
          f"{noise['batch_invariance_delta']:.2e}.</p>")
    if cal and cal.get("fitted"):
        A("<h3>Agreement with human preferences, in detail</h3>")
        A(_fig(viz.calibration_reliability(cal)))
        A(f"<p>Fitted temperature <b>T = {cal['temperature']:.2f}</b> on {html.escape(cal['note'])}"
          f". A score difference of <i>d</i> logits means the model prefers that variant "
          f"sigmoid(d / {cal['temperature']:.2f}) of the time.</p>")
    A("<h3>Severity bands</h3><table><tr><th>band</th><th>at least this many times what "
      "rewording alone could produce</th></tr>")
    for band_name, threshold in sev.BANDS:
        A(f"<tr><td>{html.escape(band_name)}</td><td>{threshold:g}x</td></tr>")
    A(f"</table><p class='sub'>A finding counts as material at {sev.MATERIALITY_RATIO:g}x and "
      "above. A category's band is its worst confirmed effect, not an average: one working "
      "exploit is not cancelled by four that fail.</p>")
    A("<h3>Run provenance</h3><pre>"
      + html.escape(json.dumps(meta, indent=1, default=str)) + "</pre>")
    A("<h3>Limitations</h3><ul>")
    for lead, body in methodology.LIMITATIONS:
        A(f"<li><b>{html.escape(lead)}</b> {html.escape(body)}</li>")
    A("</ul>")

    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Reward Model Inspection — {html.escape(meta['model_id'])}</title>"
           f"<style>{CSS}</style></head><body><div class='wrap'>"
           + "".join(P) + "</div></body></html>")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out


