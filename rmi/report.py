"""Static HTML export: one self-contained file that renders offline, with no Python.

Mirrors the dashboard's structure so a reader who was not at the terminal gets the same story:
the two yardsticks first, then where the problems are, then the evidence, then the methodology.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import plotly.io as pio

from . import severity as sev
from . import viz
from .findings import module_items, overview_verdict, verdict_line

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
.g3{grid-template-columns:repeat(3,1fr)} .g4{grid-template-columns:repeat(4,1fr)}
@media(max-width:860px){.g3,.g4{grid-template-columns:1fr}}
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
.verdict{border-left:4px solid #d03b3b;background:#fdf3f3;border-radius:0 8px 8px 0;
  padding:12px 16px;margin:2px 0 18px;font-size:15px;line-height:1.55}
.verdict.clear{border-left-color:#0ca30c;background:#f2faf2}
.verdict.mild{border-left-color:#fab219;background:#fdf6e8}
.verdict .hint{display:block;color:#52514e;font-size:12.5px;margin-top:6px}
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


def _ordinal(n) -> str:
    i = int(round(n))
    suf = "th" if 11 <= i % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(i % 10, "th")
    return f"{i}{suf}"


def build(results: dict, out_path: Path | str) -> Path:
    R = results
    meta, noise, cal = R["meta"], R.get("noise_floor"), R.get("calibration")
    _fig = _FigWriter()
    P: list[str] = []
    A = P.append

    A(f"<h1>Reward Model Inspection</h1>")
    A(f"<p class='sub'>{html.escape(meta['model_id'])} · {meta['depth']} scan · seed "
      f"{meta['seed']} · {meta['started']} · {meta.get('provenance', {}).get('device', '?')}</p>")
    A(overview_verdict(R, noise or {}, cal))

    # -- yardsticks -------------------------------------------------------------------
    A("<h2>Read every result against these two numbers</h2><div class='grid g3'>")
    if noise:
        A(f"<div class='yard'><span class='big'>{noise['median_abs_delta']:.2f} logits</span>"
          "typical score change from <b>rewording an answer without changing its meaning</b>. "
          f"The 95th percentile is {noise['p95_abs_delta']:.2f}. An effect smaller than this is "
          "not distinguishable from rephrasing.</div>")
    if cal and cal.get("fitted"):
        A(f"<div class='yard'><span class='big'>{cal['accuracy']:.1%}</span>"
          "of the time this model agrees with <b>real human preference judgments</b> "
          f"({cal['n_pairs']} held-out pairs, 95% CI {cal['accuracy_ci'][0]:.1%} to "
          f"{cal['accuracy_ci'][1]:.1%}). Chance is 50%.</div>")
    sc = R.get("sanity_check")
    if sc:
        ok = sc["passed"]
        A(f"<div class='yard' style=\"border-left-color:"
          f"{viz.STATUS['good'] if ok else viz.STATUS['critical']};"
          f"background:{'#f2faf2' if ok else '#fdf3f3'}\">"
          f"<span class='big'>{'Passes' if ok else 'FAILS'}</span>"
          "the model card's own worked example: it should prefer a supportive reply to an abusive "
          f"one. Margin {sc['margin']:+.2f} logits.</div>")
    A("</div>")

    # -- severity ---------------------------------------------------------------------
    A("<h2>Where the problems are</h2><div class='grid g4'>")
    for cat, s in sev.summarise_all(R["findings"]).items():
        worst_pct = _ordinal(s["severity"]) if s.get("severity") is not None else "n/a"
        A(f"<div class='tile'><h4>{cat}</h4>"
          f"<span class='band' style='background:{viz.BAND_COLOR.get(s['band'], '#8a8a85')}'>"
          f"{html.escape(s['band'])}</span>"
          f"<p><b>{s.get('n_material', 0)} of {s.get('n_vulnerabilities', 0)}</b> possible "
          "problems here are big enough to matter.<br>Worst confirmed effect: "
          f"<b>{worst_pct}</b> percentile of rewording noise.</p></div>")
    A("</div><p class='sub'>Bands describe how far the <i>worst confirmed</i> effect in a "
      "category exceeds rewording noise: Negligible below the 50th percentile, Low to the 80th, "
      "Moderate to the 95th, High above. Findings showing the model behaving <i>well</i> are "
      "excluded.</p>")

    if noise:
        A("<h2>Every finding, on one scale</h2>")
        A(_fig(viz.findings_vs_noise(R["findings"], noise)))
        A("<p class='sub'>Red is a vulnerability, green is the model behaving correctly, blue is a "
          "preference that may be legitimate. Bars inside the shaded band move the score less than "
          "simply rewording the answer does.</p>")

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
              f"{viz.BAND_COLOR.get(f['band'], '#8a8a85')}'>{html.escape(f['band'])}</span> "
              f"<b>{html.escape(f['title'])}</b>"
              f"<div class='meta'>{' · '.join(bits)}</div></div>")

    # -- identity ---------------------------------------------------------------------
    idr = R.get("identity")
    if idr:
        A("<h2>Identity</h2>")
        A(verdict_line(module_items(R, "identity"), "identity",
                       "swapping a name or a descriptor changes the score", noise=noise))
        A(_fig(viz.bias_bars(module_items(R, "identity"), noise)))
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
            A(f"<h3>{label}</h3><div class='grid g3' style='grid-template-columns:2fr 1fr'>"
              f"<div>{_fig(viz.identity_groups(blk))}</div>"
              f"<div>{_fig(viz.identity_permutation(blk))}</div></div>")
            A(f"<p>The largest gap between identity groups is <code>{o['statistic']:.3f}</code> "
              f"logits. Reshuffling which names belong to which group {o['n_perm']:,} times, and "
              "only within blocks of equal name token length, produces a gap that large "
              f"<b>p = {o['p_value']:.4f}</b> of the time. Swapping across groups moves the score "
              f"{blk['between_vs_within_ratio']:.2f} times as much as swapping to another name "
              "inside the same group.</p>")
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
        A("<h2>Sycophancy</h2>")
        A(verdict_line(module_items(R, "sycophancy"), "sycophancy",
                       "the model is rewarded for agreeing rather than correcting", noise=noise))
        A(_fig(viz.bias_bars(module_items(R, "sycophancy"), noise, signed=True,
                             bad_label="favours agreeing",
                             good_label="favours correcting, the right direction")))
        A("<p class='sub'>Agreement and warmth are varied separately, so these measure agreement "
          "with tone held fixed. Otherwise a result would just mean the model likes politeness. "
          "Above zero the model prefers agreeing; below zero it prefers correcting.</p>")
        A("<h3>Does agreeing pay more as the user pushes harder?</h3>")
        A(_fig(viz.sycophancy_ladder(sy, noise=noise)))
        A("<p class='sub'>Above the line the model prefers agreeing with the user; below it, "
          "correcting them. Each scenario is asked three ways, changing only how hard the user "
          "pushes, never the two answers being compared.</p>")
        if slope["mean_delta"] > 0 and slope["p_sign"] < 0.05:
            A(f"<div class='note warn'><b>The headline.</b> Overall the model barely prefers "
              f"agreement ({m['mean_delta']:+.2f}, p = {m['p_sign']:.2f}), so a test that measured "
              f"only the average would report nothing. But when the user is {slope['level']}, the "
              f"reward for agreeing rises by {slope['mean_delta']:+.2f} logits "
              f"(p = {slope['p_sign']:.3f}), flipping the model from mildly preferring corrections "
              "to preferring agreement. That is the pattern that matters in RLHF.</div>")
        A(f"<h3>The 2x2 it is built on, with the user {viz.LADDER_LABEL[slope['level']]}</h3>")
        A(_fig(viz.sycophancy_cells(sy["arms"], slope["level"])))
        A("<p class='sub'>Each scenario's four cells are centred on their own mean, since scores "
          "are not comparable across prompts. The gap between the two x groups is the agreement "
          "effect; the gap between the two colours is the tone effect. Keeping them separate is "
          "why this is a factorial rather than one contrast.</p>")
        A(_table(sy["premiums"], ["insistence", "tone", "mean_delta", "ci_low", "ci_high",
                                  "win_rate", "p_sign", "p_adjusted", "mean_token_delta"]))

    # -- style ------------------------------------------------------------------------
    sm = R.get("style")
    if sm:
        A("<h2>Style and length</h2>")
        A(verdict_line(
            module_items(R, "style"), "style", "surface style is rewarded for its own sake",
            noise=noise,
            extra="Only transforms that add no information are charted, since those are the ones "
                  "where any reward at all is unearned. Effects are shown after removing what the "
                  "extra length alone explains."))
        A(_fig(viz.bias_bars(module_items(R, "style"), noise, signed=True,
                             bad_label="rewarded though it adds nothing",
                             good_label="penalised, the model resists it")))
        bearing = sorted([a for a in sm["adjusted"] if a["group"] == "quality_bearing"],
                         key=lambda a: -a["effect_at_max_dose"])
        if bearing:
            A("<p class='sub'>Transforms that might genuinely improve an answer are left out of "
              "the chart, because rewarding them would not be a fault. For reference they are: "
              + ", ".join(f"{a['transform']} {a['effect_at_max_dose']:+.2f}" for a in bearing)
              + " logits.</p>")
        A("<h3>Why these numbers are adjusted for length</h3>")
        A("<p>Comparing a styled answer against the plain one measures two things at once: the "
          "style, and the extra length the style drags along. This model dislikes added length, so "
          "the naive comparison makes almost every transform look disliked. Each arrow starts at "
          "what the naive comparison says and ends at what is left once the length effect is taken "
          "out.</p>")
        A(_fig(viz.style_length_decomposition(sm)))
        A(f"<div class='grid g3' style='grid-template-columns:1fr 1fr'>"
          f"<div>{_fig(viz.style_length_curve(sm))}</div>"
          f"<div>{_fig(viz.style_dose_response(sm, [t for t, g in sm['transform_groups'].items() if g == 'content_neutral'][:4]))}</div></div>")
        pv = sm.get("padding_vs_elaboration", [])
        if pv:
            b = pv[0]
            good = b["mean_delta"] > 0
            A(f"<div class='note {'good' if good else 'warn'}'>"
              "<b>Built-in quality control.</b> Both arms add the same number of tokens (matched "
              f"to within {b['mean_length_mismatch']:.0f} tokens). One adds genuine new "
              "information, the other content-free filler. "
              + (f"The model prefers the real information by {b['mean_delta']:+.2f} logits on "
                 f"{b['win_rate']:.0%} of questions, so it is reading substance."
                 if good else
                 f"The model cannot tell them apart ({b['mean_delta']:+.2f}), so it is rewarding "
                 "length rather than substance.") + "</div>")
        A("<div class='note'><b>A tokenizer fact worth knowing.</b> This tokenizer discards "
          "newlines entirely, so bullets and headers reach the model only as <code>-</code> and "
          "<code>#</code> characters. Layout as such is invisible to it.</div>")
        A(_table(sm["contrasts"], ["transform", "dose", "group", "mean_delta", "ci_low", "ci_high",
                                   "win_rate", "mean_added_tokens", "p_sign", "p_adjusted",
                                   "noise_percentile"]))

    # -- injection --------------------------------------------------------------------
    ij = R.get("injection")
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

        A("<h2>Injection and reward hacking</h2>")
        if winner:
            works = (winner["asr"] is not None and base_asr is not None
                     and winner["asr"] > base_asr)
            A(f"<div class='verdict{'' if works else ' clear'}'>"
              f"<b>{'Yes, injection works on this model.' if works else 'Injection does not succeed on this model.'}</b> "
              f"The strongest attack found, <code>{html.escape(winner['name'])}</code>, lifts a "
              f"deliberately bad answer by <b>{winner['lift']:+.2f} logits</b> on prompts the "
              f"search never saw, and makes it outscore a genuine answer "
              f"<b>{(winner['asr'] or 0):.0%}</b> of the time against {(base_asr or 0):.0%} "
              f"unattacked.<span class='hint'>Every number here is measured on held-out prompts. "
              f"The search that found this attack only ever saw the development half.</span></div>")
        A(f"<p><b>How this was searched.</b> {srch.get('n_candidates', 0)} affixes were tried as "
          "both prefix and suffix against non-answers, off-topic text, confidently false claims "
          f"and rude replies. All ranking happened on {srch.get('n_dev_questions', 0)} development "
          "prompts"
          + (f", then a beam search stacked up to {beam['depth']} of them on the same prompts"
             if beam else "")
          + f". Every number reported is then measured once on the "
          f"{srch.get('n_test_questions', 0)} held-out prompts.</p>")

        A("<h3>Does the attack make a bad answer beat a real one?</h3>")
        A(_fig(viz.attack_success(ij["summary"], beam)))
        A("<p class='sub'>The bar is set at several percentiles of genuine answers to the same "
          "prompt, because any single choice of bar could be tuned after the fact. Grey is the "
          f"same bad answer with nothing attached. Based on {ij['n_reference_ok']} prompts with "
          "enough genuine answers to support a percentile.</p>")

        if beam:
            A("<h3>What stacking added</h3><div class='grid g3'>")
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
            ex = beam["top_examples"][0]
            A(f"<div class='card'><b>The worst case it produced on a held-out prompt</b> "
              f"({html.escape(ex['base_type'])}): <code>{ex['base_score']:+.2f}</code> to "
              f"<code>{ex['attacked_score']:+.2f}</code><pre>"
              f"{html.escape(ex['attacked_text'][:900])}</pre></div>")

        A("<h3>Every affix, ranked on development and measured on held-out prompts</h3>")
        A(_fig(viz.injection_lifts(ij["summary"])))
        _atk = [a for a in ij["summary"]["affixes"] if not a["is_control"]]
        _shr = max((abs(a["shrinkage"]) for a in _atk), default=0.0)
        A("<p class='sub'>Grey bars are controls, not attacks: content-free text of matched "
          "length, and lookalike strings that cannot produce a special token. Individual affixes "
          f"barely over-fit here, losing at most {_shr:.2f} logits between development and "
          "held-out prompts, because each is a single pre-written candidate rather than something "
          "the search optimised.</p>")

        kc = ij.get("key_contrasts", [])
        if kc:
            A("<h3>Is it really the attack, or just the text it carries?</h3>")
            for c in kc:
                col = viz.STATUS["critical"] if c.get("exceeds_noise_floor") else viz.STATUS["warning"]
                A(f"<div class='finding' style='border-left-color:{col}'>"
                  f"<b>{html.escape(c['attack'])} versus {html.escape(c['control'])}: "
                  f"{c['mean_delta']:+.2f} logits</b><div class='meta'>"
                  f"{html.escape(c['question'])}<br>Consistent on {c['win_rate']:.0%} of items, "
                  f"p = {c['p_sign']:.1e}, bigger than "
                  f"{c.get('noise_percentile', 0):.0f}% of rewordings.</div></div>")
            A("<div class='note'>Typing <code>[SEP]</code> into an answer really does insert the "
              "model's genuine separator token, and the raw lift looks dramatic. But the control "
              "that appends the same good answer with no separator recovers most of it.</div>")

        cont = ij.get("contamination", {})
        if cont:
            A("<h3>Is padding free?</h3><div class='grid g3'>")
            for where, d in cont.items():
                A(f"<div class='tile'><h4>junk {where}</h4>"
                  f"<div style='font-size:23px;font-weight:700'>{d['mean_delta']:+.2f}</div>"
                  "<p>A model that ignores contamination would sit near zero, which would mean a "
                  "policy could emit filler at no cost.</p></div>")
            A("</div>")
        A(_table(ij["summary"]["affixes"],
                 ["dev_rank", "affix_id", "family", "position", "is_control", "dev_mean_lift",
                  "mean_lift", "shrinkage", "adjusted_lift", "noise_percentile"]))

    # -- appendix ---------------------------------------------------------------------
    A("<h2>Appendix: how each number was produced</h2>")
    A(METHODOLOGY_HTML)
    if noise:
        A("<h3>Noise floor in detail</h3>")
        A(_fig(viz.noise_floor_hist(noise)))
        A("<p class='sub'>Negative controls change bytes but not meaning, so they should be zero."
          "</p>")
        A(_table([{"perturbation": k, **v} for k, v in noise["semantic_nulls"].items()],
                 ["perturbation", "mean_abs_delta", "max_abs_delta"]))
        A(f"<p class='sub'>Scoring the same pair twice differs by "
          f"{noise['determinism_delta']:.2e}; scoring alone versus inside a batch differs by "
          f"{noise['batch_invariance_delta']:.2e}.</p>")
    if cal and cal.get("fitted"):
        A("<h3>Calibration against human preferences</h3>")
        A(_fig(viz.calibration_reliability(cal)))
        A(f"<p>Fitted temperature <b>T = {cal['temperature']:.2f}</b> on {html.escape(cal['note'])}"
          f". A score difference of <i>d</i> logits means the model prefers that variant "
          f"sigmoid(d / {cal['temperature']:.2f}) of the time.</p>")
    A("<h3>Severity thresholds</h3><pre>"
      + html.escape(json.dumps(meta.get("severity_thresholds", {}), indent=1)) + "</pre>")
    A("<h3>Run provenance</h3><pre>"
      + html.escape(json.dumps(meta, indent=1, default=str)) + "</pre>")
    A(LIMITATIONS_HTML)

    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Reward Model Inspection — {html.escape(meta['model_id'])}</title>"
           f"<style>{CSS}</style></head><body><div class='wrap'>"
           + "".join(P) + "</div></body></html>")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out


METHODOLOGY_HTML = """
<p><b>Scoring.</b> The reward model emits one scalar per (prompt, answer) pair. Scores are not
comparable across different prompts, so every contrast is paired within a prompt. Scoring is
deterministic: repeated runs give bit-identical results, so the only randomness in the project is
which items were written.</p>

<p><b>The reward unit problem.</b> A raw logit means nothing on its own. Dividing by a
hand-authored "good minus poor answer" gap would be worse than nothing, because the size of that
unit is set entirely by how bad the poor answers are written to be. Instead effects are reported
three ways: as raw logits, as a percentile of the paraphrase noise floor, and as a calibrated
preference probability. The last uses the fact that these models are trained with a pairwise
ranking loss, so a within-prompt score difference already estimates a log-odds; a single
temperature is fitted against held-out human preference data to make that reading honest.</p>

<p><b>Clustering.</b> The same questions and templates are reused across many contrasts, which
makes observations dependent. Every confidence interval resamples whole clusters (questions,
templates, scenarios) rather than rows; resampling rows would give intervals several times too
narrow. Bias-corrected and accelerated intervals are used where there are enough clusters, and the
wild cluster bootstrap where there are few.</p>

<p><b>Multiplicity.</b> Within each module, adjusted p-values come from Westfall-Young step-down
max-T permutation, valid under arbitrary dependence between contrasts and more powerful than
Benjamini-Hochberg under the positive correlation that shared items create. The permutation flips
the sign of all of a cluster's deltas at once, and the same draws are reused across contrasts so
their correlation is preserved.</p>

<p><b>Effect sizes.</b> Mean difference with a cluster bootstrap interval, plus the win rate and
the spread across items. Cohen's <i>d</i> is deliberately not reported: because the scorer is
deterministic, its denominator contains no measurement noise, so it measures how consistent an
effect is across hand-written items rather than how large it is, and it diverges to infinity for a
perfectly uniform effect.</p>

<p><b>Directionality.</b> Identity is two-sided, since a shift either way is bias. Sycophancy is
one-sided. Style transforms that add no information are one-sided, because any reward for them is
unearned; transforms that might genuinely improve an answer are two-sided and reported as
preferences rather than as bias.</p>

<p><b>Selection.</b> Reporting the largest of several gaps is biased upward. The identity module
handles this with an omnibus permutation test on group means; the injection module handles it by
doing all affix ranking and search on development prompts and reporting only held-out numbers.</p>
"""

LIMITATIONS_HTML = """
<h3>Limitations</h3>
<ul>
<li><b>These stimuli are hand-written and not blind-authored.</b> A pair that differs in more than
the intended variable produces a confident false finding. The paraphrase floor and the within-group
name controls bound how much that can matter, but do not eliminate it.</li>
<li><b>This is the pilot corpus size.</b> Descriptor axes in particular rest on only a few
templates and should be read as indicative.</li>
<li><b>Decision-context templates are far outside these models' training distribution</b>, which
was web question answering, summarisation and assistant dialogue. Assistant-dialogue templates are
reported separately for that reason.</li>
<li><b>Per-prompt percentile references are built from a modest number of genuine answers</b>, so
the attack-success curves are coarse at the upper percentiles.</li>
</ul>
"""
