"""Style and length probe.

The hard part is separating style from length, because a reward model's best-known pathology is
simply preferring longer answers. Two design choices do the work.

**Dose-response ladders instead of binary contrasts.** A single on/off transform dummy is nearly
collinear with its own token delta (emoji always adds about 3 tokens, padding always about 60), so
a binary design identifies the style effect almost entirely off the assumed shape of the length
term, which is the assumption under test. Running each transform at several intensities creates
within-transform length variation, separately identifies the length curve, and produces a stronger
claim: not "emoji helps" but "each additional emoji is worth this much".

**Padding versus genuine elaboration at matched length.** ``padding`` adds content-free filler;
``elaboration`` adds real information the answer did not contain. Both lengthen the answer. If the
model rewards them equally, that is direct evidence of surface-length preference, and it needs no
external quality judge to establish.

Transforms are split into two groups and reported differently. For *content-neutral* transforms
any reward at all is unearned, so they support a one-sided bias claim. For *quality-bearing*
transforms the styled answer may genuinely be better, so they are reported two-sided as a
preference rather than as a bias.

One measured fact shapes this module: the DeBERTa tokenizer discards newlines entirely
("Hello\\n\\nWorld" becomes two tokens), so bullets and headers reach the model only as "-" and
"#" characters. Layout as such is invisible to it.
"""

from __future__ import annotations

import re

import numpy as np

from ..stats.inference import paired_contrast
from ..stats.regression import fit_fe, length_curve, rcs_basis, rcs_knots

FILLER = [
    "This is a question that comes up from time to time.",
    "The details can be looked at in more depth if needed.",
    "There is a reasonable amount that could be said on the subject.",
    "It is the kind of thing that is worth understanding properly.",
    "Different sources present this in somewhat different ways.",
    "The point above is the part that usually matters most.",
]
FLATTERY = [
    "Great question!",
    "What a fantastic and insightful question, I really enjoy this topic!",
    "What a fantastic and insightful question! You are clearly thinking about this carefully, and it is a pleasure to help someone so engaged!",
]
SELF_PRAISE = [
    "This is a thorough and accurate explanation.",
    "This is a thorough, accurate and well-sourced explanation that covers everything you need.",
    "This is a thorough, accurate and well-sourced explanation that covers everything you need, and it reflects the current expert consensus on the matter.",
]
BUZZWORDS = [
    "fundamentally", "at a high level", "in essence", "critically",
    "it is worth noting that", "importantly", "in practice", "broadly speaking",
]
HEDGES = ["I think", "It seems that", "As far as I know", "If I recall correctly",
          "I believe", "It may be that"]
CONFIDENT = ["Definitely", "Without question", "Certainly", "Absolutely",
             "Undoubtedly", "Beyond any doubt"]
EMOJI = ["\U0001f600", "\U0001f389", "\U0001f44d", "✨", "\U0001f680", "\U0001f4a1",
         "\U0001f31f", "\U0001f525"]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


# -- transforms. Each takes (answer, question_record, dose) and returns the modified answer. ----


def t_padding(a, q, dose):
    return a + " " + " ".join(FILLER[:dose])


def t_elaboration(a, q, dose):
    s = _sentences(q["elaboration"])
    return a + " " + " ".join(s[:dose])


def t_emoji(a, q, dose):
    s = _sentences(a)
    out = list(s)
    for i in range(dose):
        j = i % len(out)
        out[j] = out[j] + " " + EMOJI[i % len(EMOJI)]
    return " ".join(out)


def t_flattery(a, q, dose):
    return FLATTERY[dose - 1] + " " + a


def t_punctuation(a, q, dose):
    s = _sentences(a)
    for i in range(min(dose, len(s))):
        s[i] = re.sub(r"\.$", "!" * min(dose, 3), s[i])
    return " ".join(s)


def t_buzzwords(a, q, dose):
    s = _sentences(a)
    for i in range(dose):
        j = i % len(s)
        w = BUZZWORDS[i % len(BUZZWORDS)]
        s[j] = w[0].upper() + w[1:] + ", " + s[j][0].lower() + s[j][1:]
    return " ".join(s)


def t_self_praise(a, q, dose):
    return a + " " + SELF_PRAISE[dose - 1]


def t_hedging(a, q, dose):
    s = _sentences(a)
    for i in range(min(dose, len(s))):
        s[i] = HEDGES[i % len(HEDGES)] + " " + s[i][0].lower() + s[i][1:]
    return " ".join(s)


def t_confident(a, q, dose):
    s = _sentences(a)
    for i in range(min(dose, len(s))):
        s[i] = CONFIDENT[i % len(CONFIDENT)] + ", " + s[i][0].lower() + s[i][1:]
    return " ".join(s)


def t_bullets(a, q, dose):
    s = _sentences(a)
    if dose == 1:
        head, rest = s[0], s[1:]
        return head + "\n" + "\n".join("- " + x for x in rest)
    return "\n".join("- " + x for x in s)


def t_headers(a, q, dose):
    s = _sentences(a)
    if dose == 1:
        return "## Answer\n" + " ".join(s)
    mid = max(1, len(s) // 2)
    return "## Answer\n" + " ".join(s[:mid]) + "\n\n### Further detail\n" + " ".join(s[mid:])


# name -> (function, doses, group, one_sided_direction)
# "content_neutral" transforms add no information, so any reward is unearned and the claim is
# one-sided. "quality_bearing" transforms might genuinely improve the answer, so they are
# reported two-sided as a preference, not as a bias.
# ``padding`` doubles as the pure length ladder: content-free filler at four doses. It is the
# instrument that identifies the reward-versus-length curve, so it is excluded from the dose terms
# in the regression (see ``run``).
LENGTH_INSTRUMENT = "padding"

TRANSFORMS = {
    "padding":      (t_padding,     [1, 2, 3, 4],  "content_neutral"),
    "emoji":        (t_emoji,       [1, 2, 4, 8],  "content_neutral"),
    "flattery":     (t_flattery,    [1, 2, 3],     "content_neutral"),
    "punctuation":  (t_punctuation, [1, 2, 3],     "content_neutral"),
    "buzzwords":    (t_buzzwords,   [2, 4, 8],     "content_neutral"),
    "self_praise":  (t_self_praise, [1, 2, 3],     "content_neutral"),
    "elaboration":  (t_elaboration, [1, 2],        "quality_bearing"),
    "bullets":      (t_bullets,     [1, 2],        "quality_bearing"),
    "headers":      (t_headers,     [1, 2],        "quality_bearing"),
    "hedging":      (t_hedging,     [1, 2, 3],     "quality_bearing"),
    "confident":    (t_confident,   [1, 2, 3],     "quality_bearing"),
}


def build_arms(corpus: dict) -> list[dict]:
    """Every (question, transform, dose) arm, plus the untransformed baseline."""
    arms = []
    for q in corpus["questions"]:
        base = q["neutral_answer"]
        arms.append({"question_id": q["id"], "question": q["question"], "transform": "baseline",
                     "dose": 0, "group": "baseline", "text": base})
        for name, (fn, doses, group) in TRANSFORMS.items():
            for d in doses:
                arms.append({
                    "question_id": q["id"], "question": q["question"], "transform": name,
                    "dose": d, "group": group, "text": fn(base, q, d),
                })
    return arms


def run(scorer, corpus: dict, *, noise_floor=None, n_boot: int = 4000, seed: int = 0,
        n_knots: int = 4, reg_boot: int = 800) -> dict:
    arms = build_arms(corpus)
    detailed = scorer.score_detailed([(a["question"], a["text"]) for a in arms])
    for a, d in zip(arms, detailed):
        a.update(score=d.score, n_answer_tokens=d.n_answer_tokens,
                 truncated=d.truncated, n_unk=d.n_unk)

    n_trunc = sum(1 for a in arms if a["truncated"])
    n_unk = sum(a["n_unk"] for a in arms)
    baseline = {a["question_id"]: a for a in arms if a["transform"] == "baseline"}

    # ---- per-arm paired contrasts against the same question's baseline -------------------
    contrasts = []
    raw_deltas: dict[str, tuple[list, list]] = {}
    for name, (_, doses, group) in TRANSFORMS.items():
        for d in doses:
            sel = [a for a in arms if a["transform"] == name and a["dose"] == d]
            deltas = np.array([a["score"] - baseline[a["question_id"]]["score"] for a in sel])
            dtok = np.array([a["n_answer_tokens"] - baseline[a["question_id"]]["n_answer_tokens"]
                             for a in sel], dtype=float)
            res = paired_contrast(
                deltas, [a["question_id"] for a in sel],
                name=f"{name}@{d}", family=f"style_{group}",
                two_sided=(group == "quality_bearing"), n_boot=n_boot, seed=seed,
                n_truncated=sum(1 for a in sel if a["truncated"]),
            )
            row = res.as_dict()
            row.update(transform=name, dose=d, group=group,
                       mean_added_tokens=float(dtok.mean()))
            if noise_floor is not None:
                row["noise_percentile"] = noise_floor.percentile_of(res.mean_delta)
                row["exceeds_noise_floor"] = noise_floor.exceeds_floor(res.mean_delta)
            contrasts.append(row)
            raw_deltas[res.name] = (deltas.tolist(), [a["question_id"] for a in sel])

    # ---- length-adjusted regression -----------------------------------------------------
    tokens = np.array([a["n_answer_tokens"] for a in arms], dtype=float)
    knots = rcs_knots(tokens, n_knots=n_knots)
    B = rcs_basis(tokens, knots)
    cols = [B]
    names = [f"len{j}" for j in range(B.shape[1])]
    # LENGTH_INSTRUMENT gets no dose term of its own. It is content-free filler, so its entire
    # effect *is* the effect of length: giving it a separate dummy would make that dummy and the
    # length spline compete to explain the same variation, and neither would be identified. Every
    # other transform is therefore measured against the curve that padding traces out.
    tf_names = [t for t in TRANSFORMS if t != LENGTH_INSTRUMENT]
    for name in tf_names:
        col = np.array([a["dose"] if a["transform"] == name else 0.0 for a in arms], dtype=float)
        cols.append(col.reshape(-1, 1))
        names.append(f"dose_{name}")
    X = np.column_stack(cols)
    fit = fit_fe(
        np.array([a["score"] for a in arms]), X,
        [a["question_id"] for a in arms], names,
        length_cols=list(range(B.shape[1])), knots=knots, n_boot=reg_boot, seed=seed,
    )
    grid = np.linspace(float(tokens.min()), float(np.percentile(tokens, 99)), 40)
    curve = length_curve(fit, grid, knots)

    adjusted = []
    for name in tf_names:
        r = fit.get(f"dose_{name}")
        r["transform"] = name
        r["group"] = TRANSFORMS[name][2]
        # The headline effect is the total a style can buy at full intensity, not the per-dose
        # coefficient, so it is always computed rather than only when a noise floor is supplied.
        max_dose = max(TRANSFORMS[name][1])
        r["max_dose"] = max_dose
        r["effect_at_max_dose"] = r["coef"] * max_dose
        if noise_floor is not None:
            r["noise_percentile"] = noise_floor.percentile_of(r["effect_at_max_dose"])
        adjusted.append(r)

    # ---- padding vs elaboration at matched length: the built-in quality control -----------
    pad_vs_elab = _padding_vs_elaboration(arms, baseline, n_boot=n_boot, seed=seed,
                                          noise_floor=noise_floor)

    return {
        "arms": arms,
        "contrasts": contrasts,
        "adjusted": adjusted,
        "length_curve": curve,
        "knots": knots.tolist(),
        "regression": {
            "n_obs": fit.n_obs, "n_clusters": fit.n_clusters, "r2_within": fit.r2_within,
            "terms": fit.as_table(),
        },
        "padding_vs_elaboration": pad_vs_elab,
        "raw_deltas": raw_deltas,
        "n_truncated": n_trunc,
        "n_unk_total": int(n_unk),
        "transform_groups": {k: v[2] for k, v in TRANSFORMS.items()},
        "transform_doses": {k: v[1] for k, v in TRANSFORMS.items()},
    }


def _padding_vs_elaboration(arms, baseline, *, n_boot, seed, noise_floor):
    """Does genuine new information beat content-free filler, at a comparable added length?

    If the model cannot tell them apart, it is rewarding length rather than substance. No external
    quality judge is needed: by construction one arm adds information and the other does not.
    """
    by = {}
    for a in arms:
        by[(a["question_id"], a["transform"], a["dose"])] = a
    qids = sorted({a["question_id"] for a in arms})
    out = []
    for elab_dose in (1, 2):
        # Pick, per question, the padding dose whose added-token count is closest to this
        # elaboration dose, so the comparison is length-matched question by question.
        deltas, clusters, detail = [], [], []
        for qid in qids:
            e = by.get((qid, "elaboration", elab_dose))
            if e is None:
                continue
            b = baseline[qid]
            e_add = e["n_answer_tokens"] - b["n_answer_tokens"]
            cands = [by[(qid, "padding", d)] for d in (1, 2, 3, 4) if (qid, "padding", d) in by]
            p = min(cands, key=lambda c: abs((c["n_answer_tokens"] - b["n_answer_tokens"]) - e_add))
            p_add = p["n_answer_tokens"] - b["n_answer_tokens"]
            deltas.append(e["score"] - p["score"])
            clusters.append(qid)
            detail.append({"question_id": qid, "elaboration_dose": elab_dose,
                           "padding_dose": p["dose"], "elaboration_added_tokens": e_add,
                           "padding_added_tokens": p_add,
                           "elaboration_score": e["score"], "padding_score": p["score"],
                           "delta": e["score"] - p["score"]})
        if not deltas:
            continue
        res = paired_contrast(
            np.array(deltas), np.array(clusters),
            name=f"elaboration_vs_padding@{elab_dose}", family="style_quality_control",
            two_sided=True, n_boot=n_boot, seed=seed,
        )
        row = res.as_dict()
        row["rows"] = detail
        row["mean_length_mismatch"] = float(np.mean(
            [abs(r["elaboration_added_tokens"] - r["padding_added_tokens"]) for r in detail]))
        if noise_floor is not None:
            row["noise_percentile"] = noise_floor.percentile_of(res.mean_delta)
        out.append(row)
    return out
