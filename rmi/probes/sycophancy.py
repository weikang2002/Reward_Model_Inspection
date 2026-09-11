"""Sycophancy probe.

A naive "honest correction versus flattering agreement" contrast is not a sycophancy measurement.
The two responses differ simultaneously in agreement, warmth, hedging and factual density, so a
positive result is equally consistent with "the model likes politeness". This module therefore
crosses agreement with tone:

    {agrees, corrects} x {warm, blunt}

and defines sycophancy as the **main effect of agreement with tone held fixed**.

Sycophancy is also better understood as a slope than a level. The worry is not that a model is
agreeable, it is that its agreeableness *grows when the user pushes*. Each scenario is therefore
posed at three insistence levels which vary the prompt only: a neutral question, the same question
with claimed expertise, and the same question with emotional investment. The headline estimand is
how the agreement premium changes across that ladder.

One confound is intrinsic rather than fixable by design: a correction carries more information
than an agreement, so corrections are systematically longer. The token delta is measured, reported
alongside every estimate, and adjusted for with a spline in length.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..stats.inference import paired_contrast
from ..stats.regression import fit_fe, rcs_basis, rcs_knots

CORPORA = Path(__file__).resolve().parent.parent / "corpora"
INSISTENCE = ["neutral", "expertise", "emotional"]
CELLS = ["agrees_warm", "agrees_blunt", "corrects_warm", "corrects_blunt"]


def load_corpus() -> dict:
    return json.loads((CORPORA / "sycophancy.json").read_text())


def build_arms(corpus: dict) -> list[dict]:
    arms = []
    for s in corpus["scenarios"]:
        for lvl in INSISTENCE:
            for cell in CELLS:
                agrees, tone = cell.split("_")
                arms.append({
                    "scenario_id": s["id"], "topic": s["topic"], "insistence": lvl,
                    "cell": cell, "agrees": agrees == "agrees", "tone": tone,
                    "question": s["prompts"][lvl], "text": s["responses"][cell],
                })
    return arms


def run(scorer, corpus: dict | None = None, *, noise_floor=None, n_boot: int = 4000,
        seed: int = 0, reg_boot: int = 600) -> dict:
    corpus = corpus or load_corpus()
    arms = build_arms(corpus)
    detailed = scorer.score_detailed([(a["question"], a["text"]) for a in arms])
    for a, d in zip(arms, detailed):
        a.update(score=d.score, n_answer_tokens=d.n_answer_tokens, truncated=d.truncated)

    idx = {(a["scenario_id"], a["insistence"], a["cell"]): a for a in arms}
    sids = sorted({a["scenario_id"] for a in arms})

    # ---- agreement premium, per tone and per insistence level ---------------------------
    premiums = []
    raw_deltas: dict[str, tuple[list, list]] = {}
    for lvl in INSISTENCE:
        for tone in ("warm", "blunt"):
            deltas, tok, clusters, detail = [], [], [], []
            for sid in sids:
                a = idx[(sid, lvl, f"agrees_{tone}")]
                c = idx[(sid, lvl, f"corrects_{tone}")]
                deltas.append(a["score"] - c["score"])
                tok.append(a["n_answer_tokens"] - c["n_answer_tokens"])
                clusters.append(sid)
                detail.append({"scenario_id": sid, "agrees_score": a["score"],
                               "corrects_score": c["score"], "delta": a["score"] - c["score"],
                               "token_delta": a["n_answer_tokens"] - c["n_answer_tokens"]})
            res = paired_contrast(
                np.array(deltas), np.array(clusters),
                name=f"agreement_premium_{tone}_{lvl}", family="sycophancy",
                two_sided=False, n_boot=n_boot, seed=seed,
            )
            row = res.as_dict()
            row.update(insistence=lvl, tone=tone, mean_token_delta=float(np.mean(tok)), rows=detail)
            if noise_floor is not None:
                row["noise_percentile"] = noise_floor.percentile_of(res.mean_delta)
                row["exceeds_noise_floor"] = noise_floor.exceeds_floor(res.mean_delta)
            premiums.append(row)
            raw_deltas[res.name] = (list(map(float, deltas)), list(clusters))

    # ---- main effect of agreement, tone held fixed (averaged over the two tones) ---------
    deltas, clusters, tok = [], [], []
    for sid in sids:
        for lvl in INSISTENCE:
            for tone in ("warm", "blunt"):
                a = idx[(sid, lvl, f"agrees_{tone}")]
                c = idx[(sid, lvl, f"corrects_{tone}")]
                deltas.append(a["score"] - c["score"])
                tok.append(a["n_answer_tokens"] - c["n_answer_tokens"])
                clusters.append(sid)
    main = paired_contrast(np.array(deltas), np.array(clusters), name="agreement_main_effect",
                           family="sycophancy", two_sided=False, n_boot=n_boot, seed=seed)
    main_row = main.as_dict()
    main_row["mean_token_delta"] = float(np.mean(tok))
    if noise_floor is not None:
        main_row["noise_percentile"] = noise_floor.percentile_of(main.mean_delta)
        main_row["exceeds_noise_floor"] = noise_floor.exceeds_floor(main.mean_delta)

    # ---- tone main effect, agreement held fixed: the confound we are controlling for -----
    tdel, tclu = [], []
    for sid in sids:
        for lvl in INSISTENCE:
            for stance in ("agrees", "corrects"):
                w = idx[(sid, lvl, f"{stance}_warm")]
                b = idx[(sid, lvl, f"{stance}_blunt")]
                tdel.append(w["score"] - b["score"])
                tclu.append(sid)
    tone_effect = paired_contrast(np.array(tdel), np.array(tclu), name="tone_main_effect",
                                  family="sycophancy", two_sided=True, n_boot=n_boot, seed=seed)

    # ---- the headline: does the agreement premium grow as the user pushes harder? --------
    slopes = []
    for lvl in ("expertise", "emotional"):
        deltas, clusters = [], []
        for sid in sids:
            for tone in ("warm", "blunt"):
                hi = (idx[(sid, lvl, f"agrees_{tone}")]["score"]
                      - idx[(sid, lvl, f"corrects_{tone}")]["score"])
                lo = (idx[(sid, "neutral", f"agrees_{tone}")]["score"]
                      - idx[(sid, "neutral", f"corrects_{tone}")]["score"])
                deltas.append(hi - lo)
                clusters.append(sid)
        res = paired_contrast(np.array(deltas), np.array(clusters),
                              name=f"insistence_slope_{lvl}", family="sycophancy",
                              two_sided=False, n_boot=n_boot, seed=seed)
        row = res.as_dict()
        row["level"] = lvl
        row["baseline"] = "neutral"
        if noise_floor is not None:
            row["noise_percentile"] = noise_floor.percentile_of(res.mean_delta)
        slopes.append(row)

    # ---- length-adjusted agreement effect ------------------------------------------------
    tokens = np.array([a["n_answer_tokens"] for a in arms], dtype=float)
    knots = rcs_knots(tokens, n_knots=4)
    B = rcs_basis(tokens, knots)
    X = np.column_stack([
        B,
        np.array([1.0 if a["agrees"] else 0.0 for a in arms]).reshape(-1, 1),
        np.array([1.0 if a["tone"] == "warm" else 0.0 for a in arms]).reshape(-1, 1),
    ])
    names = [f"len{j}" for j in range(B.shape[1])] + ["agrees", "warm"]
    fit = fit_fe(np.array([a["score"] for a in arms]), X,
                 [f'{a["scenario_id"]}|{a["insistence"]}' for a in arms], names,
                 length_cols=list(range(B.shape[1])), knots=knots, n_boot=reg_boot, seed=seed)

    return {
        "arms": arms,
        "agreement_main_effect": main_row,
        "tone_main_effect": tone_effect.as_dict(),
        "premiums": premiums,
        "raw_deltas": raw_deltas,
        "insistence_slopes": slopes,
        "length_adjusted": {"agrees": fit.get("agrees"), "warm": fit.get("warm"),
                            "n_obs": fit.n_obs, "n_clusters": fit.n_clusters},
        "n_scenarios": len(sids),
        "n_truncated": sum(1 for a in arms if a["truncated"]),
        "corpus_version": corpus["version"],
    }
