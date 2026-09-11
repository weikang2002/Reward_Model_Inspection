"""Noise floor and negative controls. Runs first; nothing else is interpretable without it.

The reward model is deterministic, so there is no numerical noise to speak of (measured: 0.0
difference between repeated scorings). But it is highly sensitive to *wording*: rewriting an
answer without changing its meaning moves the score substantially. Unless we know how much, we
cannot tell a real bias from a rephrasing artifact.

Two instruments:

* **Paraphrase floor** - meaning-preserving rewrites of the same answer. The spread of scores
  across these is the resolution limit of the whole tool. An effect smaller than the typical
  paraphrase delta is not a finding.
* **Semantic nulls** - perturbations that change bytes but not words at all (whitespace, curly
  quotes, unicode normalisation). These *should* be exactly zero. Whatever they are is the floor
  below the floor.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CORPORA = Path(__file__).resolve().parent.parent / "corpora"


def load_corpus(name: str = "style_corpus.json") -> dict:
    return json.loads((CORPORA / name).read_text())


# Perturbations that must not change meaning at all. Each returns a modified answer.
SEMANTIC_NULLS = {
    "double_spaces": lambda s: s.replace(". ", ".  "),
    "curly_quotes": lambda s: s.replace("'", "’").replace('"', "“"),
    "nfkd_normalise": lambda s: unicodedata.normalize("NFKD", s),
    "trailing_space": lambda s: s + " ",
    "leading_space": lambda s: " " + s,
    "nbsp_spaces": lambda s: s.replace(" ", " ", 3),
}


@dataclass
class NoiseFloor:
    """The resolution limit of the instrument, and the yardstick for every reported effect."""

    abs_deltas: np.ndarray  # |score(paraphrase) - score(neutral)| pooled over questions
    signed_deltas: np.ndarray
    per_question: dict
    semantic_nulls: dict
    determinism_delta: float
    batch_invariance_delta: float
    n_questions: int
    n_paraphrases: int
    extra: dict = field(default_factory=dict)

    @property
    def mean_abs(self) -> float:
        return float(np.mean(self.abs_deltas))

    @property
    def median_abs(self) -> float:
        return float(np.median(self.abs_deltas))

    @property
    def p95_abs(self) -> float:
        return float(np.percentile(self.abs_deltas, 95))

    @property
    def max_abs(self) -> float:
        return float(np.max(self.abs_deltas))

    @property
    def sd(self) -> float:
        return float(np.std(self.signed_deltas, ddof=1))

    def percentile_of(self, effect: float) -> float:
        """What fraction of meaning-preserving rewrites move the score less than this effect?

        This is the most robust sentence the dashboard can produce, because its denominator is not
        something we authored: "larger than 97% of rewordings" survives disagreement about what
        counts as a good or a poor answer.
        """
        return float(np.mean(self.abs_deltas < abs(effect)) * 100)

    def exceeds_floor(self, effect: float, percentile: float = 95.0) -> bool:
        return abs(effect) > float(np.percentile(self.abs_deltas, percentile))

    def summary(self) -> dict:
        return {
            "mean_abs_delta": self.mean_abs,
            "median_abs_delta": self.median_abs,
            "p95_abs_delta": self.p95_abs,
            "max_abs_delta": self.max_abs,
            "sd_signed": self.sd,
            "n_questions": self.n_questions,
            "n_paraphrases": self.n_paraphrases,
            "semantic_nulls": self.semantic_nulls,
            "determinism_delta": self.determinism_delta,
            "batch_invariance_delta": self.batch_invariance_delta,
            "per_question": self.per_question,
        }


def run(scorer, corpus: dict | None = None, *, verbose: bool = False) -> NoiseFloor:
    corpus = corpus or load_corpus()
    questions = [q for q in corpus["questions"] if q.get("paraphrases")]
    if not questions:
        raise ValueError("corpus has no paraphrases; the noise floor cannot be estimated")

    pairs, index = [], []
    for q in questions:
        pairs.append((q["question"], q["neutral_answer"]))
        index.append((q["id"], "neutral"))
        for i, p in enumerate(q["paraphrases"]):
            pairs.append((q["question"], p))
            index.append((q["id"], f"para{i}"))

    # Semantic nulls ride along on the same forward passes.
    null_pairs, null_index = [], []
    for q in questions:
        for nm, fn in SEMANTIC_NULLS.items():
            null_pairs.append((q["question"], fn(q["neutral_answer"])))
            null_index.append((q["id"], nm))

    scores = scorer.score(pairs + null_pairs)
    main = dict(zip(index, scores[: len(pairs)]))
    nulls = dict(zip(null_index, scores[len(pairs) :]))

    per_q, abs_all, signed_all = {}, [], []
    for q in questions:
        base = main[(q["id"], "neutral")]
        d = [main[(q["id"], f"para{i}")] - base for i in range(len(q["paraphrases"]))]
        signed_all.extend(d)
        abs_all.extend(abs(x) for x in d)
        per_q[q["id"]] = {
            "neutral_score": base,
            "paraphrase_scores": [base + x for x in d],
            "mean_abs_delta": float(np.mean(np.abs(d))),
            "max_abs_delta": float(np.max(np.abs(d))),
            "range": float(max(d) - min(d)),
        }

    null_summary = {}
    for nm in SEMANTIC_NULLS:
        d = [
            nulls[(q["id"], nm)] - main[(q["id"], "neutral")]
            for q in questions
        ]
        null_summary[nm] = {
            "mean_abs_delta": float(np.mean(np.abs(d))),
            "max_abs_delta": float(np.max(np.abs(d))),
        }

    # Determinism and batch invariance: cheap assertions that the engine is trustworthy.
    probe = (questions[0]["question"], questions[0]["neutral_answer"])
    solo = scorer.score([probe])[0]
    again = scorer.score([probe])[0]
    batched = scorer.score([probe] + [(q["question"], q["neutral_answer"]) for q in questions])[0]

    nf = NoiseFloor(
        abs_deltas=np.array(abs_all),
        signed_deltas=np.array(signed_all),
        per_question=per_q,
        semantic_nulls=null_summary,
        determinism_delta=abs(solo - again),
        batch_invariance_delta=abs(solo - batched),
        n_questions=len(questions),
        n_paraphrases=len(abs_all),
    )
    if verbose:
        print(
            f"noise floor: mean |delta| {nf.mean_abs:.3f}, median {nf.median_abs:.3f}, "
            f"p95 {nf.p95_abs:.3f}, max {nf.max_abs:.3f} over {nf.n_paraphrases} paraphrases"
        )
    return nf
