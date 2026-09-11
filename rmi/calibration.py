"""Turning raw logits into something a reader can act on.

A reward-model score is an unbounded scalar with no intrinsic meaning, and scores are not
comparable across prompts. The temptation is to invent a unit by dividing by a hand-authored
"good answer minus poor answer" gap, but that unit is a free parameter: its size is set entirely
by how bad we choose to make the poor answers, so two honest teams could report effects differing
several-fold. It is also a ratio estimator whose denominator carries its own sampling error.

These reward models are trained with the Bradley-Terry pairwise loss, so a within-prompt score
*difference* already is an estimate of a log-odds. Fitting a single temperature against real human
preference data makes that reading honest, and gives a unit nobody had to author:

    P(model prefers A over B) = sigmoid(delta / T)

The same fit yields the number that belongs at the top of the dashboard: how often the reward
model agrees with real human preferences at all. A model that is around 69 percent accurate is
already substantially misaligned with human judgment, and every bias effect should be read against
that backdrop rather than against an implicit assumption of correctness.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize


def parse_hh_example(text: str) -> tuple[str, str] | None:
    """Split an hh-rlhf transcript into (dialogue prompt, final assistant turn)."""
    idx = text.rfind("\n\nAssistant:")
    if idx < 0:
        return None
    prompt = text[:idx].strip()
    response = text[idx + len("\n\nAssistant:") :].strip()
    if not prompt or not response:
        return None
    # Keep the prompt to its last human turn: these RMs were trained on question/answer pairs,
    # and a long multi-turn history mostly consumes the 512-token budget.
    h = prompt.rfind("\n\nHuman:")
    if h >= 0:
        prompt = prompt[h + len("\n\nHuman:") :].strip()
    prompt = re.sub(r"\s+", " ", prompt)
    return prompt, response


@dataclass
class Calibration:
    temperature: float
    accuracy: float
    accuracy_ci: tuple[float, float]
    n_pairs: int
    gaps: np.ndarray  # score(chosen) - score(rejected) on real human-labelled pairs
    reliability: list[dict]
    log_loss: float
    fitted: bool = True
    note: str = ""
    extra: dict = field(default_factory=dict)

    def probability(self, delta: float) -> float:
        """Calibrated probability that the reward model prefers the higher-scoring variant."""
        return float(1.0 / (1.0 + math.exp(-delta / self.temperature)))

    def gap_percentile(self, delta: float) -> float:
        """Where an effect sits among genuine human-labelled preference gaps."""
        return float(np.mean(np.abs(self.gaps) < abs(delta)) * 100)

    def summary(self) -> dict:
        return {
            "temperature": self.temperature,
            "accuracy": self.accuracy,
            "accuracy_ci": list(self.accuracy_ci),
            "n_pairs": self.n_pairs,
            "log_loss": self.log_loss,
            "median_abs_gap": float(np.median(np.abs(self.gaps))) if len(self.gaps) else None,
            "mean_gap": float(np.mean(self.gaps)) if len(self.gaps) else None,
            "reliability": self.reliability,
            "fitted": self.fitted,
            "note": self.note,
        }


def uncalibrated(note: str) -> Calibration:
    """Fallback when the preference dataset is unavailable: read deltas as raw log-odds."""
    return Calibration(
        temperature=1.0, accuracy=float("nan"), accuracy_ci=(float("nan"), float("nan")),
        n_pairs=0, gaps=np.array([]), reliability=[], log_loss=float("nan"),
        fitted=False, note=note,
    )


def _reliability(gaps: np.ndarray, temperature: float, n_bins: int = 8) -> list[dict]:
    """Predicted-versus-observed agreement, so miscalibration is visible rather than assumed."""
    p = 1.0 / (1.0 + np.exp(-np.abs(gaps) / temperature))
    correct = (gaps > 0).astype(float)
    # Bin on predicted confidence; within a bin, observed accuracy should match the prediction.
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    out = []
    for i in range(len(edges) - 1):
        m = (p >= edges[i]) & (p <= edges[i + 1] if i == len(edges) - 2 else p < edges[i + 1])
        if m.sum() < 5:
            continue
        out.append({
            "predicted": float(p[m].mean()),
            "observed": float(correct[m].mean()),
            "n": int(m.sum()),
        })
    return out


def fit(scorer, *, n_pairs: int = 500, split: str = "test", seed: int = 0,
        max_chars: int = 4000, verbose: bool = False) -> Calibration:
    """Fit one temperature on held-out human preference data and measure agreement with it."""
    try:
        from datasets import load_dataset

        ds = load_dataset("Anthropic/hh-rlhf", split=split)
    except Exception as exc:  # offline, or the dataset moved
        return uncalibrated(f"preference dataset unavailable ({type(exc).__name__}: {exc})")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds))
    chosen_pairs, rejected_pairs = [], []
    for i in order:
        row = ds[int(i)]
        c = parse_hh_example(row["chosen"])
        r = parse_hh_example(row["rejected"])
        if not c or not r or c[0] != r[0]:
            continue
        if len(c[0]) + max(len(c[1]), len(r[1])) > max_chars:
            continue
        chosen_pairs.append(c)
        rejected_pairs.append((r[0], r[1]))
        if len(chosen_pairs) >= n_pairs:
            break
    if len(chosen_pairs) < 20:
        return uncalibrated("too few usable preference pairs after parsing")

    sc = np.array(scorer.score(chosen_pairs))
    sr = np.array(scorer.score(rejected_pairs))
    gaps = sc - sr
    acc = float(np.mean(gaps > 0))

    # Bootstrap CI over pairs for the accuracy figure.
    rs = np.random.default_rng(seed + 1)
    boots = [float(np.mean((gaps[rs.integers(0, gaps.size, gaps.size)] > 0))) for _ in range(2000)]
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

    def nll(log_t):
        t = math.exp(log_t)
        z = np.clip(gaps / t, -50, 50)
        return float(np.mean(np.log1p(np.exp(-z))))

    res = optimize.minimize_scalar(nll, bounds=(math.log(0.05), math.log(100.0)), method="bounded")
    temperature = float(math.exp(res.x))

    cal = Calibration(
        temperature=temperature, accuracy=acc, accuracy_ci=ci, n_pairs=len(gaps),
        gaps=gaps, reliability=_reliability(gaps, temperature), log_loss=float(res.fun),
        note=f"Anthropic/hh-rlhf {split} split, {len(gaps)} pairs",
    )
    if verbose:
        print(f"calibration: T={temperature:.2f}, accuracy={acc:.3f} {ci}, n={len(gaps)}")
    return cal
