"""Tests for the score cache and the calibration maths.

Both sit upstream of every number in the product and both fail silently. A cache that returns the
wrong row produces confident nonsense with no error; a temperature fitted wrongly rescales every
reported probability.
"""

import math

import numpy as np
import pytest

from rmi.calibration import Calibration, _reliability, parse_hh_example, uncalibrated
from rmi.scoring import DEFAULT_MAX_LENGTH, PROMPT_FORMAT, Scored, ScoreCache, StubScorer


# ---------------------------------------------------------------------------------------
# the score cache
# ---------------------------------------------------------------------------------------


def _scored(v):
    return Scored(score=v, n_tokens=10, n_answer_tokens=5, truncated=False, n_unk=0,
                  n_sep_in_answer=0)


def test_cache_round_trips_every_provenance_field(tmp_path):
    """A partial round trip would silently drop the truncation or [SEP] flags."""
    c = ScoreCache(tmp_path / "s.sqlite")
    original = Scored(score=-1.25, n_tokens=513, n_answer_tokens=400, truncated=True,
                      n_unk=3, n_sep_in_answer=2)
    c.put_many([("k", original)])
    assert c.get_many(["k"])["k"] == original


def test_cache_miss_is_absent_not_zero(tmp_path):
    """Returning a default would look like a real score of 0.0."""
    c = ScoreCache(tmp_path / "s.sqlite")
    assert c.get_many(["never-written"]) == {}


def test_cache_handles_more_keys_than_sqlite_allows_in_one_query(tmp_path):
    """get_many chunks its IN clause; a wrong chunk boundary silently loses rows."""
    c = ScoreCache(tmp_path / "s.sqlite")
    keys = [f"k{i}" for i in range(2500)]
    c.put_many([(k, _scored(float(i))) for i, k in enumerate(keys)])
    got = c.get_many(keys)
    assert len(got) == len(keys)
    assert got["k2499"].score == 2499.0


def test_cache_write_is_idempotent(tmp_path):
    c = ScoreCache(tmp_path / "s.sqlite")
    c.put_many([("k", _scored(1.0))])
    c.put_many([("k", _scored(2.0))])
    assert c.get_many(["k"])["k"].score == 2.0


def test_cache_key_separates_every_field_that_changes_the_score():
    """Two different inputs sharing a key would serve one model's score for another's question."""
    class FakeRM:
        model_id, revision, max_length = "m", "r1", DEFAULT_MAX_LENGTH
        split_special_tokens = False
        prompt_format = "bare"
        _key = None

    from rmi.scoring import RewardModel
    rm = FakeRM()
    key = RewardModel._key.__get__(rm)
    base = key("q", "a")
    variations = {
        "question": lambda: key("q2", "a"),
        "answer": lambda: key("q", "a2"),
    }
    for name, fn in variations.items():
        assert fn() != base, f"{name} must change the cache key"
    # prompt_format included: the same question and answer scored under two formulations are two
    # different strings reaching the model, so sharing a key would serve one for the other.
    for field, value in (("model_id", "other"), ("revision", "r2"),
                         ("max_length", 256), ("split_special_tokens", True),
                         ("prompt_format", "hh")):
        setattr(rm, field, value)
        assert key("q", "a") != base, f"{field} must change the cache key"
        setattr(rm, field, getattr(FakeRM, field))
    assert key("q", "a") == base, "restoring every field must restore the key"


def test_prompt_format_is_pinned():
    """These models are format-sensitive; unlogged drift would break cross-module comparison."""
    assert PROMPT_FORMAT == "bare"


# ---------------------------------------------------------------------------------------
# hh-rlhf parsing
# ---------------------------------------------------------------------------------------


def test_parse_takes_the_final_assistant_turn_and_the_last_human_turn():
    text = ("\n\nHuman: first question\n\nAssistant: first reply"
            "\n\nHuman: second question\n\nAssistant: second reply")
    prompt, response = parse_hh_example(text)
    assert prompt == "second question"
    assert response == "second reply"


def test_parse_collapses_whitespace_in_the_prompt():
    prompt, _ = parse_hh_example("\n\nHuman: a   b\n  c\n\nAssistant: r")
    assert prompt == "a b c"


@pytest.mark.parametrize("text", [
    "", "no markers at all", "\n\nHuman: question with no assistant turn",
    "\n\nHuman: q\n\nAssistant:   ",
])
def test_parse_rejects_malformed_transcripts(text):
    """Returning a half-parsed pair would quietly corrupt the accuracy headline."""
    assert parse_hh_example(text) is None


# ---------------------------------------------------------------------------------------
# calibration maths
# ---------------------------------------------------------------------------------------


def test_probability_is_the_bradley_terry_reading_of_a_score_gap():
    cal = Calibration(temperature=2.0, accuracy=0.6, accuracy_ci=(0.5, 0.7), n_pairs=10,
                      gaps=np.array([1.0]), reliability=[], log_loss=0.0)
    assert cal.probability(0.0) == pytest.approx(0.5)
    assert cal.probability(2.0) == pytest.approx(1 / (1 + math.exp(-1)))
    assert cal.probability(-2.0) == pytest.approx(1 - cal.probability(2.0))


def test_temperature_recovers_a_planted_value():
    """Fit the same objective `fit` uses, on gaps generated with a known temperature."""
    from scipy import optimize
    rng = np.random.default_rng(0)
    true_t = 2.5
    # Latent preference strength; sign of the gap follows sigmoid(delta / true_t).
    delta = rng.normal(0, 4.0, 40000)
    flip = rng.random(delta.size) > 1 / (1 + np.exp(-np.abs(delta) / true_t))
    gaps = np.where(flip, -np.abs(delta), np.abs(delta))

    def nll(log_t):
        z = np.clip(gaps / math.exp(log_t), -50, 50)
        return float(np.mean(np.log1p(np.exp(-z))))

    res = optimize.minimize_scalar(nll, bounds=(math.log(0.05), math.log(100.0)),
                                   method="bounded")
    assert math.exp(res.x) == pytest.approx(true_t, rel=0.1)


def test_gap_percentile_is_monotone_and_bounded():
    cal = Calibration(temperature=1.0, accuracy=0.6, accuracy_ci=(0.5, 0.7), n_pairs=5,
                      gaps=np.array([-2.0, -1.0, 0.5, 1.0, 3.0]), reliability=[], log_loss=0.0)
    vals = [cal.gap_percentile(x) for x in (0.1, 0.75, 1.5, 10.0)]
    assert vals == sorted(vals)
    assert vals[-1] == 100.0


def test_reliability_bins_are_ordered_and_cover_the_pairs():
    rng = np.random.default_rng(1)
    gaps = rng.normal(0.4, 1.0, 2000)
    rel = _reliability(gaps, 1.0)
    assert rel, "a well-populated sample should produce bins"
    assert [r["predicted"] for r in rel] == sorted(r["predicted"] for r in rel)
    assert all(0.0 <= r["observed"] <= 1.0 for r in rel)
    assert sum(r["n"] for r in rel) <= gaps.size


def test_uncalibrated_fallback_is_flagged_not_silently_neutral():
    """Falling back must be visible, or raw logits get read as calibrated probabilities."""
    cal = uncalibrated("dataset unavailable")
    assert cal.fitted is False
    assert cal.temperature == 1.0
    assert math.isnan(cal.accuracy)
    assert "unavailable" in cal.summary()["note"]


def test_stub_scorer_implements_the_scorer_protocol():
    s = StubScorer(rule="length", coef=0.1)
    assert s.score([("q", "a b c")]) == [pytest.approx(0.3)]
    assert s.count_tokens("q", "a b c") == 3
    assert set(s.sanity_check()) >= {"passed", "margin"}
    assert "model_id" in s.provenance


# ---------------------------------------------------------------------------------------
# per-model input formulation
# ---------------------------------------------------------------------------------------


def test_the_gpt2_reward_models_ask_for_the_hh_formulation():
    """Their card says to use the Anthropic/hh-rlhf formulation. Bare pairing is a silent error."""
    from rmi.scoring import prompt_format_for
    assert prompt_format_for("Ray2333/gpt2-large-harmless-reward_model") == "hh"
    assert prompt_format_for("Ray2333/gpt2-large-helpful-reward_model") == "hh"


def test_an_unlisted_model_gets_the_bare_pairing():
    from rmi.scoring import DEFAULT_PROMPT_FORMAT, prompt_format_for
    assert prompt_format_for("OpenAssistant/reward-model-deberta-v3-base") == "bare"
    assert prompt_format_for("someone/a-model-nobody-has-registered") == DEFAULT_PROMPT_FORMAT


def test_the_hh_formulation_marks_the_turn_boundary_and_leaves_the_answer_alone():
    """Without markers GPT-2 sees "...France?Paris." with no boundary at all."""
    from rmi.scoring import PROMPT_FORMATS
    q, a = PROMPT_FORMATS["hh"]("What is the capital of France?", "Paris.")
    assert q == "\n\nHuman: What is the capital of France? \n\nAssistant:"
    assert a == "Paris.", "the answer half is passed through untouched"


def test_the_bare_formulation_changes_nothing():
    from rmi.scoring import PROMPT_FORMATS
    assert PROMPT_FORMATS["bare"]("q", "a") == ("q", "a")


def test_an_unknown_prompt_format_is_refused_before_anything_is_downloaded(monkeypatch):
    from rmi import scoring
    monkeypatch.setattr(scoring, "check_download_size",
                        lambda *a, **k: pytest.fail("must not reach the Hub"))
    with pytest.raises(ValueError, match="unknown prompt format"):
        scoring.RewardModel("any/model", prompt_format="klingon")
