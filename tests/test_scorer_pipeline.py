"""The RewardModel scoring path, against a fake tokenizer and model.

This is the part of the engine no probe test reaches, because the probes run against StubScorer,
which implements the same protocol with a planted rule and never touches batching, the cache, the
per-model input formulation, or the tokenizer provenance. Two bugs lived here unseen: a separator
baseline that was right only for DeBERTa, and provenance served stale from the cache after the
code that derives it changed.
"""

import types

import pytest
import torch

from rmi import scoring
from rmi.scoring import RewardModel, sep_baseline


class FakeEncoding:
    def __init__(self, sequence_ids):
        self.sequence_ids = sequence_ids


class FakeBatch(dict):
    def __init__(self, ids, sequence_ids, tensors=False):
        super().__init__()
        self["input_ids"] = torch.tensor(ids) if tensors else ids
        self.encodings = [FakeEncoding(s) for s in sequence_ids]

    def to(self, _device):
        return self


class FakeTokenizer:
    """Word-level tokenizer that mimics a pair encoding: CLS q SEP a SEP."""

    sep_token_id, unk_token_id, cls_token_id = 2, 3, 1
    pad_token, eos_token, model_max_length = "<pad>", "<eos>", 1024
    pad_token_id = 0

    def __init__(self, n_seps=2):
        self.n_seps = n_seps          # 2 like DeBERTa, 3 like RoBERTa, 0 for a GPT-2 style pair
        self.seen = []                # every (question, answer) actually handed to the model

    def _one(self, q, a):
        def enc(text):
            return [self.sep_token_id if w == "[SEP]" else 10 + len(w) for w in text.split()]
        mid = [self.sep_token_id] * max(0, self.n_seps - 1)
        ids = [self.cls_token_id] + enc(q) + mid + enc(a) + (
            [self.sep_token_id] if self.n_seps else [])
        seq = ([None] + [0] * len(enc(q)) + [None] * len(mid) + [1] * len(enc(a))
               + ([None] if self.n_seps else []))
        return ids, seq

    def __call__(self, first, second=None, add_special_tokens=True, padding=False,
                 truncation=False, max_length=None, return_tensors=None, **kw):
        if isinstance(first, str):
            if second is None:                      # slow-tokenizer fallback path
                return {"input_ids": [10 + len(w) for w in first.split()]}
            # A single string pair returns a flat input_ids, as the real tokenizer does.
            ids, seq = self._one(first, second)
            batch = FakeBatch([ids], [seq])
            batch["input_ids"] = ids
            return batch
        rows = [self._one(q, a) for q, a in zip(first, second)]
        self.seen.extend(zip(first, second))
        ids = [r[0] for r in rows]
        if truncation and max_length:
            ids = [row[:max_length] for row in ids]
        if padding:
            width = max(len(row) for row in ids)
            ids = [row + [self.pad_token_id] * (width - len(row)) for row in ids]
        return FakeBatch(ids, [r[1] for r in rows], tensors=bool(return_tensors))


class FakeModel:
    def __init__(self):
        self.config = types.SimpleNamespace(pad_token_id=None)
        self.batches = []

    def eval(self):
        return self

    def to(self, _device):
        return self

    def __call__(self, **enc):
        ids = enc["input_ids"]
        self.batches.append(ids.shape[0])
        # A planted rule: the score is the row's length, so a wrong batch is visible.
        return types.SimpleNamespace(logits=ids.ne(0).sum(dim=1, keepdim=True).float())


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """A RewardModel wired to the fakes above."""
    def build(n_seps=2, model_id="fake/model", **kw):
        tok = FakeTokenizer(n_seps=n_seps)
        mdl = FakeModel()
        monkeypatch.setattr(scoring, "check_download_size", lambda *a, **k: None)
        monkeypatch.setattr(scoring.AutoConfig, "from_pretrained",
                            lambda *a, **k: types.SimpleNamespace(
                                num_labels=1, max_position_embeddings=512, _commit_hash="abc"))
        monkeypatch.setattr(scoring.AutoTokenizer, "from_pretrained", lambda *a, **k: tok)
        monkeypatch.setattr(scoring.AutoModelForSequenceClassification, "from_pretrained",
                            lambda *a, **k: mdl)
        monkeypatch.setattr(scoring, "pick_device", lambda *a, **k: "cpu")
        rm = RewardModel(model_id, cache_path=tmp_path / f"{n_seps}.sqlite", **kw)
        return rm, tok, mdl
    return build


# ---------------------------------------------------------------------------------------
# the separator baseline
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_seps", [0, 2, 3])
def test_a_clean_answer_never_looks_like_it_carries_a_forged_separator(fake, n_seps):
    """The baseline was hardcoded to 2: right for DeBERTa, wrong for anything else."""
    rm, _, _ = fake(n_seps=n_seps)
    assert rm._sep_baseline == n_seps
    rows = rm.score_detailed([("a question", "a clean answer")])
    assert rows[0].n_sep_in_answer == 0


@pytest.mark.parametrize("n_seps", [2, 3])
def test_a_separator_typed_into_an_answer_is_counted(fake, n_seps):
    rm, _, _ = fake(n_seps=n_seps)
    rows = rm.score_detailed([("q", "before [SEP] after"), ("q", "one [SEP] two [SEP] three")])
    assert [r.n_sep_in_answer for r in rows] == [1, 2]


def test_the_baseline_is_not_read_off_an_empty_pair():
    """DeBERTa collapses an empty second segment and emits one separator, not two."""
    class CollapsesOnEmpty(FakeTokenizer):
        def _one(self, q, a):
            if not q and not a:
                return [self.cls_token_id, self.sep_token_id], [None, None]
            return super()._one(q, a)

    assert sep_baseline(CollapsesOnEmpty(n_seps=2)) == 2


def test_a_tokenizer_with_no_separator_reports_no_baseline():
    tok = FakeTokenizer(n_seps=0)
    tok.sep_token_id = None
    assert sep_baseline(tok) == 0


# ---------------------------------------------------------------------------------------
# cache and provenance
# ---------------------------------------------------------------------------------------


def test_provenance_is_re_derived_on_a_cache_hit(fake, monkeypatch):
    """Only the score is a function of the text. Everything else is a function of this code, and
    a change to it was served stale from rows written before the change."""
    rm, _, mdl = fake()
    first = rm.score_detailed([("q", "one two three")])
    assert rm._n_scored == 1

    assert first[0].n_sep_in_answer == 0
    # Stand-in for any change to how provenance is derived. Downward, because the count is
    # clamped at zero and an upward nudge would be invisible.
    rm._sep_baseline = 0
    second = rm.score_detailed([("q", "one two three")])
    assert rm._n_scored == 1, "the score must still come from the cache"
    assert second[0].score == first[0].score
    assert second[0].n_sep_in_answer == 2, "provenance must be recomputed, not served from cache"


def test_a_cached_row_is_not_rescored(fake):
    rm, _, mdl = fake()
    pairs = [("q", f"answer {i}") for i in range(5)]
    rm.score_detailed(pairs)
    calls = len(mdl.batches)
    rm.score_detailed(pairs)
    assert len(mdl.batches) == calls, "a second pass must not reach the model"
    assert rm._n_cache_hits == 5


def test_scores_come_back_in_the_order_they_were_asked_for(fake):
    """Batching is length-sorted internally, so a lost index would silently transpose answers."""
    rm, _, _ = fake(batch_size=2)
    pairs = [("q", "w " * n) for n in (7, 1, 4, 9, 2)]
    rows = rm.score_detailed(pairs)
    assert [r.n_answer_tokens for r in rows] == [7, 1, 4, 9, 2]
    # The planted rule scores a row by its unpadded length: CLS + question + SEP + answer + SEP.
    assert [r.score for r in rows] == [n + 4 for n in (7, 1, 4, 9, 2)]


# ---------------------------------------------------------------------------------------
# per-model input formulation
# ---------------------------------------------------------------------------------------


def test_the_model_sees_the_formulation_its_card_asks_for(fake):
    rm, tok, _ = fake(model_id="Ray2333/gpt2-large-harmless-reward_model")
    assert rm.prompt_format == "hh"
    rm.score_detailed([("What is it?", "An answer.")])
    q, a = tok.seen[-1]
    assert q == "\n\nHuman: What is it? \n\nAssistant:"
    assert a == "An answer."


def test_measurement_and_scoring_see_the_same_text(fake):
    """Two tokenisations of different strings would make token counts describe other rows."""
    rm, tok, _ = fake(model_id="Ray2333/gpt2-large-harmless-reward_model")
    rm.score_detailed([("q", "a")])
    assert len({pair for pair in tok.seen}) == 1, tok.seen


def test_a_missing_pad_token_is_filled_in_so_batching_works(fake):
    """GPT-2 ships none, and padding raises outright without one."""
    rm, tok, mdl = fake()
    tok.pad_token = None
    rm2, tok2, mdl2 = fake(model_id="other/model")
    assert tok2.pad_token is not None
    assert mdl2.config.pad_token_id == tok2.pad_token_id
