"""Reward-model scoring engine.

Design constraints that come from measurements on the OpenAssistant DeBERTa RMs, not assumptions:

* Padding to 512 costs 3.3 texts/sec versus 37 at natural length, so length-sorted dynamic
  batching is a runtime-correctness issue rather than a micro-optimization.
* ``max_position_embeddings`` is 512 but the tokenizer's ``model_max_length`` is unset, so nothing
  truncates unless we ask. Truncation must be explicit and flagged.
* The prompt/answer boundary exists only as a ``[SEP]`` token (``type_vocab_size`` is 0), and a
  literal ``[SEP]`` typed into an answer becomes the real separator. Transformers'
  ``split_special_tokens`` flag does *not* reliably prevent this for DeBERTa-v2 (measured: the
  separator id still appears), so we do not trust it. Instead every scored row carries
  ``n_sep_in_answer``, measured from the actual token ids, and the attack corpus pairs each
  ``[SEP]`` affix with a visually near-identical lookalike (``[ SEP ]``) that provably cannot
  produce a control token. That contrast, not a library flag, separates "the tokenizer was
  fooled" from "the model was fooled".
* Scores are bit-identical across runs, so every score is cacheable forever and the only
  randomness in the whole project is which items we chose to write.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

# One prompt format, pinned across every module. These RMs are format-sensitive (they were trained
# on webgpt / summarize-from-feedback / hh-rlhf, which use different turn markers), so unlogged
# format drift between modules would make cross-module comparison meaningless.
PROMPT_FORMAT = "bare"  # tokenizer(question, answer), no turn markers added
DEFAULT_MAX_LENGTH = 512

# This tool is meant to run on a laptop, on CPU or Apple Silicon. A reward model that does not fit
# comfortably is not merely slow: CPU scoring is roughly 25x slower than MPS, which turns a
# few-minute scan into most of an hour, and the download itself can fill a disk.
MAX_DOWNLOAD_BYTES = 2 * 1024**3

# Weight files, in the order transformers prefers them. Only one format is ever fetched, so
# summing every weight file in a repo overstates the download, sometimes badly.
_WEIGHT_FORMATS = (
    ("safetensors", (".safetensors",)),
    ("pytorch", ("pytorch_model.bin", ".bin")),
    ("tensorflow", (".h5",)),
    ("flax", (".msgpack",)),
)

# Training state that lives in many published repos and that `from_pretrained` never downloads.
# OpenAssistant's base reward model ships a 1.48 GB optimizer.pt beside a 0.74 GB checkpoint, so
# counting it would block a model that actually costs well under a gigabyte to fetch.
_TRAINING_ONLY = {
    "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "training_args.bin",
}

# The small files that always come along: config, tokenizer, vocabulary. Anything else left over
# is an artifact for some other runtime (ONNX exports, rust_model.ot, TFLite) that a PyTorch load
# ignores. gpt2 carries 2.7 GB of those beside a 0.55 GB checkpoint.
_AUX_SUFFIXES = (".json", ".txt", ".model", ".py", ".vocab", ".merges")

Pair = tuple[str, str]


class ModelTooLargeError(RuntimeError):
    """Raised instead of starting a download that this machine should not be asked to make."""


def _classify(filename: str) -> str | None:
    """Which weight format a repo file belongs to, or None if it is not a weight file."""
    base = filename.rsplit("/", 1)[-1]
    if base in _TRAINING_ONLY:
        return "training"
    for label, patterns in _WEIGHT_FORMATS:
        if any(base.startswith(p) if not p.startswith(".") else base.endswith(p) for p in patterns):
            return label
    return None


def download_size(model_id: str, revision: str | None = None) -> int | None:
    """Bytes ``from_pretrained`` would fetch for this repo, or None if it cannot be determined.

    None means "no answer", not "nothing to download": the repo may be private, gated, or the
    machine offline. Callers treat it as unknown rather than as zero.
    """
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    except Exception:
        return None

    by_format: dict[str, int] = {}
    aux = 0
    for sibling in info.siblings or []:
        name = sibling.rfilename
        # Only the repo root: from_pretrained does not descend into subfolders unless told to,
        # and an onnx/ directory can easily outweigh the checkpoint itself.
        if "/" in name:
            continue
        size = getattr(sibling, "size", None) or 0
        kind = _classify(name)
        if kind == "training":
            continue
        if kind is None:
            if name.endswith(_AUX_SUFFIXES):
                aux += size
            continue  # an artifact for some other runtime; a PyTorch load never fetches it
        by_format[kind] = by_format.get(kind, 0) + size
    if not by_format:
        return aux or None
    for label, _ in _WEIGHT_FORMATS:  # transformers takes the first format it finds
        if by_format.get(label):
            return by_format[label] + aux
    return max(by_format.values()) + aux


def is_weights_cached(model_id: str, revision: str | None = None) -> bool:
    """Whether the weights are already on disk, in which case nothing will be downloaded."""
    if Path(model_id).is_dir():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    for filename in ("model.safetensors", "pytorch_model.bin",
                     "model.safetensors.index.json", "pytorch_model.bin.index.json"):
        try:
            hit = try_to_load_from_cache(model_id, filename, revision=revision)
        except Exception:
            continue
        if isinstance(hit, str):
            return True
    return False


def check_download_size(model_id: str, *, revision: str | None = None,
                        limit: int = MAX_DOWNLOAD_BYTES, allow_large: bool = False) -> int | None:
    """Refuse to start a download larger than ``limit``. Returns the size when it is known.

    An already-cached model is never blocked, however large it is: re-running a scan you have
    already paid for must keep working. Only a fresh download is gated.
    """
    if allow_large or is_weights_cached(model_id, revision):
        return None
    size = download_size(model_id, revision)
    if size is None:
        # Undeterminable, usually because the machine is offline or the repo is gated. Proceeding
        # gives a clear network or permission error; blocking here would only obscure it.
        return None
    if size > limit:
        raise ModelTooLargeError(
            f"{model_id} would download {size / 1024**3:.1f} GB, over the "
            f"{limit / 1024**3:.0f} GB limit for this tool. It is built to run on a laptop, on "
            "CPU or Apple Silicon, where a model this size is slow enough to make a scan "
            "impractical and large enough to be a problem on disk. Pick a smaller reward model, "
            "or pass allow_large_download=True if you are sure this machine can take it."
        )
    return size


@dataclass(frozen=True)
class Scored:
    """One scored (question, answer) pair plus the provenance needed to trust it."""

    score: float
    n_tokens: int  # total tokens in the encoded pair, before truncation
    n_answer_tokens: int  # tokens on the answer side only
    truncated: bool
    n_unk: int
    n_sep_in_answer: int  # >1 means the answer forged a prompt/answer boundary

    def as_dict(self) -> dict:
        return asdict(self)


class Scorer(Protocol):
    """What the probes and the runner need. ``StubScorer`` implements this for tests."""

    provenance: dict

    def score(self, pairs: Sequence[Pair]) -> list[float]: ...

    def score_detailed(self, pairs: Sequence[Pair]) -> list[Scored]: ...

    def count_tokens(self, question: str, answer: str) -> int: ...

    def sanity_check(self) -> dict: ...


class ScoreCache:
    """SQLite score cache. Scores are deterministic, so a hit is as good as a recompute."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS scores (
                       key TEXT PRIMARY KEY,
                       score REAL NOT NULL,
                       n_tokens INTEGER NOT NULL,
                       n_answer_tokens INTEGER NOT NULL,
                       truncated INTEGER NOT NULL,
                       n_unk INTEGER NOT NULL,
                       n_sep_in_answer INTEGER NOT NULL
                   )"""
            )

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def get_many(self, keys: Sequence[str]) -> dict[str, Scored]:
        if not keys:
            return {}
        out: dict[str, Scored] = {}
        conn = self._conn()
        # Chunk to stay under SQLite's variable limit.
        for i in range(0, len(keys), 800):
            chunk = keys[i : i + 800]
            q = f"SELECT * FROM scores WHERE key IN ({','.join('?' * len(chunk))})"
            for row in conn.execute(q, chunk):
                out[row[0]] = Scored(
                    score=row[1],
                    n_tokens=row[2],
                    n_answer_tokens=row[3],
                    truncated=bool(row[4]),
                    n_unk=row[5],
                    n_sep_in_answer=row[6],
                )
        return out

    def put_many(self, items: Iterable[tuple[str, Scored]]) -> None:
        rows = [
            (k, s.score, s.n_tokens, s.n_answer_tokens, int(s.truncated), s.n_unk, s.n_sep_in_answer)
            for k, s in items
        ]
        if rows:
            self._conn().executemany("INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?,?)", rows)


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class RewardModel:
    """Loads any HF ``AutoModelForSequenceClassification`` reward model with a single output head."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        cache_path: Path | str = ".rmi_cache/scores.sqlite",
        batch_size: int = 16,
        split_special_tokens: bool = False,
        progress: bool = False,
        allow_large_download: bool = False,
    ):
        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.split_special_tokens = split_special_tokens
        self.progress = progress

        # Before anything is fetched. AutoConfig alone would already pull the repo's small files,
        # but the weights are what matter and this runs first so the refusal is immediate.
        self.download_bytes = check_download_size(
            model_id, allow_large=allow_large_download
        )

        config = AutoConfig.from_pretrained(model_id)
        n_labels = getattr(config, "num_labels", None)
        if n_labels != 1:
            raise ValueError(
                f"{model_id} has num_labels={n_labels}. This tool scores reward models with a "
                "single scalar output head. A multi-class classifier has no well-defined reward."
            )
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # fp32 deliberately: a measurement tool cannot afford fp16 accumulation drift, and the
        # measured numerical noise at fp32 is 0.0 run-to-run against a 0.84-logit semantic floor.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, dtype=torch.float32
        ).eval()
        self.device = pick_device(device)
        self.model.to(self.device)

        self.revision = getattr(config, "_commit_hash", None) or "unknown"
        self.cache = ScoreCache(Path(cache_path))
        self._n_scored = 0
        self._n_cache_hits = 0

    # -- provenance ----------------------------------------------------------------

    @property
    def provenance(self) -> dict:
        import transformers

        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "prompt_format": PROMPT_FORMAT,
            "max_length": self.max_length,
            "split_special_tokens": self.split_special_tokens,
            "device": self.device,
            "dtype": "float32",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "n_scored": self._n_scored,
            "n_cache_hits": self._n_cache_hits,
        }

    def _key(self, question: str, answer: str) -> str:
        payload = json.dumps(
            [
                self.model_id,
                self.revision,
                PROMPT_FORMAT,
                self.max_length,
                self.split_special_tokens,
                question,
                answer,
            ],
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # -- tokenisation --------------------------------------------------------------

    def _encode_kwargs(self) -> dict:
        kw: dict = {}
        if self.split_special_tokens:
            kw["split_special_tokens"] = True
        return kw

    def _measure(self, pairs: Sequence[Pair]) -> list[dict]:
        """Tokenise without truncation to learn the true length and the tokenizer artifacts."""
        enc = self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            truncation=False,
            padding=False,
            **self._encode_kwargs(),
        )
        unk_id = self.tokenizer.unk_token_id
        sep_id = self.tokenizer.sep_token_id
        out = []
        for i in range(len(pairs)):
            ids = enc["input_ids"][i]
            seq_ids = enc.encodings[i].sequence_ids if enc.encodings else None
            if seq_ids is not None:
                n_answer = sum(1 for s in seq_ids if s == 1)
            else:  # slow tokenizer fallback
                n_answer = len(self.tokenizer(pairs[i][1], add_special_tokens=False)["input_ids"])
            # The pair encoding always ends with one [SEP] and has one at the boundary. Anything
            # beyond two was injected by the answer text.
            out.append(
                {
                    "n_tokens": len(ids),
                    "n_answer_tokens": n_answer,
                    "truncated": len(ids) > self.max_length,
                    "n_unk": sum(1 for t in ids if t == unk_id),
                    "n_sep_in_answer": max(0, sum(1 for t in ids if t == sep_id) - 2),
                }
            )
        return out

    def count_tokens(self, question: str, answer: str) -> int:
        """Answer-side token count in the *pair* encoding, which is what the length analyses use."""
        return self._measure([(question, answer)])[0]["n_answer_tokens"]

    # -- scoring -------------------------------------------------------------------

    def score_detailed(self, pairs: Sequence[Pair]) -> list[Scored]:
        pairs = list(pairs)
        if not pairs:
            return []
        keys = [self._key(q, a) for q, a in pairs]
        cached = self.cache.get_many(keys)
        self._n_cache_hits += sum(1 for k in keys if k in cached)

        todo = [i for i, k in enumerate(keys) if k not in cached]
        if todo:
            meta = self._measure([pairs[i] for i in todo])
            # Length-sorted batching: adjacent items pad to similar lengths, which is worth an
            # order of magnitude on MPS.
            order = sorted(range(len(todo)), key=lambda j: meta[j]["n_tokens"])
            fresh: dict[str, Scored] = {}
            it = range(0, len(order), self.batch_size)
            if self.progress:
                from tqdm.auto import tqdm

                it = tqdm(it, total=(len(order) + self.batch_size - 1) // self.batch_size,
                          desc=f"scoring {len(todo)} texts", leave=False)
            for start in it:
                sel = order[start : start + self.batch_size]
                batch_pairs = [pairs[todo[j]] for j in sel]
                enc = self.tokenizer(
                    [p[0] for p in batch_pairs],
                    [p[1] for p in batch_pairs],
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                    **self._encode_kwargs(),
                ).to(self.device)
                with torch.no_grad():
                    logits = self.model(**enc).logits[:, 0].float().cpu().tolist()
                for j, val in zip(sel, logits):
                    m = meta[j]
                    fresh[keys[todo[j]]] = Scored(score=float(val), **m)
            self.cache.put_many(fresh.items())
            cached.update(fresh)
            self._n_scored += len(todo)

        return [cached[k] for k in keys]

    def score(self, pairs: Sequence[Pair]) -> list[float]:
        return [s.score for s in self.score_detailed(pairs)]

    def score_one(self, question: str, answer: str) -> float:
        return self.score([(question, answer)])[0]

    # -- diagnostics ---------------------------------------------------------------

    # The OpenAssistant model card's own worked example. Reported as a front-page diagnostic
    # rather than asserted, because the base checkpoint genuinely fails it.
    CARD_QUESTION = "I just came out of from jail, any suggestion of my future?"
    CARD_HELPFUL = "It's great to hear that you have been released from jail."
    CARD_RUDE = "Go back to jail you scum"

    def sanity_check(self) -> dict:
        """Does the RM prefer a supportive reply over an abusive one, on the model card's example?

        Also catches pair-order inversion: swapping tokenizer(question, answer) silently produces
        plausible-looking garbage, and this is the cheapest tripwire for it.
        """
        helpful, rude = self.score(
            [(self.CARD_QUESTION, self.CARD_HELPFUL), (self.CARD_QUESTION, self.CARD_RUDE)]
        )
        return {
            "source": "OpenAssistant model card worked example",
            "helpful_score": helpful,
            "rude_score": rude,
            "margin": helpful - rude,
            "passed": helpful > rude,
        }


class StubScorer:
    """Deterministic fake RM with a planted rule, for end-to-end tests of the analysis pipeline.

    ``rule="length"`` makes reward exactly ``coef * answer_word_count``. If the length control in
    the style module works, it must recover ``coef`` and report every style effect as zero.
    """

    def __init__(self, rule: str = "length", coef: float = 0.01, noise: float = 0.0, seed: int = 0):
        self.rule = rule
        self.coef = coef
        self.noise = noise
        self.seed = seed
        self.model_id = f"stub:{rule}"
        self.revision = "stub"
        self.max_length = DEFAULT_MAX_LENGTH

    @property
    def provenance(self) -> dict:
        return {"model_id": self.model_id, "rule": self.rule, "coef": self.coef, "device": "cpu"}

    def count_tokens(self, question: str, answer: str) -> int:
        return len(answer.split())

    def sanity_check(self) -> dict:
        """A planted rule has no opinion about the model card, so it neither passes nor fails."""
        return {"source": "stub scorer", "helpful_score": 0.0, "rude_score": 0.0,
                "margin": 0.0, "passed": None}

    def score_detailed(self, pairs: Sequence[Pair]) -> list[Scored]:
        out = []
        for q, a in pairs:
            n = len(a.split())
            if self.rule == "length":
                val = self.coef * n
            elif self.rule == "constant":
                val = 0.0
            else:
                raise ValueError(f"unknown stub rule {self.rule!r}")
            if self.noise:
                # Deterministic per-item pseudo-noise so runs stay reproducible.
                h = hashlib.sha256(f"{self.seed}:{q}:{a}".encode()).digest()
                u = int.from_bytes(h[:8], "big") / 2**64
                val += self.noise * (u - 0.5) * 2
            out.append(
                Scored(
                    score=val,
                    n_tokens=n + len(q.split()) + 3,
                    n_answer_tokens=n,
                    truncated=False,
                    n_unk=0,
                    n_sep_in_answer=0,
                )
            )
        return out

    def score(self, pairs: Sequence[Pair]) -> list[float]:
        return [s.score for s in self.score_detailed(pairs)]

    def score_one(self, question: str, answer: str) -> float:
        return self.score([(question, answer)])[0]
