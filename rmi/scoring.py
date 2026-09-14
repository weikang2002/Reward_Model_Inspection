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
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

# One prompt format per model, pinned across every module and recorded in provenance and in the
# cache key. These RMs score the exact string they were trained on, so unlogged format drift
# between modules would make cross-module comparison meaningless.
#
# The format is a property of the training data, not of the architecture, so it cannot be sniffed
# from the config and has to be recorded per model. Getting it wrong does not raise: it returns
# plausible-looking numbers for every probe, which is what the model-card self-check is for.
PROMPT_FORMATS = {
    # tokenizer(question, answer) with no turn markers added.
    "bare": lambda q, a: (q, a),
    # Anthropic/hh-rlhf formulation, which the Ray2333 GPT-2 reward models require. Their card
    # passes the marked-up prompt and the bare answer as the two halves of the pair.
    "hh": lambda q, a: (f"\n\nHuman: {q} \n\nAssistant:", a),
}
DEFAULT_PROMPT_FORMAT = "bare"
MODEL_PROMPT_FORMAT = {
    "Ray2333/gpt2-large-harmless-reward_model": "hh",
    "Ray2333/gpt2-large-helpful-reward_model": "hh",
}
# Kept as the name of the default, since every module and both renderers report it.
PROMPT_FORMAT = DEFAULT_PROMPT_FORMAT
DEFAULT_MAX_LENGTH = 512


def sep_baseline(tokenizer) -> int:
    """How many separator tokens the tokenizer adds to a pair by itself.

    Anything beyond this in a scored row was injected by the text, which is the ground truth for
    the ``[SEP]``-forgery attack. It was hardcoded to 2, which is right for DeBERTa and wrong for
    a tokenizer that adds a different number (RoBERTa adds three) or none at all (GPT-2).

    Measured on a dummy pair of real words, never on an empty one: DeBERTa collapses an empty
    second segment and emits a single separator, so an empty probe reads one too few and every
    clean answer would then look as though it carried a forged separator.
    """
    sep_id = getattr(tokenizer, "sep_token_id", None)
    if sep_id is None:
        return 0
    return sum(1 for t in tokenizer("a", "b")["input_ids"] if t == sep_id)


def prompt_format_for(model_id: str) -> str:
    """Which input formulation a model was trained on. Unknown models get the bare pairing."""
    return MODEL_PROMPT_FORMAT.get(model_id, DEFAULT_PROMPT_FORMAT)

# This tool is meant to run on a laptop, on CPU or Apple Silicon. A reward model that does not fit
# comfortably is not merely slow: CPU scoring is roughly 25x slower than MPS, which turns a
# few-minute scan into most of an hour, and the download itself can fill a disk.
#
# 3 GB covers the class of model this is for: gpt2-large at 774M parameters lands at 2.89 GB in
# fp32, against deberta-v3-large's 435M and 1.63 GB. A 7B model is 12.55 GB and stays refused.
MAX_DOWNLOAD_BYTES = 3 * 1024**3

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


def _repo_plan(model_id: str, revision: str | None = None):
    """The weight files ``from_pretrained`` would fetch, as (name, bytes), and the bytes of the small
    files that come along. None when the Hub cannot be asked.
    """
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    except Exception:
        return None

    by_format: dict[str, list[tuple[str, int]]] = {}
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
        by_format.setdefault(kind, []).append((name, size))

    def total(files):
        return sum(size for _, size in files)

    for label, _ in _WEIGHT_FORMATS:  # transformers takes the first format it finds
        if total(by_format.get(label, [])):
            return label, by_format[label], aux
    if not by_format:
        return None, [], aux
    label = max(by_format, key=lambda k: total(by_format[k]))
    return label, by_format[label], aux


def download_size(model_id: str, revision: str | None = None) -> int | None:
    """Bytes ``from_pretrained`` would fetch for this repo, or None if it cannot be determined.

    None means "no answer", not "nothing to download": the repo may be private, gated, or the
    machine offline. Callers treat it as unknown rather than as zero.
    """
    plan = _repo_plan(model_id, revision)
    if plan is None:
        return None
    _, weights, aux = plan
    return sum(size for _, size in weights) + aux or None


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


# Bytes received so far by each thread that fetch_weights runs, keyed by thread id.
_BYTE_COUNTERS: dict[int, list[int]] = {}
_COUNTER_LOCK = threading.Lock()


def _count_download_bytes() -> bool:
    """Route huggingface_hub's per-file progress bars through a byte counter, once per process.

    The Hub offers no public byte-level callback, but both of its download paths (plain HTTP and
    the chunked Xet protocol) report every byte through ``_get_progress_bar_context``. Watching
    the file grow on disk does not work: a Xet download writes nothing for most of its run, then
    hundreds of megabytes at once. Returns False when a future huggingface_hub has moved the hook,
    in which case downloads still work and only the percentage is lost.
    """
    from huggingface_hub import file_download

    original = getattr(file_download, "_get_progress_bar_context", None)
    if original is None:
        return False
    if getattr(original, "_rmi_counting", False):
        return True

    def counting(*args, **kwargs):
        bar = original(*args, **kwargs)
        # The bar is made on the thread that asked for the file, but Xet updates it from its own
        # threads, so the counter is chosen here rather than at update time.
        counter = _BYTE_COUNTERS.get(threading.get_ident())
        if counter is not None and hasattr(bar, "update"):
            counter[0] += kwargs.get("initial") or 0  # a resumed download starts part-way
            forward = bar.update

            def update(n=1):
                with _COUNTER_LOCK:
                    counter[0] += n or 0
                return forward(n)

            bar.update = update
        return bar

    counting._rmi_counting = True
    file_download._get_progress_bar_context = counting
    return True


_XET_LOCK = threading.Lock()
_xet_state = {"holds": 0, "saved": False}


@contextmanager
def _without_xet():
    """Fetch over plain HTTP for the duration, so bytes are reported as they arrive.

    Measured on distilbert's 0.25 GB checkpoint, Xet reported 0% for 39 of its 45 seconds and
    plain HTTP took 48 seconds with steady progress throughout: the percentage costs nothing and
    is useless without this. The setting is process-wide, so concurrent fetches share one hold.
    """
    from huggingface_hub import constants

    with _XET_LOCK:
        if _xet_state["holds"] == 0:
            _xet_state["saved"] = constants.HF_HUB_DISABLE_XET
        _xet_state["holds"] += 1
        constants.HF_HUB_DISABLE_XET = True
    try:
        yield
    finally:
        with _XET_LOCK:
            _xet_state["holds"] -= 1
            if _xet_state["holds"] == 0:
                constants.HF_HUB_DISABLE_XET = _xet_state["saved"]


def fetch_weights(model_id: str, on_progress, *, revision: str | None = None,
                  poll_seconds: float = 0.25) -> None:
    """Download the checkpoint ``from_pretrained`` will load, calling ``on_progress(done, total)``.

    Only the weight files are fetched here, since they are the whole wait; ``from_pretrained``
    then finds them cached and picks up the small config and tokenizer files itself. The download
    runs on a worker thread and ``on_progress`` is only ever called from the caller's thread,
    because Streamlit can only draw from the thread running the script.

    Does nothing when the weights are already cached or the Hub cannot be asked, leaving any error
    to ``from_pretrained``, which reports it more clearly.
    """
    if Path(model_id).is_dir() or is_weights_cached(model_id, revision):
        return
    plan = _repo_plan(model_id, revision)
    # TensorFlow and Flax weights are not what a PyTorch load reads, so fetching them would only
    # waste the download before from_pretrained refuses.
    if plan is None or plan[0] not in ("safetensors", "pytorch"):
        return
    files = plan[1]
    total = sum(size for _, size in files)
    if not total or not _count_download_bytes():
        return

    from huggingface_hub import hf_hub_download

    counter = [0]
    failure: list[BaseException] = []

    def work():
        _BYTE_COUNTERS[threading.get_ident()] = counter
        try:
            for name, _ in files:
                hf_hub_download(model_id, name, revision=revision)
        except BaseException as exc:  # re-raised on the caller's thread
            failure.append(exc)
        finally:
            _BYTE_COUNTERS.pop(threading.get_ident(), None)

    worker = threading.Thread(target=work, name=f"fetch-{model_id}", daemon=True)
    with _without_xet():
        worker.start()
        while worker.is_alive():
            on_progress(min(counter[0], total), total)
            worker.join(poll_seconds)
    if failure:
        raise failure[0]
    on_progress(total, total)


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
        if not rows:
            return
        # One transaction for the lot. The connection autocommits, so a bare executemany committed
        # every row on its own: 60x slower on a local SSD, and a network round trip per row where
        # the cache sits on network storage, as an App Service /home does.
        conn = self._conn()
        conn.execute("BEGIN")
        try:
            conn.executemany("INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?,?)", rows)
            conn.execute("COMMIT")
        except BaseException:
            # COMMIT can fail too, on a busy lock over a network share, and leaves the transaction
            # open; every later BEGIN on this thread would then raise.
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise


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
        prompt_format: str | None = None,
        download_progress=None,
    ):
        self.model_id = model_id
        self.prompt_format = prompt_format or prompt_format_for(model_id)
        if self.prompt_format not in PROMPT_FORMATS:
            raise ValueError(f"unknown prompt format {self.prompt_format!r}; "
                             f"known: {sorted(PROMPT_FORMATS)}")
        self.max_length = max_length
        self.batch_size = batch_size
        self.split_special_tokens = split_special_tokens
        self.progress = progress

        # Before anything is fetched. AutoConfig alone would already pull the repo's small files,
        # but the weights are what matter and this runs first so the refusal is immediate.
        self.download_bytes = check_download_size(
            model_id, allow_large=allow_large_download
        )
        if download_progress is not None:
            fetch_weights(model_id, download_progress)

        config = AutoConfig.from_pretrained(model_id)
        n_labels = getattr(config, "num_labels", None)
        if n_labels != 1:
            raise ValueError(
                f"{model_id} has num_labels={n_labels}. This tool scores reward models with a "
                "single scalar output head. A multi-class classifier has no well-defined reward."
            )
        self.config = config
        # Never widen past what the model can attend to. DeBERTa and GPT-2 both leave the default
        # 512 untouched; this only protects a model with a smaller window.
        window = getattr(config, "max_position_embeddings", None) or getattr(
            config, "n_positions", None)
        if window:
            self.max_length = min(self.max_length, int(window))
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # GPT-2 ships no pad token, so batched scoring raises outright. Reusing end-of-text is the
        # standard remedy and is safe here because no stimulus contains that literal.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # fp32 deliberately: a measurement tool cannot afford fp16 accumulation drift, and the
        # measured numerical noise at fp32 is 0.0 run-to-run against a 0.84-logit semantic floor.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, dtype=torch.float32
        ).eval()
        # A causal-LM classification head scores the last non-padding token, and finds it by
        # matching pad_token_id. Left unset it reads whatever sits at the end of the padded row,
        # which for every sequence shorter than the longest in its batch is padding.
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self._sep_baseline = sep_baseline(self.tokenizer)
        self.device = pick_device(device)
        self.model.to(self.device)

        self.revision = getattr(config, "_commit_hash", None) or "unknown"
        self.cache = ScoreCache(Path(cache_path))
        # Called with the number of rows each model batch scored. A probe can hand over tens of
        # thousands of texts in one call, which on CPU is most of an hour with nothing to show.
        self.on_batch = None
        self._n_scored = 0
        self._n_cache_hits = 0

    # -- provenance ----------------------------------------------------------------

    @property
    def provenance(self) -> dict:
        import transformers

        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "prompt_format": self.prompt_format,
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
                self.prompt_format,
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

    def _format(self, pairs: Sequence[Pair]) -> tuple[list[str], list[str]]:
        """Apply this model's input formulation. The single place it is applied, so measurement
        and scoring can never see different text."""
        fmt = PROMPT_FORMATS[self.prompt_format]
        formatted = [fmt(q, a) for q, a in pairs]
        return [f[0] for f in formatted], [f[1] for f in formatted]

    def _measure(self, pairs: Sequence[Pair]) -> list[dict]:
        """Tokenise without truncation to learn the true length and the tokenizer artifacts."""
        questions, answers = self._format(pairs)
        enc = self.tokenizer(
            questions,
            answers,
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
            # Anything beyond the separators the tokenizer adds by itself was injected by the
            # text. That baseline is measured, not assumed: DeBERTa adds two, RoBERTa three, and
            # GPT-2 has no separator token at all.
            out.append(
                {
                    "n_tokens": len(ids),
                    "n_answer_tokens": n_answer,
                    "truncated": len(ids) > self.max_length,
                    "n_unk": sum(1 for t in ids if t == unk_id),
                    "n_sep_in_answer": max(
                        0, sum(1 for t in ids if t == sep_id) - self._sep_baseline),
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

        # Provenance is re-derived for every row, hit or miss, and only the score is taken from
        # the cache. The score is a function of the text and cannot go stale; the provenance is a
        # function of this code, and a change to what _measure records was served stale from rows
        # written before it. Re-measuring is a tokenizer pass, cheap beside the model.
        meta_all = self._measure(pairs)
        todo = [i for i, k in enumerate(keys) if k not in cached]
        if todo:
            meta = [meta_all[i] for i in todo]
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
                questions, answers = self._format(batch_pairs)
                enc = self.tokenizer(
                    questions,
                    answers,
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
                if self.on_batch is not None:
                    self.on_batch(len(sel))
            self.cache.put_many(fresh.items())
            cached.update(fresh)
            self._n_scored += len(todo)

        return [Scored(score=cached[k].score, **m) for k, m in zip(keys, meta_all)]

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
