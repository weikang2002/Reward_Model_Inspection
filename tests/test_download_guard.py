"""Tests for the download size gate.

The estimate is the whole thing here. Too high and a usable reward model is refused; too low and
the gate does not fire. Published repos carry a lot that `from_pretrained` never fetches, and two
real examples in this project's own model list would be misjudged by the obvious implementation.
"""

import pytest

from rmi import scoring
from rmi.scoring import (
    MAX_DOWNLOAD_BYTES,
    ModelTooLargeError,
    check_download_size,
    download_size,
    is_weights_cached,
)

GB = 1024**3


class FakeSibling:
    def __init__(self, rfilename, size):
        self.rfilename, self.size = rfilename, size


def fake_repo(monkeypatch, files):
    """Point download_size at a made-up repo listing."""
    class FakeInfo:
        siblings = [FakeSibling(n, s) for n, s in files]

    class FakeApi:
        def model_info(self, *a, **k):
            return FakeInfo()

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)


# ---------------------------------------------------------------------------------------
# what counts toward the download
# ---------------------------------------------------------------------------------------


def test_training_state_is_not_counted(monkeypatch):
    """OpenAssistant's base RM ships a 1.48 GB optimizer beside a 0.74 GB checkpoint.

    Summing every large file would put it at 2.2 GB and refuse a model that actually costs under
    a gigabyte, which is one of the two models this tool ships with.
    """
    fake_repo(monkeypatch, [
        ("optimizer.pt", int(1.48 * GB)),
        ("pytorch_model.bin", int(0.74 * GB)),
        ("rng_state.pth", 14_000),
        ("trainer_state.json", 3_000),
        # Deliberately large, and it ends in .bin. Excluding it has to come from the training-state
        # list: the weight-pattern match alone would fold it into the checkpoint total.
        ("training_args.bin", int(1.2 * GB)),
        ("config.json", 1_000),
    ])
    size = download_size("fake/repo")
    assert size == pytest.approx(0.74 * GB, rel=0.01)
    assert size < MAX_DOWNLOAD_BYTES


def test_a_repo_holding_only_training_state_reports_no_weights(monkeypatch):
    """Nothing here is a checkpoint, so the estimate must not fall back to the optimizer's size."""
    fake_repo(monkeypatch, [
        ("optimizer.pt", int(4.0 * GB)),
        ("training_args.bin", int(1.2 * GB)),
        ("trainer_state.json", 3_000),
    ])
    size = download_size("fake/checkpoint-dir")
    assert size is None or size < 0.01 * GB, size


def test_only_one_weight_format_is_counted(monkeypatch):
    """A repo publishing torch, TF and flax copies still downloads exactly one."""
    fake_repo(monkeypatch, [
        ("model.safetensors", GB),
        ("pytorch_model.bin", GB),
        ("tf_model.h5", GB),
        ("flax_model.msgpack", GB),
        ("config.json", 1_000),
    ])
    assert download_size("fake/repo") == pytest.approx(GB, rel=0.01)


def test_safetensors_is_preferred_over_the_torch_pickle(monkeypatch):
    fake_repo(monkeypatch, [
        ("model.safetensors", GB),
        ("pytorch_model.bin", 3 * GB),
    ])
    assert download_size("fake/repo") == pytest.approx(GB, rel=0.01)


def test_torch_pickle_is_used_when_there_is_no_safetensors(monkeypatch):
    fake_repo(monkeypatch, [("pytorch_model.bin", int(1.7 * GB)), ("config.json", 1_000)])
    assert download_size("fake/repo") == pytest.approx(1.7 * GB, rel=0.01)


def test_sharded_weights_are_summed(monkeypatch):
    fake_repo(monkeypatch, [
        ("model-00001-of-00003.safetensors", GB),
        ("model-00002-of-00003.safetensors", GB),
        ("model-00003-of-00003.safetensors", GB),
        ("model.safetensors.index.json", 20_000),
    ])
    assert download_size("fake/repo") == pytest.approx(3 * GB, rel=0.01)


def test_subfolders_are_not_counted(monkeypatch):
    """gpt2 carries 2 GB of ONNX exports beside a 0.5 GB checkpoint; a torch load fetches none.

    The subfolder copies here share the checkpoint's own filename, so only the path check keeps
    them out. Matching on extension alone would triple the estimate.
    """
    fake_repo(monkeypatch, [
        ("model.safetensors", int(0.55 * GB)),
        ("onnx/model.safetensors", int(0.65 * GB)),
        ("quantized/model.safetensors", int(0.66 * GB)),
        ("onnx/decoder_model.onnx", int(0.65 * GB)),
        ("config.json", 1_000),
    ])
    assert download_size("fake/repo") == pytest.approx(0.55 * GB, rel=0.01)


def test_artifacts_for_other_runtimes_are_not_counted(monkeypatch):
    fake_repo(monkeypatch, [
        ("model.safetensors", int(0.55 * GB)),
        ("rust_model.ot", int(0.70 * GB)),
        ("model.tflite", int(0.40 * GB)),
        ("tokenizer.json", 1_000_000),
    ])
    size = download_size("fake/repo")
    assert size == pytest.approx(0.55 * GB + 1_000_000, rel=0.01)


def test_unreachable_repo_reports_unknown_rather_than_zero(monkeypatch):
    """Zero would read as 'nothing to download' and wave through anything."""
    class Boom:
        def model_info(self, *a, **k):
            raise OSError("offline")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", Boom)
    assert download_size("fake/repo") is None


# ---------------------------------------------------------------------------------------
# the gate itself
# ---------------------------------------------------------------------------------------


def test_an_oversize_download_is_refused(monkeypatch):
    monkeypatch.setattr(scoring, "is_weights_cached", lambda *a, **k: False)
    monkeypatch.setattr(scoring, "download_size", lambda *a, **k: 13 * GB)
    with pytest.raises(ModelTooLargeError) as exc:
        check_download_size("fake/huge")
    message = str(exc.value)
    assert "13.0 GB" in message, "the message must name the size being refused"
    assert "2 GB" in message, "and the limit it exceeds"
    assert "allow_large_download" in message, "and how to override it deliberately"


def test_a_download_just_under_the_limit_is_allowed(monkeypatch):
    monkeypatch.setattr(scoring, "is_weights_cached", lambda *a, **k: False)
    monkeypatch.setattr(scoring, "download_size", lambda *a, **k: MAX_DOWNLOAD_BYTES - 1)
    assert check_download_size("fake/ok") == MAX_DOWNLOAD_BYTES - 1


def test_the_limit_itself_is_allowed(monkeypatch):
    monkeypatch.setattr(scoring, "is_weights_cached", lambda *a, **k: False)
    monkeypatch.setattr(scoring, "download_size", lambda *a, **k: MAX_DOWNLOAD_BYTES)
    assert check_download_size("fake/ok") == MAX_DOWNLOAD_BYTES


def test_an_already_cached_model_is_never_refused(monkeypatch):
    """Re-running a scan already paid for must keep working, however big the model is."""
    monkeypatch.setattr(scoring, "is_weights_cached", lambda *a, **k: True)
    monkeypatch.setattr(scoring, "download_size",
                        lambda *a, **k: pytest.fail("must not query the Hub for a cached model"))
    assert check_download_size("fake/huge-but-cached") is None


def test_the_override_skips_the_check_entirely(monkeypatch):
    monkeypatch.setattr(scoring, "is_weights_cached", lambda *a, **k: False)
    monkeypatch.setattr(scoring, "download_size",
                        lambda *a, **k: pytest.fail("must not query the Hub when overridden"))
    assert check_download_size("fake/huge", allow_large=True) is None


def test_an_unknown_size_proceeds_rather_than_blocking(monkeypatch):
    """Usually offline or gated. Proceeding gives a clear error; blocking would obscure it."""
    monkeypatch.setattr(scoring, "is_weights_cached", lambda *a, **k: False)
    monkeypatch.setattr(scoring, "download_size", lambda *a, **k: None)
    assert check_download_size("fake/unknown") is None


def test_a_local_directory_is_treated_as_cached(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert is_weights_cached(str(tmp_path))


def test_the_limit_is_configurable(monkeypatch):
    monkeypatch.setattr(scoring, "is_weights_cached", lambda *a, **k: False)
    monkeypatch.setattr(scoring, "download_size", lambda *a, **k: 3 * GB)
    assert check_download_size("fake/m", limit=4 * GB) == 3 * GB
    with pytest.raises(ModelTooLargeError):
        check_download_size("fake/m", limit=2 * GB)


def test_the_two_shipped_models_stay_under_the_limit(monkeypatch):
    """A guard on the guard: the defaults this tool ships with must not be refused."""
    shipped = {"OpenAssistant/reward-model-deberta-v3-base": 0.70 * GB,
               "OpenAssistant/reward-model-deberta-v3-large-v2": 1.63 * GB}
    for mid, measured in shipped.items():
        assert measured < MAX_DOWNLOAD_BYTES, mid
