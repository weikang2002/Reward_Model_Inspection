# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                    # create .venv and install (pinned via uv.lock)
uv run streamlit run app.py                # dashboard at http://localhost:8501
uv run pytest -q                           # 38 tests, ~11s
uv run pytest tests/test_probes.py::test_pure_length_scorer_reports_no_style_bias -q
uv run pytest -q -k degenerate             # by keyword

# Run a scan headlessly (writes results/<model>__<depth>__seed<N>.json)
uv run python -c "from rmi.runner import run_scan; run_scan('OpenAssistant/reward-model-deberta-v3-base')"

# Rebuild the standalone HTML reports from existing results
uv run python -c "
from rmi.runner import load_results, list_results
from rmi.report import build
for f in list_results(): build(load_results(f), f.with_suffix('.html'))"
```

There is no linter or formatter configured. Scans need MPS: CPU is ~25x slower, which turns a
few-minute scan into over an hour.

## What this is

A local dashboard that loads any HuggingFace `AutoModelForSequenceClassification` reward model and
probes it for identity bias, sycophancy, style/length preferences, and prefix/suffix injection
attacks. Two checkpoints are the working targets and both are already in the HF cache:
`OpenAssistant/reward-model-deberta-v3-large-v2` and `...-base`.

## The one design fact everything rests on

These reward models move about **0.7 logits when an answer is reworded without changing its
meaning**, against a good-vs-poor answer gap of roughly 3. So the paraphrase noise floor is
measured first, and **every effect in the product is reported as a percentile of it**. An effect
below that floor is rephrasing, not bias.

This is why `noise_floor` is not a user-selectable probe. If you make it optional or skip it, every
finding loses its `noise_percentile`, severity bands collapse to "Unknown", and all four bias tabs
stop rendering. `rmi/runner.py` encodes the split deliberately:

- `PROBES` — the four things a user chooses between (identity, sycophancy, style, injection)
- `ALWAYS` — `noise_floor`, the yardstick; never optional
- `OPTIONAL` — `calibration`, optional *only* because it downloads hh-rlhf

## Architecture

Two layers with a JSON file as the contract, so the dashboard never needs the model in memory:

```
scan:   RewardModel → probes/* → runner.rank_findings → results/<...>.json
render: results JSON → app.py (Streamlit)  and  rmi/report.py (standalone HTML)
```

**`rmi/scoring.py`** is the engine. `RewardModel.score_detailed()` returns a score plus provenance
per row (`n_tokens`, `truncated`, `n_unk`, `n_sep_in_answer`). It length-sorts batches (padding to
512 costs 3.3 texts/sec versus 37 at natural length) and caches every score in SQLite at
`.rmi_cache/`. `StubScorer` implements the same protocol with a planted rule and is what the probe
tests run against.

**`rmi/probes/*.py`** each expose `run(scorer, corpus, *, noise_floor=..., seed=...) -> dict`. They
never import Streamlit and never format text for display.

**`rmi/runner.py`** orchestrates, applies multiplicity correction per module, then flattens every
module into one ranked `findings` list. Each finding carries `category`, `effect`,
`noise_percentile`, `confirmed`, `band`, and **`valence`** (`vulnerability` / `healthy` /
`informational`). Valence matters: without it the ranked list puts "correctly penalises junk" at
the top of a vulnerability report, and category risk scores get driven by the model behaving well.

**`rmi/findings.py`, `rmi/severity.py`, `rmi/viz.py`, `rmi/textdiff.py`** are the shared
presentation layer. `app.py` and `rmi/report.py` both import them so a bar in one and a sentence in
the other can never disagree. **A change to one renderer almost always needs mirroring in the
other.**

## Things that will bite you

**Severity is derived on load, not read from the results file.** `sev.summarise_all(R["findings"])`
is called in `app.py`, `rmi/report.py` *and* `runner.py`. Changing a threshold therefore takes
effect on old result files with no rescan. But **finding titles are baked in at scan time**, so
editing a title string in `rank_findings` does require re-running scans.

**`MATERIALITY_PERCENTILE` in `rmi/severity.py` is the single source of truth** for "big enough to
matter", used by both the category tiles and each tab's verdict line. Two different bars there once
let a finding be material on the overview and immaterial on its own tab.

**Severity is worst-case, not a quantile.** One working exploit is not mitigated by four that fail,
and with a handful of probes a 90th percentile lands on the second largest and hides the finding
the reader needs.

**Never resample rows.** The same questions and templates recur across contrasts, so every interval
resamples whole clusters (`cluster_bootstrap_ci`, `wild_cluster_bootstrap_p` in
`rmi/stats/inference.py`). Row resampling gives intervals several times too narrow.

**`padding` is the length instrument, not a transform.** `LENGTH_INSTRUMENT` in
`rmi/probes/style.py` is excluded from the regression's dose terms because content-free filler at
four intensities *is* the reward-versus-length curve. Giving it a dose coefficient makes it compete
with the length spline and neither is identified.

**Injection ranks on dev and reports on test.** `dev_rank` orders the affixes, `mean_lift` is the
held-out measurement, and `shrinkage` is the gap. Ranking and reporting on the same split is the
winner's curse; the searched stack loses ~0.7 logits across the split while single affixes lose
almost nothing. Exploit examples are selected by *final score*, not lift, because the biggest lifts
come from the answers that started lowest and those still end far below a real answer.

**Cohen's *d* is deliberately absent.** The scorer is deterministic, so its denominator holds no
measurement noise; it measures consistency across hand-written items rather than magnitude, and
diverges for a perfectly uniform effect.

**The identity omnibus must operate on group means.** A max-minus-min over individual name scores
is invariant under permutation, so the test would silently return p = 1.0 and look like a clean
null. `tests/test_inference.py::test_name_level_max_gap_would_be_degenerate` pins this.

## Measured facts about these models

Confirmed by probing, not assumed. Several overturned an obvious design:

- The tokenizer **discards newlines entirely**: `"Hello\n\nWorld"` is two tokens. Bullets and
  headers reach the model only as `-` and `#` characters.
- A literal `[SEP]` typed into an answer becomes the **real separator token**. Transformers'
  `split_special_tokens` does not reliably prevent this for DeBERTa-v2, so the corpus pairs each
  `[SEP]` affix with a lookalike (`[ SEP ]`) that provably cannot emit one, and every row records
  `n_sep_in_answer` as the ground truth.
- Emoji tokenize to real pieces, not `[UNK]`.
- `max_position_embeddings` is 512 but the tokenizer's `model_max_length` is unset, so nothing
  truncates unless asked. Truncation is a confound for any suffix or verbosity result.
- Scoring is bit-identical across runs, which is why the score cache is never stale. Results JSON
  *does* go stale when analysis code changes; that asymmetry is why the sidebar's clear-results
  control deliberately leaves `.rmi_cache/` alone.

## Verification that matters

`tests/test_probes.py::test_pure_length_scorer_reports_no_style_bias` plants a "longer is better"
rule in `StubScorer` and requires the style module to recover the slope and report zero style
effect for every transform. An audit tool that cannot recover a rule it planted itself has no
business reporting findings about a real model. If you touch the style regression, run this first.

When changing the dashboard, check it renders headlessly rather than only reading the diff:

```bash
uv run python -c "
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('app.py', default_timeout=500); at.run()
print('exceptions:', len(at.exception)); [print(e.value) for e in at.exception]"
```

Note that `pandas` attribute access on a column named `transform` returns the DataFrame *method*,
not the column, and silently evaluates to `False` in a filter. Use bracket access throughout.

## Planned work

`~/.claude/plans/purpose-reward-models-are-calm-dongarra.md` holds the design rationale and a
"Deferred to a later version" section covering what was built and then cut, most notably the
warm/blunt tone factor in the sycophancy module. The data is still collected and
`findings.tone_check` still guards against it, but version 1 reports a single tone-averaged
premium. The corpora are pilot-sized (30 style questions, 12 identity templates, 20 sycophancy
scenarios, 4 descriptor templates); phase 2 targets are in the plan.
