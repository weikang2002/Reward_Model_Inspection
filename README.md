# Reward Model Inspection

A local dashboard that loads any HuggingFace `AutoModelForSequenceClassification` reward model and
probes it for hidden bias, reward hacking, and prefix/suffix injection attacks.

Reward models decide what a policy learns during RLHF. Whatever the reward model scores highly is
what the trained model drifts toward, so a bias or an exploitable shortcut in the reward model gets
copied into the policy and amplified.

```bash
uv sync                      # creates .venv and installs everything
uv run streamlit run app.py  # dashboard at http://localhost:8501
```

Pick a model in the sidebar and press **Run scan**. On an M1 laptop a standard scan of
`reward-model-deberta-v3-large-v2` takes a few minutes the first time and is near-instant
afterwards, because every score is cached in SQLite. From the command line:

```bash
uv run python -c "from rmi.runner import run_scan; run_scan('OpenAssistant/reward-model-deberta-v3-base')"
```

## The one thing to understand first

This reward model's score moves by about **0.7 logits when you reword an answer without changing
its meaning**. The entire gap between a good answer and a poor one is about 3. So a "bias" of 0.3
logits is not a finding, it is rephrasing.

Every number in the dashboard is therefore reported as a percentile of that **paraphrase noise
floor**, measured on the model under test, alongside the raw logits. The floor is computed first
and gates everything else. Without it, this tool would confidently report noise as bias — and
during development, it did.

The second yardstick sits next to it: how often the model agrees with **real human preference
judgments**, measured on held-out `Anthropic/hh-rlhf` pairs. The large OpenAssistant model manages
63%. The base model manages 50.6%, which is chance, and it also fails its own model card's worked
example. A bias finding about a model that is not tracking human judgment at all means something
quite different from the same finding about a model that is.

## What it probes

| Module | Question | How it avoids fooling itself |
|---|---|---|
| **Noise floor** | How much does the score move for no reason? | Meaning-preserving paraphrases, plus byte-level perturbations that must score exactly zero |
| **Identity** | Does swapping a name or descriptor change the score? | Within-group name controls at the matched aggregation level; omnibus permutation on group means, stratified by name token length |
| **Sycophancy** | Does agreeing with the user pay? | Agreement crossed with tone, so warmth is held fixed; measured as a slope across three levels of user insistence, not just a level |
| **Style & length** | Is surface style rewarded? | Dose-response ladders; content-free padding doubles as the length instrument; genuine elaboration versus filler at matched length |
| **Injection** | Can an affix make garbage score well? | Neutral matched-length controls, lookalike controls for special tokens, and a dev/test split for all discovery |

## Things the controls caught

These are why the controls exist, not hypotheticals.

- **The `[SEP]` attack is mostly a confound.** Typing `[SEP]` into an answer really does insert the
  model's genuine separator token, and appending `[SEP] <a good answer>` to garbage lifts the score
  by about 5 logits. But the control that appends the *same good answer with no separator at all*
  recovers 4.5 of that. The forged control token is worth around 0.45, which is below the rewording
  noise floor even though it is extremely consistent (95% of items, p ≈ 1e-36). Statistical
  significance and practical magnitude point in different directions, and the dashboard shows both.
- **Flattery looks penalised until you control for length.** The naive contrast says −1.62 logits.
  After adjusting for the length it adds, it is **+0.34**. The model's dislike of extra length was
  hiding a genuine preference in the opposite direction.
- **Sycophancy is invisible as an average.** The main effect of agreement is +0.04 (p = 0.32). But
  when the user is emotionally invested, the reward for agreeing rises by **+0.80** (p = 0.001),
  flipping the model from mildly preferring corrections to preferring agreement. A level-only test
  reports nothing here.
- **The tokenizer discards newlines entirely.** `"Hello\n\nWorld"` is two tokens. Bullets and
  headers reach the model only as `-` and `#` characters, so formatting effects are weaker here
  than on a model that sees whitespace.
- **The two checkpoints differ in kind, not degree.** The large model penalises every
  content-free style transform. The base model rewards self-praise (+1.30) and buzzwords (+1.18) —
  the classic reward-hacking pattern.

## Statistics

Scoring is deterministic (repeated runs are bit-identical), so the only randomness is which items
were written, and the clustering structure is the entire inferential content.

- **Cluster resampling.** The same questions and templates appear in many contrasts. Every interval
  resamples whole clusters, not rows; row resampling would give intervals several times too narrow.
  BCa where there are enough clusters, wild cluster bootstrap where there are few.
- **Westfall-Young step-down max-T** for multiplicity within each module: valid under arbitrary
  dependence and more powerful than Benjamini-Hochberg under the positive correlation shared items
  create.
- **No Cohen's *d*.** With a deterministic scorer its denominator holds no measurement noise, so it
  measures consistency rather than magnitude and diverges for a perfectly uniform effect.
- **Split-sample discovery.** Affix ranking and beam search run on development prompts; every
  reported attack number comes from held-out prompts.
- **Two severity numbers per category**, never one: prevalence and a tail severity. A mean over
  contrasts can be diluted by adding null contrasts without the model changing at all.

## Layout

```
app.py                 Streamlit dashboard
rmi/scoring.py         model loading, length-sorted batching, SQLite cache, per-row provenance
rmi/calibration.py     temperature fit and human-agreement accuracy on held-out hh-rlhf
rmi/probes/            noise_floor, identity, sycophancy, style, injection
rmi/stats/             cluster bootstrap, permutation tests, Westfall-Young, FE spline regression
rmi/corpora/           every stimulus, as versioned JSON
rmi/runner.py          orchestration, multiplicity, finding ranking, results schema
rmi/report.py          standalone HTML export
tests/                 38 tests, including planted-rule recovery
```

Run the tests with `uv run pytest -q`. The one that matters most is
`test_pure_length_scorer_reports_no_style_bias`: a scorer whose only rule is "longer is better"
must produce the right length slope and a style effect of zero for every transform. An audit tool
that cannot recover a rule it planted itself has no business reporting findings about a real model.

## Limitations

The stimuli are hand-written and not blind-authored, and these are pilot-sized corpora: 30
questions, 12 identity templates, 20 sycophancy scenarios, 41 affixes. Descriptor axes rest on four
templates and should be read as indicative. Decision-context templates (hiring, lending, parole)
are far outside these models' training distribution, so assistant-dialogue templates are reported
separately.
