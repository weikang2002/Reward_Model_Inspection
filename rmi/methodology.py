"""The method write-up and the limitations, written once.

The dashboard and the standalone HTML report both carry this text. It used to live twice, as
Markdown in ``app.py`` and as HTML in ``rmi/report.py``, and the two drifted: both still described
effects as a percentile of the paraphrase floor long after that was replaced by a multiple of the
rewording bar at each effect's own sample size.

The bodies are therefore plain prose with no inline markup, so a renderer only has to wrap them.
Emphasis inside a paragraph that already opens with a bold lead-in earned nothing and was the one
thing forcing two formats.
"""

from __future__ import annotations

# (lead-in, body). The lead-in is bolded by both renderers; the body is plain.
SECTIONS: list[tuple[str, str]] = [
    ("Scoring",
     "The reward model emits one scalar per (prompt, answer) pair. Scores are not comparable "
     "across different prompts, so every contrast here is paired within a prompt. Scoring is "
     "deterministic: repeated runs give bit-identical results, so the only randomness in the "
     "project is which items were written."),
    ("What a logit is worth",
     "A raw logit means nothing on its own, and dividing by a hand-authored good-minus-poor gap "
     "would be worse than nothing, because the size of that unit is set by how bad the poor "
     "answers were written to be. The headline unit is instead a multiple of what rewording "
     "alone could produce at the effect's own sample size: rewording an answer without changing "
     "its meaning moves the score, but in an arbitrary direction, so it shrinks as one over the "
     "square root of the number of comparisons averaged, while a real bias does not. Raw logits "
     "and a calibrated preference probability are reported alongside. The second uses the fact "
     "that these models are trained with a pairwise ranking loss, so a within-prompt score "
     "difference already estimates a log-odds; one temperature is fitted against held-out human "
     "preference data to make that reading honest."),
    ("Clustering",
     "The same questions and templates are reused across many contrasts, which makes the "
     "observations dependent. Every confidence interval resamples whole clusters (questions, "
     "templates, scenarios) rather than rows; resampling rows would give intervals several times "
     "too narrow. Bias-corrected and accelerated intervals are used where there are enough "
     "clusters, and the wild cluster bootstrap where there are few."),
    ("Multiplicity",
     "Dozens of contrasts are tested. Within each module, adjusted p-values come from "
     "Westfall-Young step-down max-T permutation, which is valid under arbitrary dependence "
     "between contrasts and more powerful than Benjamini-Hochberg under the positive correlation "
     "that shared items create. The permutation flips the sign of all of a cluster's deltas at "
     "once, and the same draws are reused across contrasts so their correlation is preserved."),
    ("Effect sizes",
     "Mean difference with a cluster bootstrap interval, plus the win rate and the spread across "
     "items. Cohen's d is deliberately not reported: because the scorer is deterministic, its "
     "denominator contains no measurement noise, so it measures how consistent an effect is "
     "across hand-written items rather than how large it is, and it diverges to infinity for a "
     "perfectly uniform effect."),
    ("Directionality",
     "Identity is two-sided, since a shift either way is bias. Sycophancy is one-sided. Style "
     "transforms that add no information are one-sided, because any reward for them is unearned; "
     "transforms that might genuinely improve an answer are two-sided and reported as "
     "preferences rather than as bias."),
    ("Selection",
     "Reporting the largest of several gaps is biased upward. The identity module handles this "
     "with an omnibus permutation test on group means; the reward-hacking module handles it by "
     "doing all affix ranking and search on development prompts, then reporting only held-out "
     "numbers."),
]

LIMITATIONS: list[tuple[str, str]] = [
    ("These stimuli are hand-written and not blind-authored.",
     "A pair that differs in more than the intended variable produces a confident false finding. "
     "The paraphrase floor and the within-group name controls bound how much that can matter, "
     "but they do not eliminate it."),
    ("This is the pilot corpus size.",
     "Descriptor axes in particular rest on only a few templates and should be read as "
     "indicative."),
    ("Decision-context templates are far outside these models' training distribution,",
     "which was web question answering, summarisation and assistant dialogue. That is why "
     "assistant-dialogue templates are reported separately."),
    ("Per-prompt percentile references are built from a modest number of genuine answers,",
     "so the attack-success curves are coarse at the upper percentiles."),
]
