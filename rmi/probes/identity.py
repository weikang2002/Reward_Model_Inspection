"""Identity bias probe.

Templates are neutral competence assessments in which the *only* thing that varies is the identity
span. Three things keep this from producing a confident false positive.

**Within-group name controls.** Greg versus Brad is the same operation as Greg versus Jamal: swap
one name for another. Comparing names inside a group gives the empirical null distribution of the
name-swap operation directly, with no distributional assumptions. If the spread within groups
matches the spread between groups, there is no race effect whatever a p-value says. This is
rendered next to every between-group gap.

**An omnibus permutation test on group means, stratified by name token count.** Reporting the
largest of six pairwise gaps is a maximum over six estimates and is biased upward. The permutation
handles that selection. It shuffles the name-to-group map rather than shuffling within templates,
because a name belongs to exactly one group, and it shuffles only within blocks of similar token
length, so a rejection means "beyond the effect of name length" rather than being a tokenisation
artifact in disguise.

**Names and templates both treated as random.** Treating the 40 names as fixed is the
language-as-fixed-effect fallacy and would license a claim about race from what is really a claim
about 40 particular strings. Intervals come from a cluster bootstrap over templates, and with few
templates the wild cluster bootstrap is used.

Decision contexts like parole and lending are far outside these reward models' training
distribution, so in-distribution assistant dialogue templates are included and reported separately.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from ..stats.inference import (
    cluster_bootstrap_ci,
    max_group_gap_permutation,
    paired_contrast,
    wild_cluster_bootstrap_p,
    within_template_max_gap_permutation,
)

CORPORA = Path(__file__).resolve().parent.parent / "corpora"


def load_corpus() -> dict:
    return json.loads((CORPORA / "identity.json").read_text())


def build_arms(corpus: dict) -> tuple[list[dict], list[dict]]:
    name_to_group = {n: g for g, names in corpus["name_groups"].items() for n in names}
    arms = []
    for t in corpus["templates"]:
        for name, group in name_to_group.items():
            arms.append({
                "template_id": t["id"], "domain": t["domain"],
                "distribution": t["distribution"], "name": name, "group": group,
                "question": t["question"].replace("{name}", name),
                "text": t["response"].replace("{name}", name),
            })
    desc_arms = []
    for t in corpus["descriptor_templates"]:
        for axis, levels in corpus["descriptor_axes"].items():
            for lvl in levels:
                desc_arms.append({
                    "template_id": t["id"], "domain": t["domain"],
                    "distribution": t["distribution"], "axis": axis, "level": lvl,
                    "question": t["question"].replace("{desc}", lvl),
                    "text": t["response"].replace("{desc}", lvl),
                })
    return arms, desc_arms


def _template_centred(arms: list[dict]) -> None:
    """Remove template difficulty so it cannot leak into identity comparisons."""
    by_t: dict[str, list[dict]] = {}
    for a in arms:
        by_t.setdefault(a["template_id"], []).append(a)
    for rows in by_t.values():
        m = float(np.mean([r["score"] for r in rows]))
        for r in rows:
            r["centred"] = r["score"] - m


def run(scorer, corpus: dict | None = None, *, noise_floor=None, n_perm: int = 10_000,
        n_boot: int = 4000, seed: int = 0) -> dict:
    corpus = corpus or load_corpus()
    arms, desc_arms = build_arms(corpus)

    for group in (arms, desc_arms):
        detailed = scorer.score_detailed([(a["question"], a["text"]) for a in group])
        for a, d in zip(group, detailed):
            a.update(score=d.score, n_answer_tokens=d.n_answer_tokens, truncated=d.truncated)
    _template_centred(arms)
    _template_centred(desc_arms)

    # Token length of each name, used to stratify the permutation so a rejection cannot be
    # explained by some names simply tokenising into more pieces.
    name_tokens = {}
    for name in {a["name"] for a in arms}:
        try:
            name_tokens[name] = len(scorer.tokenizer(name, add_special_tokens=False)["input_ids"])
        except AttributeError:
            name_tokens[name] = len(name.split())

    out = {"name_token_counts": name_tokens, "citation": corpus["citation"],
           "corpus_version": corpus["version"]}

    for subset in ("all", "ood", "id"):
        sel = arms if subset == "all" else [a for a in arms if a["distribution"] == subset]
        if not sel:
            continue
        out[f"names_{subset}"] = _analyse_names(
            sel, name_tokens, n_perm=n_perm, n_boot=n_boot, seed=seed, noise_floor=noise_floor
        )

    out["descriptors"] = _analyse_descriptors(
        desc_arms, n_perm=n_perm, n_boot=n_boot, seed=seed, noise_floor=noise_floor
    )
    out["n_truncated"] = sum(1 for a in arms + desc_arms if a["truncated"])
    out["arms"] = arms
    out["descriptor_arms"] = desc_arms
    return out


def _analyse_names(arms, name_tokens, *, n_perm, n_boot, seed, noise_floor):
    groups = sorted({a["group"] for a in arms})
    templates = sorted({a["template_id"] for a in arms})

    # per-name mean over templates, on template-centred scores
    name_mean, name_group = {}, {}
    for a in arms:
        name_mean.setdefault(a["name"], []).append(a["centred"])
        name_group[a["name"]] = a["group"]
    name_mean = {k: float(np.mean(v)) for k, v in name_mean.items()}

    group_mean = {
        g: float(np.mean([v for n, v in name_mean.items() if name_group[n] == g]))
        for g in groups
    }

    # ---- pairwise group gaps, bootstrapped over templates (the cluster) -------------------
    gaps = []
    for ga, gb in itertools.combinations(groups, 2):
        deltas, clusters = [], []
        for t in templates:
            rows = [a for a in arms if a["template_id"] == t]
            a_scores = [r["score"] for r in rows if r["group"] == ga]
            b_scores = [r["score"] for r in rows if r["group"] == gb]
            if a_scores and b_scores:
                deltas.append(float(np.mean(a_scores) - np.mean(b_scores)))
                clusters.append(t)
        if not deltas:
            continue
        d = np.array(deltas)
        lo, hi = cluster_bootstrap_ci(d, clusters, n_boot=n_boot, seed=seed)
        entry = {
            "group_a": ga, "group_b": gb, "gap": float(d.mean()),
            "ci_low": lo, "ci_high": hi, "n_templates": len(deltas),
            # With ~12 templates, cluster-robust SEs are downward-biased; the wild cluster
            # bootstrap is the standard remedy.
            "p_wild": wild_cluster_bootstrap_p(d, clusters, n_boot=min(n_boot, 3000), seed=seed),
        }
        if noise_floor is not None:
            entry["noise_percentile"] = noise_floor.percentile_of(entry["gap"])
            entry["exceeds_noise_floor"] = noise_floor.exceeds_floor(entry["gap"])
        gaps.append(entry)
    gaps.sort(key=lambda g: -abs(g["gap"]))

    # ---- the empirical null: swapping a name for another name in the SAME group -----------
    within, between = [], []
    for t in templates:
        rows = [a for a in arms if a["template_id"] == t]
        by_g: dict[str, list[float]] = {}
        for r in rows:
            by_g.setdefault(r["group"], []).append(r["score"])
        for g, vals in by_g.items():
            within.extend(abs(x - y) for x, y in itertools.combinations(vals, 2))
        for ga, gb in itertools.combinations(sorted(by_g), 2):
            between.extend(abs(x - y) for x in by_g[ga] for y in by_g[gb])
    within = np.array(within)
    between = np.array(between)

    # A matched control. Comparing a single name swap against a difference of *group means* would
    # be comparing different aggregation levels: a group mean averages ten names, so its noise is
    # much smaller than one swap's. Splitting each group's names into two random halves and
    # differencing the half-means produces a null at exactly the level the group gaps live at.
    rng = np.random.default_rng(seed)
    half_split = []
    names_by_group: dict[str, list[str]] = {}
    for n, g in name_group.items():
        names_by_group.setdefault(g, []).append(n)
    for _ in range(2000):
        for g, names in names_by_group.items():
            if len(names) < 4:
                continue
            perm = rng.permutation(names)
            h = len(perm) // 2
            a = np.mean([name_mean[n] for n in perm[:h]])
            b = np.mean([name_mean[n] for n in perm[h:]])
            half_split.append(abs(a - b))
    half_split = np.array(half_split) if half_split else np.array([0.0])

    omnibus = max_group_gap_permutation(
        [a["score"] for a in arms], [a["name"] for a in arms],
        [a["group"] for a in arms], [a["template_id"] for a in arms],
        strata=[name_tokens[a["name"]] for a in arms], n_perm=n_perm, seed=seed,
    )

    return {
        "group_means": group_mean,
        "name_means": name_mean,
        "name_group": name_group,
        "pairwise_gaps": gaps,
        "omnibus_permutation": omnibus,
        "within_group_control": {
            "mean_abs_diff": float(within.mean()),
            "p95_abs_diff": float(np.percentile(within, 95)),
            "max_abs_diff": float(within.max()),
            "n": int(within.size),
        },
        "half_split_control": {
            "mean_abs_diff": float(half_split.mean()),
            "p95_abs_diff": float(np.percentile(half_split, 95)),
            "n": int(half_split.size),
            "note": "difference between the means of two random halves of the same group, "
                    "which is the same aggregation level as a between-group gap",
        },
        "between_group_spread": {
            "mean_abs_diff": float(between.mean()),
            "p95_abs_diff": float(np.percentile(between, 95)),
            "n": int(between.size),
        },
        # If swapping within a group moves scores as much as swapping across groups, the
        # between-group gaps are not evidence of an identity effect.
        "between_vs_within_ratio": float(between.mean() / within.mean()) if within.mean() else None,
        "n_templates": len(templates),
        "n_names": len(name_mean),
    }


def _analyse_descriptors(desc_arms, *, n_perm, n_boot, seed, noise_floor):
    out = {}
    for axis in sorted({a["axis"] for a in desc_arms}):
        sel = [a for a in desc_arms if a["axis"] == axis]
        levels = sorted({a["level"] for a in sel})
        templates = sorted({a["template_id"] for a in sel})
        means = {
            lvl: float(np.mean([a["centred"] for a in sel if a["level"] == lvl]))
            for lvl in levels
        }
        hi = max(means, key=means.get)
        lo = min(means, key=means.get)
        deltas, clusters = [], []
        for t in templates:
            rows = {a["level"]: a["score"] for a in sel if a["template_id"] == t}
            if hi in rows and lo in rows:
                deltas.append(rows[hi] - rows[lo])
                clusters.append(t)
        ci = cluster_bootstrap_ci(np.array(deltas), clusters, n_boot=n_boot, seed=seed) \
            if len(deltas) > 1 else (float("nan"), float("nan"))
        perm = within_template_max_gap_permutation(
            [a["score"] for a in sel], [a["level"] for a in sel],
            [a["template_id"] for a in sel], n_perm=n_perm, seed=seed,
        )
        entry = {
            "axis": axis, "level_means": means, "n_levels": len(levels),
            "n_templates": len(templates),
            "max_gap": perm["statistic"], "highest": hi, "lowest": lo,
            "gap_ci_low": ci[0], "gap_ci_high": ci[1],
            "permutation": perm,
        }
        if noise_floor is not None:
            entry["noise_percentile"] = noise_floor.percentile_of(perm["statistic"])
            entry["exceeds_noise_floor"] = noise_floor.exceeds_floor(perm["statistic"])
        out[axis] = entry
    return out
