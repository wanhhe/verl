# coding=utf-8
"""Reward dispatcher for mixed pronoun and lexical context-ASR training.

The verl reward manager invokes one custom function per sample and provides
the row's ``data_source``.  This module routes semantic pronoun samples to
``asr_cer`` and lexical entity/distractor samples to ``context_robustness``.
Every call returns the same compact set of numeric metrics so mixed-source
batches can be stored and aggregated without ragged auxiliary arrays.
"""

from __future__ import annotations

from typing import Any

from examples.reward_funcs.asr_cer import compute_score as compute_pronoun_score
from examples.reward_funcs.context_robustness import compute_score as compute_context_score


_METRIC_KEYS = (
    "score",
    "accuracy",
    "cer",
    "pronoun_score",
    "pronoun_distance",
    "he_recall",
    "she_recall",
    "it_recall",
    "he_active",
    "she_active",
    "it_active",
    "entity_score",
    "entity_exact_rate",
    "reject_score",
    "reject_success_rate",
    "pronoun_active",
    "entity_active",
    "reject_active",
)


def _select_metrics(result: dict[str, float], **active_metrics: float) -> dict[str, float]:
    metrics = {key: 0.0 for key in _METRIC_KEYS}
    metrics.update({key: float(value) for key, value in result.items() if key in metrics})
    metrics.update({key: float(value) for key, value in active_metrics.items()})
    return metrics


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | str | None = None,
    cer_weight: float = 0.60,
    match_weight: float = 0.40,
    pronoun_exact_bonus: float = 0.50,
    context_cer_weight: float = 0.40,
    entity_weight: float = 0.35,
    reject_weight: float = 0.25,
    entity_exact_weight: float = 0.75,
    reject_gate_start: float = 0.50,
    reject_gate_full: float = 0.80,
    **_: Any,
) -> dict[str, float]:
    """Dispatch by data source and return a fixed metric schema."""
    source = str(data_source or "")
    if source == "cmrc2019_coref":
        result = compute_pronoun_score(
            data_source=source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            cer_weight=cer_weight,
            match_weight=match_weight,
            pronoun_exact_bonus=pronoun_exact_bonus,
        )
        return _select_metrics(result)

    if source == "context_rl":
        result = compute_context_score(
            data_source=source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            cer_weight=context_cer_weight,
            entity_weight=entity_weight,
            reject_weight=reject_weight,
            entity_exact_weight=entity_exact_weight,
            reject_gate_start=reject_gate_start,
            reject_gate_full=reject_gate_full,
        )
        return _select_metrics(
            result,
            entity_active=float(result["entity_total"] > 0.0),
            reject_active=float(result["reject_total"] > 0.0),
        )

    raise ValueError(f"unsupported ASR reward data_source: {source!r}")
