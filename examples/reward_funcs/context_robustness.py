# coding=utf-8
"""Self-contained reward for context-ASR hard-negative samples.

Each sample stores target entities in ``extra_info['must_emit']`` and
distractors in ``extra_info['must_reject']``. The reward combines:

* transcription accuracy: ``clip(1 - CER, 0, 1)``;
* entity score: exact entity recall plus a small character-edit similarity
  term, which gives near misses a dense learning signal;
* rejection score: the fraction of distractors absent from the hypothesis.

Rejection credit is gated by transcription accuracy: it is zero at or below
``reject_gate_start`` and reaches full credit at ``reject_gate_full``. This
prevents an empty or collapsed transcript from earning a high score simply
because it contains no distractors. Components with no labels are left out of
the weighted average.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

_LANGUAGE_PREFIX_RE = re.compile(r"^language\s+\S+<asr_text>\s*", flags=re.IGNORECASE)


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _normalize_text(text: Any) -> str:
    """Normalize text for CER and substring matching."""
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(
        char
        for char in normalized
        if unicodedata.category(char)[0] not in {"C", "P", "Z"}
    )


def _strip_language_prefix(text: Any) -> str:
    return _LANGUAGE_PREFIX_RE.sub("", str(text or "")).strip()


def _levenshtein(left: Sequence[str], right: Sequence[str]) -> int:
    """Levenshtein distance using O(min(len(left), len(right))) memory."""
    if len(left) < len(right):
        left, right = right, left
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def _transcription_scores(hypothesis: str, reference: str) -> tuple[float, float]:
    """Return ``(accuracy, CER)`` after ASR-oriented normalization."""
    normalized_hypothesis = _normalize_text(hypothesis)
    normalized_reference = _normalize_text(reference)
    if not normalized_reference:
        cer = 0.0 if not normalized_hypothesis else 1.0
    else:
        cer = _levenshtein(normalized_reference, normalized_hypothesis) / len(normalized_reference)
    return _clip01(1.0 - cer), float(cer)


def _coerce_extra_info(extra_info: Any) -> Mapping[str, Any]:
    if isinstance(extra_info, Mapping):
        return extra_info
    if isinstance(extra_info, str):
        try:
            parsed = json.loads(extra_info)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, Mapping) else {}
    return {}


def _as_string_list(value: Any) -> list[str]:
    """Convert parquet/JSON list representations to clean, unique strings."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                decoded = None
            value = decoded if isinstance(decoded, list) else [stripped]
        else:
            value = [stripped]
    elif not isinstance(value, Sequence):
        value = [value]

    result = []
    seen = set()
    for item in value:
        text = str(item).strip()
        normalized = _normalize_text(text)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(text)
    return result


def _substring_edit_distance(pattern: str, text: str) -> int:
    """Minimum edits needed to match ``pattern`` against any text substring."""
    if not pattern:
        return 0
    if not text:
        return len(pattern)

    # A zero-initialized first row lets the match begin anywhere in ``text``.
    previous = [0] * (len(text) + 1)
    for pattern_index, pattern_char in enumerate(pattern, start=1):
        current = [pattern_index]
        for text_index, text_char in enumerate(text, start=1):
            current.append(
                min(
                    previous[text_index] + 1,
                    current[text_index - 1] + 1,
                    previous[text_index - 1] + (pattern_char != text_char),
                )
            )
        previous = current
    return min(previous)


def _entity_metrics(
    hypothesis: str,
    must_emit: list[str],
    exact_weight: float,
) -> tuple[float, float, float, int]:
    if not must_emit:
        return 0.0, 0.0, 0.0, 0

    normalized_hypothesis = _normalize_text(hypothesis)
    exact_hits = 0
    similarities = []
    for entity in must_emit:
        normalized_entity = _normalize_text(entity)
        is_exact = normalized_entity in normalized_hypothesis
        exact_hits += int(is_exact)
        distance = _substring_edit_distance(normalized_entity, normalized_hypothesis)
        similarities.append(_clip01(1.0 - distance / len(normalized_entity)))

    exact_rate = exact_hits / len(must_emit)
    similarity = sum(similarities) / len(similarities)
    entity_score = exact_weight * exact_rate + (1.0 - exact_weight) * similarity
    return entity_score, exact_rate, similarity, exact_hits


def _reject_metrics(hypothesis: str, must_reject: list[str]) -> tuple[float, int]:
    if not must_reject:
        return 0.0, 0
    normalized_hypothesis = _normalize_text(hypothesis)
    reject_hits = sum(
        _normalize_text(distractor) in normalized_hypothesis for distractor in must_reject
    )
    return 1.0 - reject_hits / len(must_reject), reject_hits


def _weighted_average(components: Sequence[tuple[float, float]]) -> float:
    if any(weight < 0.0 for _, weight in components):
        raise ValueError("reward weights must be non-negative")
    denominator = sum(weight for _, weight in components)
    if denominator <= 0.0:
        raise ValueError("at least one active reward weight must be positive")
    return sum(value * weight for value, weight in components) / denominator


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | str | None = None,
    cer_weight: float = 0.40,
    entity_weight: float = 0.35,
    reject_weight: float = 0.25,
    entity_exact_weight: float = 0.75,
    reject_gate_start: float = 0.50,
    reject_gate_full: float = 0.80,
    **_: Any,
) -> dict[str, float]:
    """Compute context-ASR reward and numeric metrics for verl logging."""
    del data_source
    if not 0.0 <= entity_exact_weight <= 1.0:
        raise ValueError("entity_exact_weight must be in [0, 1]")
    if not 0.0 <= reject_gate_start < reject_gate_full <= 1.0:
        raise ValueError("reject gate must satisfy 0 <= start < full <= 1")

    hypothesis = _strip_language_prefix(solution_str)
    reference = str(ground_truth or "").strip()
    info = _coerce_extra_info(extra_info)
    must_emit = _as_string_list(info.get("must_emit"))
    must_reject = _as_string_list(info.get("must_reject"))

    accuracy, cer = _transcription_scores(hypothesis, reference)
    entity_score, entity_exact_rate, entity_similarity, entity_hits = _entity_metrics(
        hypothesis,
        must_emit,
        entity_exact_weight,
    )
    reject_success_rate, reject_hits = _reject_metrics(hypothesis, must_reject)

    # Low-quality transcripts receive no credit for merely omitting distractors.
    # Credit then increases smoothly until the transcript reaches useful ASR
    # quality, avoiding a discontinuous reward jump at a single threshold.
    reject_gate = _clip01(
        (accuracy - reject_gate_start) / (reject_gate_full - reject_gate_start)
    )
    reject_score = reject_success_rate * reject_gate

    active_components = [(accuracy, cer_weight)]
    if must_emit:
        active_components.append((entity_score, entity_weight))
    if must_reject:
        active_components.append((reject_score, reject_weight))
    score = _weighted_average(active_components)

    return {
        "score": float(score),
        "accuracy": float(accuracy),
        "cer": float(cer),
        "entity_score": float(entity_score),
        "entity_exact_rate": float(entity_exact_rate),
        "entity_similarity": float(entity_similarity),
        "entity_hits": float(entity_hits),
        "entity_total": float(len(must_emit)),
        "reject_score": float(reject_score),
        "reject_success_rate": float(reject_success_rate),
        "reject_hits": float(reject_hits),
        "reject_total": float(len(must_reject)),
        "reject_gate": float(reject_gate),
    }
