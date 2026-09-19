# coding=utf-8
"""Self-contained reward for ASR samples with an ordered ``pronoun`` label.

The hypothesis and reference are scored with character error rate. In
addition, every 他/她/它 character is extracted from the hypothesis and compared
with the ordered sequence in ``extra_info['pronoun']``. The pronoun component
combines normalized Levenshtein similarity with an explicit exact-match bonus:

``pronoun_score = exact_bonus * exact + (1 - exact_bonus) * similarity``

This gives useful partial credit while making the correct count, identity, and
order distinctly better than every near miss. Extra pronouns are penalized by
the edit distance as well.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

_LANGUAGE_PREFIX_RE = re.compile(r"^language\s+\S+<asr_text>\s*", flags=re.IGNORECASE)
_PRONOUN_CHARS = frozenset("他她它")
_PRONOUN_METRIC_NAMES = {
    "他": ("he_recall", "he_active"),
    "她": ("she_recall", "she_active"),
    "它": ("it_recall", "it_active"),
}


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _strip_language_prefix(text: Any) -> str:
    return _LANGUAGE_PREFIX_RE.sub("", str(text or "")).strip()


def _normalize_text(text: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(
        char
        for char in normalized
        if unicodedata.category(char)[0] not in {"C", "P", "Z"}
    )


def _levenshtein(left: Sequence[str], right: Sequence[str]) -> int:
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


def _pronoun_sequence(value: Any) -> list[str]:
    """Extract an ordered pronoun sequence from list/ndarray/JSON/string data."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, list):
                value = decoded
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, Sequence):
        items = value
    else:
        items = [value]
    return [char for item in items for char in str(item) if char in _PRONOUN_CHARS]


def _pronoun_class_recalls(reference: list[str], predicted: list[str]) -> dict[str, float]:
    """Return class recalls from a minimum-edit alignment of two pronoun sequences.

    When several minimum-edit alignments exist, prefer the one with the most
    exact matches.  This keeps the class recalls consistent with the ordered
    Levenshtein reward while still crediting correctly aligned subsequences.
    Classes absent from the reference receive recall zero and an explicit
    ``*_active`` value of zero so downstream aggregation can mask them.
    """
    rows = len(reference) + 1
    columns = len(predicted) + 1
    costs = [[0] * columns for _ in range(rows)]
    matches = [[0] * columns for _ in range(rows)]
    parents = [[""] * columns for _ in range(rows)]

    for row in range(1, rows):
        costs[row][0] = row
        parents[row][0] = "delete"
    for column in range(1, columns):
        costs[0][column] = column
        parents[0][column] = "insert"

    operation_priority = {"match": 0, "substitute": 1, "delete": 2, "insert": 3}
    for row in range(1, rows):
        for column in range(1, columns):
            is_match = reference[row - 1] == predicted[column - 1]
            diagonal_operation = "match" if is_match else "substitute"
            candidates = [
                (
                    costs[row - 1][column - 1] + (not is_match),
                    matches[row - 1][column - 1] + int(is_match),
                    diagonal_operation,
                ),
                (costs[row - 1][column] + 1, matches[row - 1][column], "delete"),
                (costs[row][column - 1] + 1, matches[row][column - 1], "insert"),
            ]
            cost, match_count, operation = min(
                candidates,
                key=lambda item: (item[0], -item[1], operation_priority[item[2]]),
            )
            costs[row][column] = cost
            matches[row][column] = match_count
            parents[row][column] = operation

    matched_by_class = {pronoun: 0 for pronoun in _PRONOUN_METRIC_NAMES}
    row = len(reference)
    column = len(predicted)
    while row or column:
        operation = parents[row][column]
        if operation == "match":
            matched_by_class[reference[row - 1]] += 1
            row -= 1
            column -= 1
        elif operation == "substitute":
            row -= 1
            column -= 1
        elif operation == "delete":
            row -= 1
        elif operation == "insert":
            column -= 1
        else:  # pragma: no cover - only reachable for an invalid DP table
            raise RuntimeError(f"invalid pronoun alignment operation: {operation!r}")

    result: dict[str, float] = {}
    for pronoun, (recall_name, active_name) in _PRONOUN_METRIC_NAMES.items():
        reference_count = reference.count(pronoun)
        result[active_name] = float(reference_count > 0)
        result[recall_name] = (
            float(matched_by_class[pronoun] / reference_count) if reference_count else 0.0
        )
    return result


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
    cer_weight: float = 0.60,
    match_weight: float = 0.40,
    pronoun_exact_bonus: float = 0.50,
    **_: Any,
) -> dict[str, float]:
    """Compute CER plus ordered-pronoun reward and numeric logging metrics."""
    del data_source
    if not 0.0 <= pronoun_exact_bonus <= 1.0:
        raise ValueError("pronoun_exact_bonus must be in [0, 1]")

    hypothesis = _strip_language_prefix(solution_str)
    reference = str(ground_truth or "").strip()
    info = _coerce_extra_info(extra_info)

    accuracy, cer = _transcription_scores(hypothesis, reference)
    predicted_pronouns = _pronoun_sequence(hypothesis)
    reference_pronouns = _pronoun_sequence(info.get("pronoun"))

    pronoun_distance = _levenshtein(reference_pronouns, predicted_pronouns)
    max_pronoun_length = max(len(reference_pronouns), len(predicted_pronouns), 1)
    pronoun_similarity = _clip01(1.0 - pronoun_distance / max_pronoun_length)
    pronoun_exact = float(predicted_pronouns == reference_pronouns)
    pronoun_score = (
        pronoun_exact_bonus * pronoun_exact
        + (1.0 - pronoun_exact_bonus) * pronoun_similarity
    )
    pronoun_class_metrics = _pronoun_class_recalls(reference_pronouns, predicted_pronouns)

    active_components = [(accuracy, cer_weight)]
    if reference_pronouns:
        active_components.append((pronoun_score, match_weight))
    score = _weighted_average(active_components)

    result = {
        "score": float(score),
        "accuracy": float(accuracy),
        "cer": float(cer),
        "pronoun_score": float(pronoun_score),
        "pronoun_similarity": float(pronoun_similarity),
        "pronoun_exact": float(pronoun_exact),
        "pronoun_distance": float(pronoun_distance),
        "pronoun_ref_count": float(len(reference_pronouns)),
        "pronoun_pred_count": float(len(predicted_pronouns)),
        "pronoun_active": float(bool(reference_pronouns)),
    }
    result.update(pronoun_class_metrics)
    return result
