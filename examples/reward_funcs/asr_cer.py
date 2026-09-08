# coding=utf-8
"""Custom reward function for ASR GRPO: weighted 1-CER + pronoun match reward.

Two components (both normalized to [0, 1]):

1. CER reward: ``1 - CER(hypothesis, reference)``. The `language XX<asr_text>`
   prefix emitted by Qwen3-ASR is stripped before scoring.

2. Pronoun match reward: extracts the Chinese pronouns 他/她/它 from the model
   output in order and computes the normalized Levenshtein edit distance against
   the dataset's ``pronoun`` field (passed via ``extra_info``). Closer pronoun
   usage -> higher reward.

Total: ``cer_weight * cer_reward + match_weight * match_reward`` (weights are
configurable via ``reward.custom_reward_function.reward_kwargs``).

Config:
    reward.custom_reward_function.path = examples/reward_funcs/asr_cer.py
    reward.custom_reward_function.name = compute_score
    reward.custom_reward_function.reward_kwargs.cer_weight = 0.7
    reward.custom_reward_function.reward_kwargs.match_weight = 0.3
"""

import re

import jiwer

_LANGUAGE_PREFIX_RE = re.compile(r"^language\s+\S+<asr_text>\s*", flags=re.IGNORECASE)
_PRONOUN_CHARS = frozenset("他她它")


def _strip_language_prefix(text: str) -> str:
    """Remove the `language XX<asr_text>` prefix emitted by Qwen3-ASR."""
    return _LANGUAGE_PREFIX_RE.sub("", text)


def _extract_pronouns(text: str) -> list[str]:
    """Extract 他/她/它 characters from text, preserving order."""
    return [ch for ch in text if ch in _PRONOUN_CHARS]


def _levenshtein(a: list[str], b: list[str]) -> int:
    """Levenshtein edit distance between two small sequences."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(
                prev[j] + 1,          # deletion
                cur[j - 1] + 1,       # insertion
                prev[j - 1] + (0 if a[i - 1] == b[j - 1] else 1),  # substitution
            )
        prev = cur
    return prev[n]


def _clip01(value: float) -> float:
    """Clamp a score to [0, 1] to guard against overflow/underflow values."""
    return min(1.0, max(0.0, value))


def _pronoun_match_score(extracted: list[str], reference: list[str]) -> float:
    """Normalized pronoun match: 1 - edit_distance / max(len_ref, len_ext, 1)."""
    if not reference and not extracted:
        return 1.0
    dist = _levenshtein(extracted, reference)
    return _clip01(1.0 - dist / max(len(reference), len(extracted), 1))


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    cer_weight: float = 0.5,
    match_weight: float = 0.5,
    **kwargs,
) -> float | dict:
    """Reward = cer_weight * (1 - CER) + match_weight * pronoun_match.

    Returns a dict with ``score`` plus per-component breakdown for logging when
    the reward manager supports it; the manager uses ``score`` as the reward.
    """
    hypothesis = _strip_language_prefix(solution_str or "").strip()
    reference = (ground_truth or "").strip()

    # ---- CER reward ----
    cer_reward = 0.0
    if reference and hypothesis:
        try:
            # CER can exceed 1 (extra insertions vs. a short reference), so
            # clamp 1 - CER to [0, 1].
            cer_reward = _clip01(1.0 - jiwer.cer(reference, hypothesis))
        except Exception:
            cer_reward = 0.0

    # ---- pronoun match reward ----
    reference_pronouns = list((extra_info or {}).get("pronoun") or [])
    extracted_pronouns = _extract_pronouns(hypothesis)
    match_reward = _pronoun_match_score(extracted_pronouns, reference_pronouns)

    total = cer_weight * cer_reward + match_weight * match_reward

    return {
        "score": total,
        "cer_reward": cer_reward,
        "pronoun_match_reward": match_reward,
        "extracted_pronouns": "".join(extracted_pronouns),
        "reference_pronouns": "".join(reference_pronouns),
    }
