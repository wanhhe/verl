# coding=utf-8
"""Custom reward function for context-ASR GRPO: entity match + 1-CER.

This is a thin verl adapter over ``Asrproj-train/training/rl/reward.py``'s
``ContextRewardModel``, so the reward the policy climbs and the eval score the
run is judged by are computed by the same code.  The two terms (see the
reference module for the full rationale):

1. ``match``: how many ``must_emit`` entities the hypothesis emitted, plus how
   many ``must_reject`` distractors it stayed away from.  Rejection credit is
   only paid out when ``accuracy >= reject_credit_floor``, so silence or a
   collapsed transcript cannot collect the whole rejection term.
2. ``accuracy``: ``1 - CER`` (clamped, never negative), which stops the model
   from gaming ``match`` by dumping every entity into the output.

Weights are renormalised per sample; a sample with no entities is scored on
accuracy alone.  The Qwen3-ASR ``language XX<asr_text>`` output prefix is
stripped before scoring (same as examples/reward_funcs/asr_cer.py).

Dependencies resolved automatically by the reference module: ``compute_wer``
(installed, or the checkout at the project's DEFAULT_COMPUTE_WER_PATH) and
``zhconv`` for traditional->simplified unification (optional; missing just
warns and skips the conversion — the context-RL data is already simplified).

Config:
    reward.custom_reward_function.path = examples/reward_funcs/context_robustness.py
    reward.custom_reward_function.name = compute_score
    # reward_kwargs mirror RewardConfig fields in the reference module:
    reward.custom_reward_function.reward_kwargs.match_weight = 0.5
    reward.custom_reward_function.reward_kwargs.accuracy_weight = 0.5
    reward.custom_reward_function.reward_kwargs.emit_weight = 1.0
    reward.custom_reward_function.reward_kwargs.reject_weight = 1.5
    reward.custom_reward_function.reward_kwargs.max_cer = 1.0
    reward.custom_reward_function.reward_kwargs.reject_credit_floor = 0.3

``extra_info`` comes from the context_rl parquet and carries ``must_emit``,
``must_reject`` and ``language`` (default "zh" -> per-character scoring; "en"
scoring is per-word).
"""

import importlib.util
import os
import re
import sys

_LANGUAGE_PREFIX_RE = re.compile(r"^language\s+\S+<asr_text>\s*", flags=re.IGNORECASE)

# Where the reference ContextRewardModel lives; override with the env var if the
# checkout moves.
_REWARD_PY = os.getenv(
    "CONTEXT_RL_REWARD_PY",
    "/quark_speech_nas_zjk/users/wangzilin/Asrproj-train/training/rl/reward.py",
)

# RewardConfig fields that may be passed through reward_kwargs.
_CONFIG_FIELDS = (
    "match_weight",
    "accuracy_weight",
    "emit_weight",
    "reject_weight",
    "max_cer",
    "reject_credit_floor",
    "to_simplified",
)

_model_cache: dict = {}


def _get_model(config_kwargs: dict):
    """Load ContextRewardModel once per process, reusing its normalizer caches."""
    key = tuple(sorted((k, config_kwargs[k]) for k in _CONFIG_FIELDS if k in config_kwargs))
    if key not in _model_cache:
        if not os.path.isfile(_REWARD_PY):
            raise FileNotFoundError(
                f"cannot locate the reference reward module at {_REWARD_PY}; "
                "set CONTEXT_RL_REWARD_PY to the checkout of "
                "Asrproj-train/training/rl/reward.py"
            )
        spec = importlib.util.spec_from_file_location("_context_reward_model", _REWARD_PY)
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("_context_reward_model", module)
        spec.loader.exec_module(module)

        config_cls = module.RewardConfig
        config_kwargs = {k: v for k, v in config_kwargs.items() if k in _CONFIG_FIELDS}
        _model_cache[key] = module.ContextRewardModel(config=config_cls(**config_kwargs))
    return _model_cache[key]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> float | dict:
    """Reward = (match_weight * match + accuracy_weight * accuracy) / weight_sum.

    Returns a dict with ``score`` plus the per-component breakdown for logging;
    the reward manager uses ``score`` as the reward.
    """
    hypothesis = _LANGUAGE_PREFIX_RE.sub("", solution_str or "").strip()
    reference = (ground_truth or "").strip()
    info = extra_info or {}
    # extra_info may arrive as a JSON string (verl) or pre-parsed with numpy
    # arrays (pandas round-trip); normalise to plain lists/None.
    must_emit = info.get("must_emit")
    must_reject = info.get("must_reject")

    model = _get_model(kwargs)
    result = model.score(
        hypothesis,
        reference,
        must_emit=list(must_emit) if must_emit is not None else (),
        must_reject=list(must_reject) if must_reject is not None else (),
        language=info.get("language") or "zh",
    )
    return {
        "score": result.reward,
        "accuracy": result.accuracy,
        "cer": result.cer,
        "match": result.match,
        "emit_hit": result.emit_hit,
        "emit_total": result.emit_total,
        "reject_ok": result.reject_ok,
        "reject_total": result.reject_total,
    }
