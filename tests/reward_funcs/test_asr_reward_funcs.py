from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_reward_module(name: str):
    path = _REPO_ROOT / "examples" / "reward_funcs" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def context_reward():
    return _load_reward_module("context_robustness")


@pytest.fixture(scope="module")
def pronoun_reward():
    return _load_reward_module("asr_cer")


@pytest.fixture(scope="module")
def mixed_reward():
    return _load_reward_module("asr_mixed")


def test_context_reward_is_one_for_a_perfect_hypothesis(context_reward):
    result = context_reward.compute_score(
        data_source="context_rl",
        solution_str="language zh<asr_text>我想参观苏州博物馆。",
        ground_truth="我想参观苏州博物馆。",
        extra_info={
            "must_emit": np.array(["苏州博物馆"], dtype=object),
            "must_reject": np.array(["苏州博务馆"], dtype=object),
        },
    )

    assert result["score"] == pytest.approx(1.0)
    assert result["entity_exact_rate"] == pytest.approx(1.0)
    assert result["reject_success_rate"] == pytest.approx(1.0)
    assert all(isinstance(value, float) for value in result.values())


def test_context_reward_penalizes_emitted_distractor(context_reward):
    labels = {"must_emit": ["苏州博物馆"], "must_reject": ["苏州博务馆"]}
    correct = context_reward.compute_score(
        "context_rl",
        "我想参观苏州博物馆",
        "我想参观苏州博物馆",
        labels,
    )
    distracted = context_reward.compute_score(
        "context_rl",
        "我想参观苏州博务馆",
        "我想参观苏州博物馆",
        labels,
    )

    assert distracted["reject_hits"] == pytest.approx(1.0)
    assert distracted["reject_score"] == pytest.approx(0.0)
    assert distracted["entity_exact_rate"] == pytest.approx(0.0)
    assert distracted["entity_similarity"] > 0.0
    assert distracted["score"] < correct["score"]


def test_context_reward_does_not_reward_an_empty_transcript(context_reward):
    result = context_reward.compute_score(
        "context_rl",
        "",
        "我想参观苏州博物馆",
        {"must_emit": ["苏州博物馆"], "must_reject": ["苏州博务馆"]},
    )

    assert result["accuracy"] == pytest.approx(0.0)
    assert result["reject_success_rate"] == pytest.approx(1.0)
    assert result["reject_gate"] == pytest.approx(0.0)
    assert result["score"] == pytest.approx(0.0)


def test_context_rejection_gate_runs_from_half_to_point_eight(context_reward):
    labels = {"must_emit": [], "must_reject": ["干扰词"]}
    no_credit = context_reward.compute_score(
        "context_rl",
        "abcdeXXXXX",
        "abcdefghij",
        labels,
    )
    full_credit = context_reward.compute_score(
        "context_rl",
        "abcdefghXX",
        "abcdefghij",
        labels,
    )

    assert no_credit["accuracy"] == pytest.approx(0.5)
    assert no_credit["reject_gate"] == pytest.approx(0.0)
    assert full_credit["accuracy"] == pytest.approx(0.8)
    assert full_credit["reject_gate"] == pytest.approx(1.0)


def test_context_reward_accepts_json_serialized_extra_info(context_reward):
    result = context_reward.compute_score(
        "context_rl",
        "去中山公园",
        "去中山公园",
        '{"must_emit":["中山公园"],"must_reject":["中山宫园"]}',
    )

    assert result["entity_total"] == pytest.approx(1.0)
    assert result["reject_total"] == pytest.approx(1.0)
    assert result["score"] == pytest.approx(1.0)


def test_pronoun_reward_is_one_for_exact_order(pronoun_reward):
    result = pronoun_reward.compute_score(
        data_source="cmrc2019_coref",
        solution_str="language zh<asr_text>她把书递给他。",
        ground_truth="她把书递给他。",
        extra_info={"pronoun": np.array(["她", "他"], dtype=object)},
    )

    assert result["score"] == pytest.approx(1.0)
    assert result["pronoun_exact"] == pytest.approx(1.0)
    assert result["pronoun_distance"] == pytest.approx(0.0)
    assert result["he_recall"] == pytest.approx(1.0)
    assert result["she_recall"] == pytest.approx(1.0)
    assert result["it_recall"] == pytest.approx(0.0)
    assert result["he_active"] == pytest.approx(1.0)
    assert result["she_active"] == pytest.approx(1.0)
    assert result["it_active"] == pytest.approx(0.0)
    assert all(isinstance(value, float) for value in result.values())


def test_pronoun_reward_uses_ordered_edit_distance_and_exact_bonus(pronoun_reward):
    result = pronoun_reward.compute_score(
        "cmrc2019_coref",
        "她把书递给她",
        "她把书递给他",
        {"pronoun": ["她", "他"]},
        cer_weight=0.6,
        match_weight=0.4,
        pronoun_exact_bonus=0.5,
    )

    assert result["pronoun_distance"] == pytest.approx(1.0)
    assert result["pronoun_similarity"] == pytest.approx(0.5)
    assert result["pronoun_exact"] == pytest.approx(0.0)
    assert result["pronoun_score"] == pytest.approx(0.25)
    assert result["score"] < result["accuracy"]


def test_pronoun_reward_penalizes_extra_pronouns(pronoun_reward):
    result = pronoun_reward.compute_score(
        "cmrc2019_coref",
        "他告诉她它坏了",
        "他告诉她坏了",
        {"pronoun": ["他", "她"]},
    )

    assert result["pronoun_pred_count"] == pytest.approx(3.0)
    assert result["pronoun_ref_count"] == pytest.approx(2.0)
    assert result["pronoun_distance"] == pytest.approx(1.0)
    assert result["pronoun_exact"] == pytest.approx(0.0)


def test_pronoun_reward_falls_back_to_cer_without_pronoun_labels(pronoun_reward):
    result = pronoun_reward.compute_score(
        "cmrc2019_coref",
        "今天天气很好",
        "今天天气很好",
        {"pronoun": []},
    )

    assert result["score"] == pytest.approx(result["accuracy"])
    assert result["score"] == pytest.approx(1.0)
    assert result["pronoun_active"] == pytest.approx(0.0)
    assert result["he_active"] == pytest.approx(0.0)
    assert result["she_active"] == pytest.approx(0.0)
    assert result["it_active"] == pytest.approx(0.0)


def test_pronoun_class_recalls_follow_edit_alignment(pronoun_reward):
    result = pronoun_reward.compute_score(
        "cmrc2019_coref",
        "她告诉她它坏了",
        "她告诉他它坏了",
        {"pronoun": ["她", "他", "它"]},
    )

    assert result["pronoun_distance"] == pytest.approx(1.0)
    assert result["he_recall"] == pytest.approx(0.0)
    assert result["she_recall"] == pytest.approx(1.0)
    assert result["it_recall"] == pytest.approx(1.0)
    assert result["he_active"] == pytest.approx(1.0)
    assert result["she_active"] == pytest.approx(1.0)
    assert result["it_active"] == pytest.approx(1.0)


def test_mixed_reward_returns_one_fixed_numeric_schema(mixed_reward):
    pronoun = mixed_reward.compute_score(
        "cmrc2019_coref",
        "她把书递给他",
        "她把书递给他",
        {"pronoun": ["她", "他"]},
    )
    context = mixed_reward.compute_score(
        "context_rl",
        "我想参观苏州博物馆",
        "我想参观苏州博物馆",
        {"must_emit": ["苏州博物馆"], "must_reject": ["苏州博务馆"]},
    )

    assert pronoun.keys() == context.keys()
    assert all(isinstance(value, float) for value in pronoun.values())
    assert all(isinstance(value, float) for value in context.values())

    assert pronoun["pronoun_active"] == pytest.approx(1.0)
    assert pronoun["entity_active"] == pytest.approx(0.0)
    assert pronoun["reject_active"] == pytest.approx(0.0)
    assert pronoun["entity_score"] == pytest.approx(0.0)

    assert context["pronoun_active"] == pytest.approx(0.0)
    assert context["entity_active"] == pytest.approx(1.0)
    assert context["reject_active"] == pytest.approx(1.0)
    assert context["pronoun_score"] == pytest.approx(0.0)
    assert context["score"] == pytest.approx(1.0)


def test_mixed_reward_rejects_unknown_data_source(mixed_reward):
    with pytest.raises(ValueError, match="unsupported ASR reward data_source"):
        mixed_reward.compute_score("unknown", "", "", {})
