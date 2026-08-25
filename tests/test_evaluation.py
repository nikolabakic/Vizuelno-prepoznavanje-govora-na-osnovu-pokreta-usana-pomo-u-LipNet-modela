"""Tests for paired decoder statistics and structured error analysis."""

from __future__ import annotations

import pytest

from lipnet.evaluation import paired_bootstrap_delta, slot_error_analysis


def test_paired_bootstrap_is_zero_for_identical_predictions() -> None:
    references = ["a b", "c d", "e f"]
    predictions = ["a", "c x", "e f"]
    result = paired_bootstrap_delta(
        predictions, predictions, references, iterations=200, seed=7
    )
    for metric in ("wer", "cer", "sentence_exact_match"):
        assert result[f"{metric}_delta"] == 0.0
        assert result[f"{metric}_delta_ci_low"] == 0.0
        assert result[f"{metric}_delta_ci_high"] == 0.0


def test_paired_bootstrap_detects_consistently_better_candidate() -> None:
    references = ["a b"] * 20
    baseline = ["x y"] * 20
    candidate = ["a b"] * 20
    result = paired_bootstrap_delta(
        baseline, candidate, references, iterations=200, seed=3
    )
    assert result["wer_delta"] == -1.0
    assert result["wer_delta_ci_high"] < 0
    assert result["wer_candidate_better_probability"] == 1.0


def test_slot_analysis_aligns_insertions_and_deletions() -> None:
    references = ["cmd a left b monday one", "cmd c right d friday two"]
    predictions = [
        "cmd extra a left b monday one",  # one insertion, all slots correct
        "cmd c right friday two",  # deletion in the fourth slot
    ]
    result = slot_error_analysis(
        predictions,
        references,
        slot_names=("command", "letter1", "direction", "letter2", "day", "number"),
    )
    assert result["eligible_samples"] == 2
    assert result["insertions"] == 1
    assert result["slots"]["letter2"]["deletions"] == 1
    assert result["slots"]["command"]["accuracy"] == 1.0


def test_slot_analysis_skips_nonconforming_references() -> None:
    result = slot_error_analysis(
        ["a b", "a"],
        ["a b", "a"],
        slot_names=("first", "second"),
    )
    assert result["eligible_samples"] == 1
    assert result["skipped_samples"] == 1


def test_evaluation_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="iste dužine"):
        paired_bootstrap_delta(["a"], ["a", "b"], ["a"])
