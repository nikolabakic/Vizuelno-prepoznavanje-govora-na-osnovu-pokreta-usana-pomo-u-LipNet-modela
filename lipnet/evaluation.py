"""Statistical and structured error analysis for decoder comparisons."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import editdistance
import numpy as np


def _validate_parallel_texts(*collections: Sequence[str]) -> int:
    lengths = {len(collection) for collection in collections}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("Tekstualne kolekcije moraju biti neprazne i iste dužine")
    return next(iter(lengths))


def paired_bootstrap_delta(
    baseline_predictions: Sequence[str],
    candidate_predictions: Sequence[str],
    references: Sequence[str],
    *,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int]:
    """Paired bootstrap confidence intervals for candidate-minus-baseline deltas."""
    sample_count = _validate_parallel_texts(
        baseline_predictions, candidate_predictions, references
    )
    if iterations <= 0:
        raise ValueError("Broj bootstrap iteracija mora biti pozitivan")
    if not 0 < confidence < 1:
        raise ValueError("confidence mora biti između 0 i 1")

    reference_words = np.asarray([max(len(text.split()), 1) for text in references])
    reference_characters = np.asarray([max(len(text), 1) for text in references])
    baseline_word_errors = np.asarray(
        [
            editdistance.eval(prediction.split(), reference.split())
            for prediction, reference in zip(baseline_predictions, references)
        ]
    )
    candidate_word_errors = np.asarray(
        [
            editdistance.eval(prediction.split(), reference.split())
            for prediction, reference in zip(candidate_predictions, references)
        ]
    )
    baseline_character_errors = np.asarray(
        [
            editdistance.eval(prediction, reference)
            for prediction, reference in zip(baseline_predictions, references)
        ]
    )
    candidate_character_errors = np.asarray(
        [
            editdistance.eval(prediction, reference)
            for prediction, reference in zip(candidate_predictions, references)
        ]
    )
    baseline_exact = np.asarray(
        [prediction == reference for prediction, reference in zip(baseline_predictions, references)]
    )
    candidate_exact = np.asarray(
        [prediction == reference for prediction, reference in zip(candidate_predictions, references)]
    )

    rng = np.random.default_rng(seed)
    deltas = np.empty((iterations, 3), dtype=np.float64)
    for iteration in range(iterations):
        indices = rng.integers(0, sample_count, size=sample_count)
        word_denominator = reference_words[indices].sum()
        character_denominator = reference_characters[indices].sum()
        deltas[iteration, 0] = (
            candidate_word_errors[indices].sum() - baseline_word_errors[indices].sum()
        ) / word_denominator
        deltas[iteration, 1] = (
            candidate_character_errors[indices].sum()
            - baseline_character_errors[indices].sum()
        ) / character_denominator
        deltas[iteration, 2] = (
            candidate_exact[indices].mean() - baseline_exact[indices].mean()
        )

    tail = (1.0 - confidence) / 2.0
    observed = np.asarray(
        [
            (candidate_word_errors.sum() - baseline_word_errors.sum())
            / reference_words.sum(),
            (candidate_character_errors.sum() - baseline_character_errors.sum())
            / reference_characters.sum(),
            candidate_exact.mean() - baseline_exact.mean(),
        ]
    )
    result: dict[str, float | int] = {
        "samples": sample_count,
        "iterations": iterations,
        "confidence": confidence,
    }
    for column, name in enumerate(("wer", "cer", "sentence_exact_match")):
        low, high = np.quantile(deltas[:, column], (tail, 1.0 - tail))
        result[f"{name}_delta"] = float(observed[column])
        result[f"{name}_delta_ci_low"] = float(low)
        result[f"{name}_delta_ci_high"] = float(high)
        result[f"{name}_candidate_better_probability"] = float(
            np.mean(deltas[:, column] < 0)
            if name != "sentence_exact_match"
            else np.mean(deltas[:, column] > 0)
        )
    return result


def _align_words(reference: Sequence[str], prediction: Sequence[str]) -> list[tuple[str | None, str | None]]:
    rows, columns = len(reference) + 1, len(prediction) + 1
    costs = np.zeros((rows, columns), dtype=np.int32)
    operations = np.empty((rows, columns), dtype=object)
    for row in range(1, rows):
        costs[row, 0] = row
        operations[row, 0] = "delete"
    for column in range(1, columns):
        costs[0, column] = column
        operations[0, column] = "insert"

    for row in range(1, rows):
        for column in range(1, columns):
            substitution = costs[row - 1, column - 1] + (
                reference[row - 1] != prediction[column - 1]
            )
            deletion = costs[row - 1, column] + 1
            insertion = costs[row, column - 1] + 1
            # Tuple order makes diagonal alignment deterministic on equal cost.
            cost, _, operation = min(
                (substitution, 0, "diagonal"),
                (deletion, 1, "delete"),
                (insertion, 2, "insert"),
            )
            costs[row, column] = cost
            operations[row, column] = operation

    alignment: list[tuple[str | None, str | None]] = []
    row, column = len(reference), len(prediction)
    while row or column:
        operation = operations[row, column]
        if operation == "diagonal":
            alignment.append((reference[row - 1], prediction[column - 1]))
            row -= 1
            column -= 1
        elif operation == "delete":
            alignment.append((reference[row - 1], None))
            row -= 1
        else:
            alignment.append((None, prediction[column - 1]))
            column -= 1
    alignment.reverse()
    return alignment


def slot_error_analysis(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    slot_names: Sequence[str],
) -> dict[str, Any]:
    """Measure aligned correctness for fixed-position AI-SPEAK sentences."""
    sample_count = _validate_parallel_texts(predictions, references)
    names = tuple(slot_names)
    if not names:
        raise ValueError("Potreban je najmanje jedan naziv pozicije")
    slots = {
        name: {"correct": 0, "substitutions": 0, "deletions": 0}
        for name in names
    }
    eligible_samples = 0
    insertions = 0
    for prediction, reference in zip(predictions, references):
        reference_tokens = reference.split()
        if len(reference_tokens) != len(names):
            continue
        eligible_samples += 1
        reference_index = 0
        for reference_token, prediction_token in _align_words(
            reference_tokens, prediction.split()
        ):
            if reference_token is None:
                insertions += 1
                continue
            slot = slots[names[reference_index]]
            if prediction_token is None:
                slot["deletions"] += 1
            elif prediction_token == reference_token:
                slot["correct"] += 1
            else:
                slot["substitutions"] += 1
            reference_index += 1

    for slot in slots.values():
        denominator = slot["correct"] + slot["substitutions"] + slot["deletions"]
        slot["accuracy"] = slot["correct"] / max(denominator, 1)
    return {
        "samples": sample_count,
        "eligible_samples": eligible_samples,
        "skipped_samples": sample_count - eligible_samples,
        "insertions": insertions,
        "slots": slots,
    }


def slot_confusion_analysis(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    slot_names: Sequence[str],
    vocabularies: Mapping[str, Sequence[str]] | None = None,
    other_label: str = "ostalo",
    deletion_label: str = "brisanje",
) -> dict[str, Any]:
    """Build aligned, report-ready confusion counts for fixed sentence slots.

    Rows follow each slot's reference vocabulary. Prediction columns add
    ``other_label`` for out-of-vocabulary words and ``deletion_label`` for an
    aligned deletion. Raw substitution pairs are retained for focused error
    tables such as the two AI-SPEAK letter positions.
    """
    sample_count = _validate_parallel_texts(predictions, references)
    names = tuple(slot_names)
    if not names:
        raise ValueError("Potreban je najmanje jedan naziv pozicije")
    if other_label == deletion_label:
        raise ValueError("Oznake za ostalo i brisanje moraju biti različite")

    eligible_references = [
        reference.split()
        for reference in references
        if len(reference.split()) == len(names)
    ]
    derived_vocabularies = {
        name: tuple(dict.fromkeys(tokens[index] for tokens in eligible_references))
        for index, name in enumerate(names)
    }
    resolved_vocabularies: dict[str, tuple[str, ...]] = {}
    for name in names:
        supplied = None if vocabularies is None else vocabularies.get(name)
        labels = tuple(supplied) if supplied is not None else derived_vocabularies[name]
        if not labels or len(labels) != len(set(labels)):
            raise ValueError(f"Vokabular za slot {name!r} mora biti neprazan i jedinstven")
        if other_label in labels or deletion_label in labels:
            raise ValueError(f"Rezervisane oznake se pojavljuju u vokabularu slota {name!r}")
        resolved_vocabularies[name] = labels

    counts = {
        name: {
            reference: Counter()
            for reference in resolved_vocabularies[name]
        }
        for name in names
    }
    raw_substitutions: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    eligible_samples = 0
    insertions = 0
    for prediction, reference in zip(predictions, references):
        reference_tokens = reference.split()
        if len(reference_tokens) != len(names):
            continue
        eligible_samples += 1
        reference_index = 0
        for reference_token, prediction_token in _align_words(
            reference_tokens, prediction.split()
        ):
            if reference_token is None:
                insertions += 1
                continue
            slot_name = names[reference_index]
            slot_vocabulary = resolved_vocabularies[slot_name]
            if reference_token not in counts[slot_name]:
                raise ValueError(
                    f"Referentna vrednost {reference_token!r} nije u vokabularu slota {slot_name!r}"
                )
            if prediction_token is None:
                prediction_label = deletion_label
            elif prediction_token in slot_vocabulary:
                prediction_label = prediction_token
            else:
                prediction_label = other_label
            counts[slot_name][reference_token][prediction_label] += 1
            if prediction_token is not None and prediction_token != reference_token:
                raw_substitutions[slot_name][(reference_token, prediction_token)] += 1
            reference_index += 1

    slots: dict[str, Any] = {}
    for name in names:
        reference_labels = resolved_vocabularies[name]
        prediction_labels = reference_labels + (other_label, deletion_label)
        matrix = np.asarray(
            [
                [counts[name][reference][prediction] for prediction in prediction_labels]
                for reference in reference_labels
            ],
            dtype=np.int64,
        )
        denominators = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix,
            denominators,
            out=np.zeros_like(matrix, dtype=np.float64),
            where=denominators != 0,
        )
        substitutions = [
            {"reference": reference, "prediction": prediction, "count": count}
            for (reference, prediction), count in sorted(
                raw_substitutions[name].items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
        ]
        slots[name] = {
            "reference_labels": list(reference_labels),
            "prediction_labels": list(prediction_labels),
            "counts": matrix.tolist(),
            "row_normalized": normalized.tolist(),
            "substitution_pairs": substitutions,
        }

    return {
        "samples": sample_count,
        "eligible_samples": eligible_samples,
        "skipped_samples": sample_count - eligible_samples,
        "insertions": insertions,
        "other_label": other_label,
        "deletion_label": deletion_label,
        "slots": slots,
    }
