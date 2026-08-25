"""CPU tests for decoder-only Phase-7 experiments."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
import math

import pytest
import torch

from lipnet.decoder import (
    BeamSearchConfig,
    CharacterNGramLM,
    ctc_prefix_beam_search,
    prefix_beam_decode,
)


def collapse_ctc(path: tuple[int, ...], blank: int = 0) -> tuple[int, ...]:
    collapsed = []
    previous = None
    for token in path:
        if token != previous and token != blank:
            collapsed.append(token)
        previous = token
    return tuple(collapsed)


def brute_force_best(probabilities: torch.Tensor) -> tuple[int, ...]:
    totals: dict[tuple[int, ...], float] = defaultdict(float)
    time_steps, classes = probabilities.shape
    for path in product(range(classes), repeat=time_steps):
        path_probability = math.prod(
            float(probabilities[index, token]) for index, token in enumerate(path)
        )
        totals[collapse_ctc(path)] += path_probability
    return max(totals, key=totals.__getitem__)


@pytest.mark.parametrize("seed", range(5))
def test_prefix_beam_matches_exhaustive_ctc_probability(seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    probabilities = torch.rand(4, 3, generator=generator)
    probabilities /= probabilities.sum(dim=-1, keepdim=True)
    expected = brute_force_best(probabilities)
    actual = ctc_prefix_beam_search(
        probabilities.log(),
        config=BeamSearchConfig(beam_width=100),
    )
    assert actual == expected


def test_prefix_beam_honours_real_output_length() -> None:
    logits = torch.full((1, 4, 29), -10.0)
    logits[0, 0, 2] = 10.0  # a
    logits[0, 1, 0] = 10.0  # blank
    logits[0, 2:, 3] = 10.0  # padded b must not be decoded
    assert prefix_beam_decode(
        logits,
        output_lengths=torch.tensor([2]),
        config=BeamSearchConfig(beam_width=5),
    ) == ["a"]


def test_character_lm_prefers_seen_continuation() -> None:
    lm = CharacterNGramLM(order=2, smoothing=0.01, vocabulary=(1, 2, 3)).fit(
        [(2, 3)] * 10 + [(2, 2)]
    )
    assert lm.log_probability((2,), 3) > lm.log_probability((2,), 2)

    logits = torch.full((1, 3, 4), -8.0)
    logits[0, 0, 2] = 8.0  # a
    logits[0, 1, 0] = 8.0  # blank permits aa
    logits[0, 2, 2] = 3.0  # acoustic model slightly prefers aa
    logits[0, 2, 3] = 2.8  # LM should promote ab

    without_lm = ctc_prefix_beam_search(
        logits[0].log_softmax(dim=-1),
        config=BeamSearchConfig(beam_width=10),
    )
    with_lm = ctc_prefix_beam_search(
        logits[0].log_softmax(dim=-1),
        config=BeamSearchConfig(beam_width=10, lm_weight=2.0),
        language_model=lm,
    )
    assert without_lm == (2, 2)
    assert with_lm == (2, 3)


def test_lm_rejects_blank_and_unseen_tokens() -> None:
    with pytest.raises(ValueError, match="non-blank"):
        CharacterNGramLM(vocabulary=(0, 1, 2))
    lm = CharacterNGramLM(order=2, vocabulary=(1, 2)).fit([(1, 2)])
    with pytest.raises(ValueError, match="van vokabulara"):
        lm.fit([(1, 3)])


def test_decoder_configuration_validation() -> None:
    with pytest.raises(ValueError, match="beam_width"):
        BeamSearchConfig(beam_width=0)
    with pytest.raises(ValueError, match="lm_weight"):
        BeamSearchConfig(lm_weight=-1)
    with pytest.raises(ValueError, match="language_model"):
        ctc_prefix_beam_search(
            torch.zeros(2, 3),
            config=BeamSearchConfig(lm_weight=1.0),
        )
