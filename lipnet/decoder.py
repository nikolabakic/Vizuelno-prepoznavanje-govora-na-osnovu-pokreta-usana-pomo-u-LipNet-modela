"""CTC prefix beam search and a small character n-gram language model.

The training baseline intentionally keeps VIPL's greedy decoder.  This module
adds a decoder-only experiment that can reuse cached model emissions without
changing or retraining LipNet.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
from typing import Protocol, Type

import torch

from lipnet.dataset import MyDataset, SerbianDataset


NEGATIVE_INFINITY = -math.inf


def _logadd(*values: float) -> float:
    finite = [value for value in values if value != NEGATIVE_INFINITY]
    if not finite:
        return NEGATIVE_INFINITY
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


class IncrementalLanguageModel(Protocol):
    """Minimal interface needed for shallow fusion during beam search."""

    def log_probability(self, prefix: Sequence[int], token: int) -> float:
        """Return log P(token | prefix)."""


@dataclass(frozen=True)
class BeamSearchConfig:
    """Serializable decoder configuration used by the Phase-7 experiment."""

    beam_width: int = 25
    lm_weight: float = 0.0
    word_bonus: float = 0.0
    token_topk: int | None = None

    def __post_init__(self) -> None:
        if self.beam_width <= 0:
            raise ValueError("beam_width mora biti pozitivan")
        if self.lm_weight < 0:
            raise ValueError("lm_weight ne sme biti negativan")
        if self.token_topk is not None and self.token_topk <= 0:
            raise ValueError("token_topk mora biti pozitivan ili None")


class CharacterNGramLM:
    """Add-k backoff character n-gram model over non-blank CTC token IDs.

    Every order from unigram through ``order`` is counted.  At inference the
    longest context observed in the training transcripts is used, with additive
    smoothing over the declared character vocabulary.  No validation or test
    transcript is required to fit the model.
    """

    BOS = -1

    def __init__(
        self,
        order: int = 5,
        smoothing: float = 0.1,
        vocabulary: Iterable[int] = range(1, 29),
    ) -> None:
        if order <= 0:
            raise ValueError("N-gram order mora biti pozitivan")
        if smoothing <= 0:
            raise ValueError("Smoothing mora biti pozitivan")
        normalized_vocabulary = tuple(sorted({int(token) for token in vocabulary}))
        if not normalized_vocabulary or 0 in normalized_vocabulary:
            raise ValueError("LM vokabular mora sadržati non-blank CTC tokene")
        self.order = int(order)
        self.smoothing = float(smoothing)
        self.vocabulary = normalized_vocabulary
        self._vocabulary_set = set(normalized_vocabulary)
        self._counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        self._totals: Counter[tuple[int, ...]] = Counter()
        self.training_sequences = 0

    def fit(self, sequences: Iterable[Sequence[int]]) -> "CharacterNGramLM":
        self._counts.clear()
        self._totals.clear()
        self.training_sequences = 0
        bos = (self.BOS,) * (self.order - 1)
        for sequence in sequences:
            tokens = tuple(int(token) for token in sequence)
            unknown = set(tokens) - self._vocabulary_set
            if unknown:
                raise ValueError(f"LM trening sadrži tokene van vokabulara: {sorted(unknown)}")
            history = bos + tokens
            for index, token in enumerate(tokens, start=len(bos)):
                available = history[:index]
                for context_length in range(self.order):
                    context = tuple(available[-context_length:]) if context_length else ()
                    self._counts[context][token] += 1
                    self._totals[context] += 1
            self.training_sequences += 1
        if self.training_sequences == 0:
            raise ValueError("LM zahteva najmanje jednu trening sekvencu")
        return self

    def log_probability(self, prefix: Sequence[int], token: int) -> float:
        if self.training_sequences == 0:
            raise RuntimeError("LM mora biti fit-ovan pre dekodiranja")
        token = int(token)
        if token not in self._vocabulary_set:
            raise ValueError(f"Token {token} nije u LM vokabularu")
        padded = (self.BOS,) * (self.order - 1) + tuple(int(item) for item in prefix)
        maximum_context = min(self.order - 1, len(padded))
        context: tuple[int, ...] = ()
        for context_length in range(maximum_context, -1, -1):
            candidate = tuple(padded[-context_length:]) if context_length else ()
            if self._totals[candidate] > 0:
                context = candidate
                break
        numerator = self._counts[context][token] + self.smoothing
        denominator = self._totals[context] + self.smoothing * len(self.vocabulary)
        return math.log(numerator / denominator)


def _candidate_tokens(row: torch.Tensor, token_topk: int | None, blank: int) -> list[int]:
    classes = int(row.numel())
    if token_topk is None or token_topk >= classes:
        return list(range(classes))
    candidates = row.topk(token_topk).indices.tolist()
    if blank not in candidates:
        candidates.append(blank)
    return [int(token) for token in candidates]


def ctc_prefix_beam_search(
    log_probabilities: torch.Tensor,
    *,
    config: BeamSearchConfig = BeamSearchConfig(),
    language_model: IncrementalLanguageModel | None = None,
    blank: int = 0,
    space_token: int = 1,
) -> tuple[int, ...]:
    """Return the highest-scoring CTC label prefix for one ``(T,C)`` sample."""
    if log_probabilities.ndim != 2:
        raise ValueError("Očekivane su log-verovatnoće oblika (T,C)")
    time_steps, classes = map(int, log_probabilities.shape)
    if time_steps <= 0 or classes <= 1:
        raise ValueError(f"Neispravan CTC oblik {tuple(log_probabilities.shape)}")
    if not 0 <= blank < classes:
        raise ValueError(f"Blank indeks {blank} je van C={classes}")
    if config.lm_weight and language_model is None:
        raise ValueError("lm_weight zahteva language_model")

    rows = log_probabilities.detach().to(device="cpu", dtype=torch.float64)
    beams: dict[tuple[int, ...], tuple[float, float]] = {
        (): (0.0, NEGATIVE_INFINITY)
    }

    for row in rows:
        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}

        def update(prefix: tuple[int, ...], *, blank_score: float | None = None,
                   nonblank_score: float | None = None) -> None:
            old_blank, old_nonblank = next_beams.get(
                prefix, (NEGATIVE_INFINITY, NEGATIVE_INFINITY)
            )
            next_beams[prefix] = (
                _logadd(old_blank, blank_score if blank_score is not None else NEGATIVE_INFINITY),
                _logadd(
                    old_nonblank,
                    nonblank_score if nonblank_score is not None else NEGATIVE_INFINITY,
                ),
            )

        candidates = _candidate_tokens(row, config.token_topk, blank)
        blank_probability = float(row[blank])
        for prefix, (probability_blank, probability_nonblank) in beams.items():
            total = _logadd(probability_blank, probability_nonblank)
            update(prefix, blank_score=total + blank_probability)

            for token in candidates:
                if token == blank:
                    continue
                token_probability = float(row[token])
                last_token = prefix[-1] if prefix else None
                if token == last_token:
                    update(prefix, nonblank_score=probability_nonblank + token_probability)
                    extension_base = probability_blank
                else:
                    extension_base = total
                if extension_base == NEGATIVE_INFINITY:
                    continue

                extended = prefix + (token,)
                extension_score = extension_base + token_probability
                if language_model is not None and config.lm_weight:
                    extension_score += config.lm_weight * language_model.log_probability(
                        prefix, token
                    )
                if token == space_token:
                    extension_score += config.word_bonus
                update(extended, nonblank_score=extension_score)

        ranked = sorted(
            next_beams.items(),
            key=lambda item: _logadd(*item[1]),
            reverse=True,
        )
        beams = dict(ranked[: config.beam_width])

    def final_score(item: tuple[tuple[int, ...], tuple[float, float]]) -> float:
        prefix, probabilities = item
        score = _logadd(*probabilities)
        if prefix and prefix[-1] != space_token:
            score += config.word_bonus
        return score

    return max(beams.items(), key=final_score)[0]


def prefix_beam_decode(
    logits: torch.Tensor,
    *,
    config: BeamSearchConfig = BeamSearchConfig(),
    language_model: IncrementalLanguageModel | None = None,
    dataset_type: Type[MyDataset] = SerbianDataset,
    output_lengths: torch.Tensor | Sequence[int] | None = None,
    blank: int = 0,
    space_token: int = 1,
) -> list[str]:
    """Decode batched ``(B,T,C)`` logits using CTC prefix beam search."""
    if logits.ndim != 3:
        raise ValueError("Očekivani su logits oblika (B,T,C)")
    batch_size, time_steps, _ = map(int, logits.shape)
    if output_lengths is None:
        lengths = [time_steps] * batch_size
    elif isinstance(output_lengths, torch.Tensor):
        lengths = [int(value) for value in output_lengths.detach().cpu().tolist()]
    else:
        lengths = [int(value) for value in output_lengths]
    if len(lengths) != batch_size:
        raise ValueError("Broj output dužina mora odgovarati batch dimenziji")
    if any(length <= 0 or length > time_steps for length in lengths):
        raise ValueError(f"Neispravne output dužine {lengths} za T={time_steps}")

    log_probabilities = logits.detach().log_softmax(dim=-1).cpu()
    decoded: list[str] = []
    for row, length in zip(log_probabilities, lengths):
        tokens = ctc_prefix_beam_search(
            row[:length],
            config=config,
            language_model=language_model,
            blank=blank,
            space_token=space_token,
        )
        text = dataset_type.arr2txt(tokens, start=1)
        decoded.append(" ".join(text.split()))
    return decoded
