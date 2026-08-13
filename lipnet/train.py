"""Modernized, small pieces of VIPL ``main.py`` needed through Phase 4.

The full fine-tuning loop belongs to Phase 5.  This module deliberately stops
at checkpoint transfer, greedy decode, metrics and a verified CTC backward
step. Source commit: 40209e09c49553c00c25c7d41faa3706aea3c625.
See ``LICENSE.vipl`` and ``docs/upstream-diff.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Type

import numpy as np
import torch
import torch.nn as nn

from .dataset import MyDataset, SerbianDataset, validate_ctc_batch


@dataclass(frozen=True)
class CheckpointAudit:
    loaded: tuple[str, ...]
    skipped_shape: tuple[str, ...]
    missing_in_checkpoint: tuple[str, ...]
    unexpected_in_checkpoint: tuple[str, ...]

    def summary(self) -> str:
        return (
            f"loaded={len(self.loaded)}, skipped_shape={list(self.skipped_shape)}, "
            f"missing={list(self.missing_in_checkpoint)}, "
            f"unexpected={list(self.unexpected_in_checkpoint)}"
        )


def _read_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint nije state_dict: {type(payload)}")
    return {str(key).removeprefix("module."): value for key, value in payload.items()}


def load_checkpoint_strict(model: nn.Module, path: str | Path) -> CheckpointAudit:
    """Phase 1: every GRID parameter must load, with identical shapes."""
    checkpoint = _read_state_dict(path)
    incompatibility = model.load_state_dict(checkpoint, strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(f"Neusaglašen GRID checkpoint: {incompatibility}")
    return CheckpointAudit(tuple(sorted(checkpoint)), (), (), ())


def load_vipl_transfer(
    model: nn.Module,
    path: str | Path,
    allowed_skipped: tuple[str, ...] = ("FC.weight", "FC.bias"),
) -> CheckpointAudit:
    """VIPL shape-filtered load with a strict backbone audit.

    The Serbian model may skip only the two incompatible output-head tensors.
    Any missing convolutional or BiGRU parameter aborts the transfer.
    """
    checkpoint = _read_state_dict(path)
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in checkpoint.items()
        if key in current and value.shape == current[key].shape
    }
    skipped_shape = tuple(
        sorted(key for key, value in checkpoint.items() if key in current and value.shape != current[key].shape)
    )
    missing = tuple(sorted(key for key in current if key not in checkpoint))
    unexpected = tuple(sorted(key for key in checkpoint if key not in current))
    skipped_total = set(skipped_shape) | set(missing) | set(unexpected)
    if skipped_total != set(allowed_skipped):
        raise RuntimeError(
            "Transfer bi preskočio više od srpskog head-a: "
            f"shape={list(skipped_shape)}, missing={list(missing)}, unexpected={list(unexpected)}"
        )
    incompatibility = model.load_state_dict(compatible, strict=False)
    if set(incompatibility.missing_keys) != set(allowed_skipped) or incompatibility.unexpected_keys:
        raise RuntimeError(f"Neočekivan load_state_dict rezultat: {incompatibility}")
    return CheckpointAudit(tuple(sorted(compatible)), skipped_shape, missing, unexpected)


def greedy_decode(
    logits: torch.Tensor,
    dataset_type: Type[MyDataset] = SerbianDataset,
) -> list[str]:
    """Upstream argmax + repeat/blank collapse for batched ``(B,T,C)`` logits."""
    token_ids = logits.argmax(dim=-1).detach().cpu()
    return [dataset_type.ctc_arr2txt(row, start=1) for row in token_ids]


def reference_text(batch: dict[str, torch.Tensor]) -> list[str]:
    texts = []
    for row, length in zip(batch["txt"], batch["txt_len"]):
        texts.append(SerbianDataset.arr2txt(row[: int(length)], start=1))
    return texts


def sequence_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    return {
        "wer": float(np.mean(SerbianDataset.wer(predictions, references))),
        "cer": float(np.mean(SerbianDataset.cer(predictions, references))),
        "sentence_exact_match": float(
            np.mean([prediction == reference for prediction, reference in zip(predictions, references)])
        ),
    }


def ctc_loss(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    criterion: nn.CTCLoss | None = None,
) -> torch.Tensor:
    """Compute VIPL CTC loss with real lengths and an explicit feasibility check."""
    output_lengths = batch["vid_len"].to(dtype=torch.long, device="cpu")
    output_lengths = output_lengths.clamp(max=logits.shape[1])
    validate_ctc_batch(batch, output_lengths)
    if criterion is None:
        criterion = nn.CTCLoss(blank=0, zero_infinity=False)
    return criterion(
        logits.transpose(0, 1).log_softmax(dim=-1),
        batch["txt"],
        output_lengths,
        batch["txt_len"].to(dtype=torch.long, device="cpu"),
    )


def backward_smoke_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[float, tuple[int, ...]]:
    """One Phase-4 forward/loss/backward pass; intentionally no optimizer step."""
    model.train()
    model.zero_grad(set_to_none=True)
    device_batch = {
        key: value.to(device) if key in {"vid", "txt"} else value
        for key, value in batch.items()
    }
    logits = model(device_batch["vid"])
    loss = ctc_loss(logits, device_batch)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"CTC loss nije konačan: {loss.item()}")
    loss.backward()
    missing_gradients = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    nonfinite_gradients = [
        name for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if missing_gradients or nonfinite_gradients:
        raise RuntimeError(
            f"Gradijent audit: missing={missing_gradients}, nonfinite={nonfinite_gradients}"
        )
    return float(loss.detach().cpu()), tuple(logits.shape)
