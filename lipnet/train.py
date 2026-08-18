"""Modernized VIPL ``main.py`` helpers used through Phase 5.

Phase 5 adds the smallest complete fine-tuning loop around the already verified
checkpoint transfer, greedy decode and CTC loss. Source commit:
40209e09c49553c00c25c7d41faa3706aea3c625.
See ``LICENSE.vipl`` and ``docs/upstream-diff.md``.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Type

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.utils.data import Dataset

from .dataset import MyDataset, SerbianDataset, minimum_ctc_steps, validate_ctc_batch


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


@dataclass(frozen=True)
class FineTuneConfig:
    """Serializable Phase-5 baseline configuration."""

    max_epochs: int = 30
    warmup_epochs: int = 3
    early_stopping_patience: int = 5
    batch_size: int = 2
    backbone_lr: float = 2e-5
    head_lr: float = 1e-4
    num_workers: int = 2
    random_seed: int = 0


@dataclass(frozen=True)
class EpochResult:
    loss: float
    wer: float
    cer: float
    sentence_exact_match: float
    samples: int
    predictions: tuple[str, ...]
    references: tuple[str, ...]

    def metrics(self) -> dict[str, float]:
        return {
            "loss": self.loss,
            "wer": self.wer,
            "cer": self.cer,
            "sentence_exact_match": self.sentence_exact_match,
        }


@dataclass(frozen=True)
class CTCFilterReport:
    valid_indices: tuple[int, ...]
    invalid_indices: tuple[int, ...]

    @property
    def valid_count(self) -> int:
        return len(self.valid_indices)

    @property
    def invalid_count(self) -> int:
        return len(self.invalid_indices)


@dataclass(frozen=True)
class TrainingState:
    next_epoch: int
    best_val_wer: float
    best_epoch: int
    epochs_without_improvement: int
    history: tuple[dict[str, Any], ...]
    config: dict[str, Any]


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
    output_lengths: torch.Tensor | Sequence[int] | None = None,
) -> list[str]:
    """Upstream argmax + repeat/blank collapse for batched ``(B,T,C)`` logits."""
    token_ids = logits.argmax(dim=-1).detach().cpu()
    if output_lengths is None:
        lengths = [token_ids.shape[1]] * token_ids.shape[0]
    elif isinstance(output_lengths, torch.Tensor):
        lengths = output_lengths.detach().cpu().tolist()
    else:
        lengths = list(output_lengths)
    if len(lengths) != token_ids.shape[0]:
        raise ValueError("Broj output dužina mora odgovarati batch dimenziji")
    return [
        dataset_type.ctc_arr2txt(row[: min(int(length), token_ids.shape[1])], start=1)
        for row, length in zip(token_ids, lengths)
    ]


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


def set_backbone_trainable(model: nn.Module, trainable: bool) -> tuple[str, ...]:
    """Freeze/unfreeze every LipNet parameter except the Serbian FC head."""
    if not hasattr(model, "FC"):
        raise AttributeError("Model mora da ima LipNet FC head")
    trainable_names = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = trainable or name.startswith("FC.")
        if parameter.requires_grad:
            trainable_names.append(name)
    return tuple(trainable_names)


def build_finetune_optimizer(
    model: nn.Module,
    backbone_lr: float = 2e-5,
    head_lr: float = 1e-4,
) -> Adam:
    """Create stable parameter groups that remain identical across resume."""
    backbone = []
    head = []
    for name, parameter in model.named_parameters():
        (head if name.startswith("FC.") else backbone).append(parameter)
    if not backbone or not head:
        raise ValueError("Očekivani su i backbone i FC parametri")
    return Adam(
        [
            {"params": backbone, "lr": backbone_lr, "name": "backbone"},
            {"params": head, "lr": head_lr, "name": "head"},
        ]
    )


def scan_ctc_compatibility(dataset: Dataset) -> CTCFilterReport:
    """Find samples whose targets can align to their real video lengths."""
    valid = []
    invalid = []
    for index in range(len(dataset)):
        if hasattr(dataset, "ctc_lengths"):
            video_length, target = dataset.ctc_lengths(index)
        else:
            sample = dataset[index]
            target_length = int(sample["txt_len"])
            target = sample["txt"][:target_length]
            video_length = int(sample["vid_len"])
        if minimum_ctc_steps(target) <= video_length:
            valid.append(index)
        else:
            invalid.append(index)
    if not valid:
        raise ValueError("Nijedan uzorak ne zadovoljava CTC ograničenje")
    return CTCFilterReport(tuple(valid), tuple(invalid))


def run_epoch(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    optimizer: Optimizer | None = None,
    criterion: nn.CTCLoss | None = None,
    accumulation_steps: int = 1,
    grad_clip_norm: float | None = None,
) -> EpochResult:
    is_training = optimizer is not None
    model.train(is_training)
    criterion = criterion or nn.CTCLoss(blank=0, zero_infinity=False)

    total_loss = 0.0
    sample_count = 0
    predictions: list[str] = []
    references: list[str] = []

    if is_training:
        optimizer.zero_grad(set_to_none=True)

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for batch_index, batch in enumerate(loader):
            device_batch = {
                key: value.to(device) if key in {"vid", "txt"} else value
                for key, value in batch.items()
            }

            logits = model(device_batch["vid"])
            loss = ctc_loss(logits, device_batch, criterion)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"CTC loss nije konačan: {loss.item()}")

            if optimizer is not None:
                (loss / accumulation_steps).backward()

                should_step = (
                    (batch_index + 1) % accumulation_steps == 0
                    or batch_index + 1 == len(loader)
                )

                if should_step:
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            grad_clip_norm,
                        )

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            batch_size = int(device_batch["vid"].shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            sample_count += batch_size

            predictions.extend(
                greedy_decode(
                    logits,
                    output_lengths=batch["vid_len"],
                )
            )
            references.extend(reference_text(batch))

    if sample_count == 0:
        raise ValueError("DataLoader nije vratio nijedan batch")

    metrics = sequence_metrics(predictions, references)

    return EpochResult(
        loss=total_loss / sample_count,
        wer=metrics["wer"],
        cer=metrics["cer"],
        sentence_exact_match=metrics["sentence_exact_match"],
        samples=sample_count,
        predictions=tuple(predictions),
        references=tuple(references),
    )


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    *,
    epoch: int,
    best_val_wer: float,
    best_epoch: int,
    epochs_without_improvement: int,
    history: Sequence[Mapping[str, Any]],
    config: FineTuneConfig | Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save everything needed to resume after a Colab interruption."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    config_value = asdict(config) if isinstance(config, FineTuneConfig) else dict(config)
    payload = {
        "schema_version": 1,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_wer": float(best_val_wer),
        "best_epoch": int(best_epoch),
        "epochs_without_improvement": int(epochs_without_improvement),
        "history": [dict(item) for item in history],
        "config": config_value,
        "rng_state": capture_rng_state(),
        "metadata": dict(metadata or {}),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    *,
    restore_rng: bool = True,
) -> TrainingState:
    """Restore a Phase-5 checkpoint and return the next epoch to execute."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("schema_version") != 1:
        raise ValueError("Nepoznata Phase-5 checkpoint šema")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return TrainingState(
        next_epoch=int(payload["epoch"]) + 1,
        best_val_wer=float(payload["best_val_wer"]),
        best_epoch=int(payload["best_epoch"]),
        epochs_without_improvement=int(payload["epochs_without_improvement"]),
        history=tuple(dict(item) for item in payload["history"]),
        config=dict(payload["config"]),
    )


def validation_wer_improved(current: float, best: float) -> bool:
    """The baseline checkpoint criterion is strictly lower validation WER."""
    return bool(np.isfinite(current) and current < best)

def build_strong_finetune_optimizer(model: nn.Module) -> Adam:
    param_groups = {
        "conv_early": [],
        "conv3": [],
        "gru1": [],
        "gru2": [],
        "head": [],
    }

    for name, parameter in model.named_parameters():
        if name.startswith(("conv1.", "conv2.")):
            param_groups["conv_early"].append(parameter)
        elif name.startswith("conv3."):
            param_groups["conv3"].append(parameter)
        elif name.startswith("gru1."):
            param_groups["gru1"].append(parameter)
        elif name.startswith("gru2."):
            param_groups["gru2"].append(parameter)
        elif name.startswith("FC."):
            param_groups["head"].append(parameter)
        else:
            raise ValueError(f"Neočekivan parametar modela: {name}")

    return Adam(
        [
            {
                "params": param_groups["conv_early"],
                "lr": 2e-6,
                "initial_lr": 2e-6,
                "name": "conv_early",
            },
            {
                "params": param_groups["conv3"],
                "lr": 5e-6,
                "initial_lr": 5e-6,
                "name": "conv3",
            },
            {
                "params": param_groups["gru1"],
                "lr": 1e-5,
                "initial_lr": 1e-5,
                "name": "gru1",
            },
            {
                "params": param_groups["gru2"],
                "lr": 2e-5,
                "initial_lr": 2e-5,
                "name": "gru2",
            },
            {
                "params": param_groups["head"],
                "lr": 5e-5,
                "initial_lr": 5e-5,
                "name": "head",
            },
        ],
        weight_decay=0.0,
        amsgrad=True,
    )