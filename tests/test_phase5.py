"""CPU-only tests for the Phase-5 fine-tuning helpers."""

from __future__ import annotations

from pathlib import Path

import nbformat
import torch
import torch.nn as nn

from scripts import build_phase_notebooks
from lipnet.train import (
    FineTuneConfig,
    build_finetune_optimizer,
    greedy_decode,
    load_training_checkpoint,
    run_epoch,
    save_training_checkpoint,
    scan_ctc_compatibility,
    sequence_metrics,
    set_backbone_trainable,
    validation_wer_improved,
)


class ToyCTCModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Conv3d(1, 2, kernel_size=1)
        self.FC = nn.Linear(2, 29)

    def forward(
        self,
        video: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.backbone(video).mean(dim=(-1, -2)).transpose(1, 2)
        return self.FC(features)


def toy_batch() -> dict[str, torch.Tensor]:
    return {
        "vid": torch.randn(2, 1, 4, 1, 1),
        "txt": torch.tensor([[2, 3], [2, 0]], dtype=torch.long),
        "vid_len": torch.tensor([4, 3], dtype=torch.long),
        "txt_len": torch.tensor([2, 1], dtype=torch.long),
    }


def test_greedy_decode_ignores_padded_timesteps() -> None:
    logits = torch.full((1, 4, 29), -10.0)
    logits[0, 0, 2] = 10.0  # a
    logits[0, 1, 0] = 10.0  # blank
    logits[0, 2:, 3] = 10.0  # padded b tokens must be ignored
    assert greedy_decode(logits, output_lengths=torch.tensor([2])) == ["a"]
    assert greedy_decode(logits) == ["ab"]


def test_freeze_then_unfreeze_keeps_fc_trainable() -> None:
    model = ToyCTCModel()
    frozen_names = set_backbone_trainable(model, False)
    assert set(frozen_names) == {"FC.weight", "FC.bias"}
    assert all(parameter.requires_grad for parameter in model.FC.parameters())
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())

    unfrozen_names = set_backbone_trainable(model, True)
    assert set(unfrozen_names) == {name for name, _ in model.named_parameters()}
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_toy_train_and_evaluate_epoch_are_finite() -> None:
    torch.manual_seed(0)
    model = ToyCTCModel()
    optimizer = build_finetune_optimizer(model)
    train_result = run_epoch(model, [toy_batch()], torch.device("cpu"), optimizer)
    eval_result = run_epoch(model, [toy_batch()], torch.device("cpu"))
    for result in (train_result, eval_result):
        assert torch.isfinite(torch.tensor(result.loss))
        assert result.samples == 2
        assert len(result.predictions) == len(result.references) == 2
        assert set(result.metrics()) == {"loss", "wer", "cer", "sentence_exact_match"}


def test_checkpoint_round_trip_restores_training_and_rng_state(tmp_path: Path) -> None:
    torch.manual_seed(123)
    model = ToyCTCModel()
    optimizer = build_finetune_optimizer(model)
    run_epoch(model, [toy_batch()], torch.device("cpu"), optimizer)
    saved_weight = model.FC.weight.detach().clone()
    checkpoint = tmp_path / "latest.pt"
    history = [{"epoch": 0, "validation": {"wer": 0.75}}]

    save_training_checkpoint(
        checkpoint,
        model,
        optimizer,
        epoch=0,
        best_val_wer=0.75,
        best_epoch=0,
        epochs_without_improvement=0,
        history=history,
        config=FineTuneConfig(),
    )
    expected_random = torch.rand(3)
    with torch.no_grad():
        model.FC.weight.zero_()
    torch.rand(20)

    state = load_training_checkpoint(checkpoint, model, optimizer)
    actual_random = torch.rand(3)
    assert torch.equal(model.FC.weight, saved_weight)
    assert torch.equal(actual_random, expected_random)
    assert state.next_epoch == 1
    assert state.best_epoch == 0
    assert state.best_val_wer == 0.75
    assert list(state.history) == history
    assert state.config["max_epochs"] == 30


def test_best_checkpoint_criterion_is_strict_validation_wer() -> None:
    assert validation_wer_improved(0.4, 0.5)
    assert not validation_wer_improved(0.5, 0.5)
    assert not validation_wer_improved(0.6, 0.5)


def test_sequence_metrics_use_corpus_denominators() -> None:
    metrics = sequence_metrics(
        predictions=["a", "a b c"],
        references=["a b", "a b c"],
    )
    assert metrics["wer"] == 1 / 5
    assert metrics["cer"] == 2 / 8
    assert metrics["sentence_exact_match"] == 0.5


def test_ctc_scan_reports_invalid_samples_without_manifest() -> None:
    class LengthOnlyDataset:
        def __len__(self) -> int:
            return 2

        def ctc_lengths(self, index: int) -> tuple[int, torch.Tensor]:
            # Repeated labels require an intervening blank: [a, a] needs 3 steps.
            return ((3, torch.tensor([2, 2])) if index == 0 else (2, torch.tensor([2, 2])))

    report = scan_ctc_compatibility(LengthOnlyDataset())
    assert report.valid_indices == (0,)
    assert report.invalid_indices == (1,)


def test_generated_phase_notebooks_are_valid_and_clean() -> None:
    root = Path(__file__).parents[1]
    builders = (
        build_phase_notebooks.phase0,
        build_phase_notebooks.phase1,
        build_phase_notebooks.phase2,
        build_phase_notebooks.phase3,
        build_phase_notebooks.phase4,
        build_phase_notebooks.phase5,
        build_phase_notebooks.phase6,
    )
    for phase in range(7):
        path = next((root / "playground").glob(f"{phase:02d}_*.ipynb"))
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert notebook == builders[phase](), f"Regeneriši {path.name} iz generatora"
        assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                assert cell.execution_count is None
                assert not cell.outputs
                compile(cell.source, f"{path.name}:{cell.id}", "exec")

        if phase == 4:
            code_source = "\n".join(
                cell.source for cell in notebook.cells if cell.cell_type == "code"
            )
            assert (
                "set(audit.skipped_shape) == {'FC.weight', 'FC.bias'}"
                in code_source
            )

    phase5 = nbformat.read(
        root / "playground/05_faza_5_baseline_finetuning.ipynb", as_version=4
    )
    headings = "\n".join(
        cell.source for cell in phase5.cells if cell.cell_type == "markdown"
    )
    for heading in ("## Goal", "## Setup", "## Steps", "## Checks", "## Next Steps"):
        assert heading in headings
