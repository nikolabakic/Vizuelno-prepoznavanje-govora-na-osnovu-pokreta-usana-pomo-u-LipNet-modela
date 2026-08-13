"""Lightweight compatibility tests for Phases 0-4 (no model forward/training)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from data.splits import SPLITS
from lipnet.dataset import (
    MyDataset,
    SERBIAN_LETTERS,
    SerbianDataset,
    parse_ai_speak_alignment,
    variable_length_collate,
)
from lipnet.model import LipNet
from lipnet.train import load_vipl_transfer


def test_speaker_splits_are_disjoint() -> None:
    names = tuple(SPLITS)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert not set(SPLITS[left]) & set(SPLITS[right])


def test_grid_ctc_collapse_matches_upstream_contract() -> None:
    # blank, A, A, blank, space, B, B -> "A B"
    assert MyDataset.ctc_arr2txt([0, 2, 2, 0, 1, 3, 3], start=1) == "A B"


def test_serbian_round_trip_and_class_count() -> None:
    text = "".join(SERBIAN_LETTERS).strip()
    encoded = SerbianDataset.txt2arr(text, start=1)
    assert SerbianDataset.arr2txt(encoded, start=1) == text
    assert 1 + len(SERBIAN_LETTERS) == 29


def test_local_alignment_removes_only_silence_tokens(tmp_path: Path) -> None:
    alignment = tmp_path / "sample.align"
    alignment.write_text(
        "0\t10\tsil\n10\t20\tPošalji\n20\t30\tsp\n30\t40\ta\n",
        encoding="utf-8",
    )
    assert parse_ai_speak_alignment(alignment) == "pošalji a"


def test_vipl_frame_loader_shape_and_values(tmp_path: Path) -> None:
    for index, value in enumerate((0, 255), start=1):
        frame = np.full((32, 64, 3), value, dtype=np.uint8)
        assert cv2.imwrite(str(tmp_path / f"{index:06d}.jpg"), frame)
    frames = MyDataset._load_vid(tmp_path)
    assert frames.shape == (2, 64, 128, 3)
    normalized = frames / 255.0
    assert 0.0 <= normalized.min() <= normalized.max() <= 1.0


def test_variable_length_batch_preserves_real_lengths() -> None:
    samples = [
        {
            "vid": torch.ones(3, 5, 64, 128),
            "txt": torch.tensor([1, 2, 3]),
            "vid_len": 5,
            "txt_len": 3,
        },
        {
            "vid": torch.ones(3, 7, 64, 128),
            "txt": torch.tensor([1, 2]),
            "vid_len": 7,
            "txt_len": 2,
        },
    ]
    batch = variable_length_collate(samples)
    assert set(batch) == {"vid", "txt", "vid_len", "txt_len"}
    assert batch["vid"].shape == (2, 3, 7, 64, 128)
    assert batch["vid_len"].tolist() == [5, 7]
    assert torch.count_nonzero(batch["vid"][0, :, 5:]) == 0


def test_transfer_skips_only_incompatible_fc(tmp_path: Path) -> None:
    checkpoint = tmp_path / "english.pt"
    torch.save(LipNet(num_classes=28).state_dict(), checkpoint)
    serbian_model = LipNet(num_classes=29)
    audit = load_vipl_transfer(serbian_model, checkpoint)
    assert set(audit.skipped_shape) == {"FC.weight", "FC.bias"}
    assert not audit.missing_in_checkpoint
    assert not audit.unexpected_in_checkpoint
