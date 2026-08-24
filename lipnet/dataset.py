"""VIPL-compatible GRID dataset plus the minimal AI-SPEAK adapter.

Adapted from VIPL ``dataset.py`` at commit
``40209e09c49553c00c25c7d41faa3706aea3c625``.  Frame loading,
normalization, text conversion, greedy CTC collapse, and per-sentence WER/CER
helpers keep the upstream behavior. Phase-5 reporting aggregates edit counts at
corpus level in ``lipnet.train.sequence_metrics``. Local changes are documented
in ``docs/upstream-diff.md``.
See ``LICENSE.vipl``.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import editdistance
import numpy as np
import torch
from torch.utils.data import Dataset

from .cvtransforms import ColorNormalize, HorizontalFlip


ENGLISH_LETTERS = tuple(" ABCDEFGHIJKLMNOPQRSTUVWXYZ")
# Character inventory observed in the supplied corpus. Serbian digraphs remain
# two Unicode characters, which is the intended character-level CTC contract.
SERBIAN_LETTERS = tuple(" abcčćdđefghijklmnoprsštuvzž")
BLANK_ID = 0


def _numeric_jpegs(folder: Path) -> list[Path]:
    files = [path for path in folder.iterdir() if path.suffix.lower() in {".jpg", ".jpeg"}]
    try:
        return sorted(files, key=lambda path: int(path.stem))
    except ValueError as exc:
        raise ValueError(f"JPEG nazivi moraju biti numerički u {folder}") from exc


class MyDataset(Dataset):
    """Original GRID Dataset contract with current pathlib/import conventions."""

    letters = ENGLISH_LETTERS

    def __init__(
        self,
        video_path: str | Path,
        anno_path: str | Path,
        file_list: str | Path,
        vid_pad: int,
        txt_pad: int,
        phase: str,
    ) -> None:
        self.anno_path = Path(anno_path)
        self.vid_pad = vid_pad
        self.txt_pad = txt_pad
        self.phase = phase
        root = Path(video_path)
        lines = Path(file_list).read_text(encoding="utf-8").splitlines()
        self.videos = [root / line.strip() for line in lines if line.strip()]
        self.data: list[tuple[Path, str, str]] = []
        for video in self.videos:
            items = video.parts
            if len(items) < 4:
                raise ValueError(f"GRID putanja nema očekivanu dubinu: {video}")
            self.data.append((video, items[-4], items[-1]))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        video_path, speaker, name = self.data[idx]
        video = self._load_vid(video_path)
        annotation = self._load_anno(self.anno_path / speaker / "align" / f"{name}.align")
        if self.phase == "train":
            video = HorizontalFlip(video)
        video = ColorNormalize(video)
        video_len = video.shape[0]
        annotation_len = annotation.shape[0]
        video = self._padding(video, self.vid_pad)
        annotation = self._padding(annotation, self.txt_pad)
        return {
            "vid": torch.from_numpy(np.ascontiguousarray(video.transpose(3, 0, 1, 2))).float(),
            "txt": torch.from_numpy(annotation).long(),
            "txt_len": annotation_len,
            "vid_len": video_len,
        }

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def _load_vid(path: str | Path) -> np.ndarray:
        folder = Path(path)
        files = _numeric_jpegs(folder)
        images = [cv2.imread(str(file)) for file in files]
        images = [image for image in images if image is not None]
        if not images:
            raise ValueError(f"Nema čitljivih JPEG frejmova u {folder}")
        images = [
            cv2.resize(image, (128, 64), interpolation=cv2.INTER_LANCZOS4)
            for image in images
        ]
        return np.stack(images, axis=0).astype(np.float32)

    def _load_anno(self, name: str | Path) -> np.ndarray:
        lines = [line.strip().split(" ") for line in Path(name).read_text().splitlines()]
        tokens = [line[2] for line in lines if len(line) >= 3]
        tokens = [token for token in tokens if token.upper() not in {"SIL", "SP"}]
        return self.txt2arr(" ".join(tokens).upper(), start=1)

    @staticmethod
    def _padding(array: np.ndarray, length: int) -> np.ndarray:
        if array.shape[0] > length:
            raise ValueError(f"Sekvenca dužine {array.shape[0]} ne staje u padding {length}")
        items = [array[index] for index in range(array.shape[0])]
        size = items[0].shape
        items.extend(np.zeros(size, dtype=array.dtype) for _ in range(length - len(items)))
        return np.stack(items, axis=0)

    @classmethod
    def txt2arr(cls, txt: str, start: int = 1) -> np.ndarray:
        try:
            return np.asarray([cls.letters.index(char) + start for char in txt], dtype=np.int64)
        except ValueError as exc:
            unknown = sorted(set(txt) - set(cls.letters))
            raise ValueError(f"Karakteri van vokabulara: {unknown}") from exc

    @classmethod
    def arr2txt(cls, arr: Iterable[int], start: int = 1) -> str:
        text = []
        for value in arr:
            token = int(value)
            if token >= start:
                text.append(cls.letters[token - start])
        return "".join(text).strip()

    @classmethod
    def ctc_arr2txt(cls, arr: Iterable[int], start: int = 1) -> str:
        previous = -1
        text = []
        for value in arr:
            token = int(value)
            if previous != token and token >= start:
                character = cls.letters[token - start]
                if not (text and text[-1] == " " and character == " "):
                    text.append(character)
            previous = token
        return "".join(text).strip()

    @staticmethod
    def wer(predict: Sequence[str], truth: Sequence[str]) -> list[float]:
        pairs = [(pred.split(" "), target.split(" ")) for pred, target in zip(predict, truth)]
        return [editdistance.eval(pred, target) / max(len(target), 1) for pred, target in pairs]

    @staticmethod
    def cer(predict: Sequence[str], truth: Sequence[str]) -> list[float]:
        return [
            editdistance.eval(pred, target) / max(len(target), 1)
            for pred, target in zip(predict, truth)
        ]


def normalize_serbian_text(tokens: Iterable[str]) -> str:
    normalized = []
    for raw_token in tokens:
        token = unicodedata.normalize("NFC", raw_token.strip().lower())
        if not token or token in {"sil", "sp"}:
            continue
        normalized.append(token)
    return " ".join(normalized)


def parse_ai_speak_alignment(path: str | Path) -> str:
    """Read the local tab-separated ``start, end, token`` format."""
    tokens: list[str] = []
    previous_speech_end = -1
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        columns = raw_line.split("\t")
        if len(columns) != 3:
            raise ValueError(f"{path}:{line_number}: očekivane su 3 tab-separated kolone")
        # Timing is not an input to CTC, but malformed speech rows should not be
        # silently accepted. SIL/SP timing is ignored, matching the target rule.
        token = unicodedata.normalize("NFC", columns[2].strip().lower())
        if token not in {"sil", "sp"}:
            start, end = int(columns[0]), int(columns[1])
            if start < 0 or end <= start or start < previous_speech_end:
                raise ValueError(f"{path}:{line_number}: neispravan vremenski interval")
            previous_speech_end = end
        tokens.append(token)
    text = normalize_serbian_text(tokens)
    if not text:
        raise ValueError(f"Prazan transkript: {path}")
    unknown = sorted(set(text) - set(SERBIAN_LETTERS))
    if unknown:
        raise ValueError(f"{path}: karakteri van srpskog vokabulara: {unknown}")
    return text


class SerbianDataset(MyDataset):
    """AI-SPEAK adapter that discovers VIPL-compatible JPEG folders at runtime."""

    letters = SERBIAN_LETTERS

    def __init__(
        self,
        video_path: str | Path,
        anno_path: str | Path,
        speakers: Sequence[str],
        phase: str,
    ) -> None:
        self.video_root = Path(video_path)
        self.anno_path = Path(anno_path)
        self.phase = phase
        self.vid_pad = 0
        self.txt_pad = 0
        self.videos = []
        self.data: list[tuple[Path, str, str]] = []
        for speaker in speakers:
            sample_dirs = sorted((self.video_root / speaker / "video" / "video_a").glob("*"))
            for sample_dir in sample_dirs:
                if sample_dir.is_dir() and _numeric_jpegs(sample_dir):
                    self.data.append((sample_dir, speaker, sample_dir.name))
        if not self.data:
            raise ValueError(
                f"Nema mouth JPEG foldera za {list(speakers)} ispod {self.video_root}. "
                "Prvo pokreni Fazu 2."
            )

    def _alignment_path(self, speaker: str, name: str) -> Path:
        candidates = (
            self.anno_path / speaker / "alignment" / f"{name}.align",
            self.anno_path / speaker / "align" / f"{name}.align",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Anotacija nije pronađena: {candidates}")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        video_path, speaker, name = self.data[idx]
        video = self._load_vid(video_path)
        text = parse_ai_speak_alignment(self._alignment_path(speaker, name))
        annotation = self.txt2arr(text, start=1)
        if self.phase == "train":
            video = HorizontalFlip(video)
        video = ColorNormalize(video)
        return {
            "vid": torch.from_numpy(np.ascontiguousarray(video.transpose(3, 0, 1, 2))).float(),
            "txt": torch.from_numpy(annotation).long(),
            "txt_len": int(annotation.shape[0]),
            "vid_len": int(video.shape[0]),
        }

    def ctc_lengths(self, idx: int) -> tuple[int, torch.Tensor]:
        """Read only frame names and transcript for a fast pre-training CTC audit."""
        video_path, speaker, name = self.data[idx]
        video_length = len(_numeric_jpegs(video_path))
        text = parse_ai_speak_alignment(self._alignment_path(speaker, name))
        return video_length, torch.from_numpy(self.txt2arr(text, start=1)).long()


def variable_length_collate(
    samples: Sequence[dict[str, torch.Tensor | int]],
) -> dict[str, torch.Tensor]:
    """Pad variable videos/targets while preserving the four VIPL keys."""
    if not samples:
        raise ValueError("Prazan batch")
    video_lengths = torch.tensor([int(sample["vid_len"]) for sample in samples], dtype=torch.long)
    text_lengths = torch.tensor([int(sample["txt_len"]) for sample in samples], dtype=torch.long)
    max_video = int(video_lengths.max())
    max_text = int(text_lengths.max())
    first_video = samples[0]["vid"]
    if not isinstance(first_video, torch.Tensor):
        raise TypeError("vid mora biti torch.Tensor")
    batch_video = torch.zeros(
        (len(samples), first_video.shape[0], max_video, first_video.shape[2], first_video.shape[3]),
        dtype=first_video.dtype,
    )
    batch_text = torch.zeros((len(samples), max_text), dtype=torch.long)
    for index, sample in enumerate(samples):
        video = sample["vid"]
        text = sample["txt"]
        if not isinstance(video, torch.Tensor) or not isinstance(text, torch.Tensor):
            raise TypeError("vid i txt moraju biti torch.Tensor")
        batch_video[index, :, : video.shape[1]] = video
        batch_text[index, : text.shape[0]] = text
    return {"vid": batch_video, "txt": batch_text, "txt_len": text_lengths, "vid_len": video_lengths}


def minimum_ctc_steps(target: torch.Tensor) -> int:
    values = target.tolist()
    return len(values) + sum(left == right for left, right in zip(values, values[1:]))


def validate_ctc_batch(batch: dict[str, torch.Tensor], output_lengths: torch.Tensor) -> None:
    """Fail before CTCLoss if a target cannot align to the model time axis."""
    for index, (target_length, output_length) in enumerate(
        zip(batch["txt_len"].tolist(), output_lengths.tolist())
    ):
        target = batch["txt"][index, :target_length]
        required = minimum_ctc_steps(target)
        if required > output_length:
            raise ValueError(
                f"Uzorak {index}: CTC target traži najmanje {required} koraka, "
                f"model daje {output_length}"
            )
