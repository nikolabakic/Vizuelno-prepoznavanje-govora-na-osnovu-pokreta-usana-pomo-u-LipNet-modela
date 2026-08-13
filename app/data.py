from __future__ import annotations

import csv
import json
import random
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path


VOCAB_SYMBOLS = ["<blank>", " ", *list("abcčćdđefghijklmnoprsštuvzž")]


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    speaker_id: str
    video_path: str
    alignment_path: str
    transcript: str
    duration_s: float
    num_frames: int
    fps: float
    width: int
    height: int
    target_length: int
    min_ctc_frames: int


def parse_alignment(path: Path) -> str:
    tokens: list[str] = []
    previous_end = -1
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw_line.strip():
            continue
        parts = raw_line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"{path}:{line_number}: očekivane su 3 kolone")
        token = unicodedata.normalize("NFC", parts[2].strip().lower())
        start, end = int(parts[0]), int(parts[1])
        # SIL granice se ne koriste za CTC target. U lokalnom korpusu postoji
        # bar jedan pogrešan završni SIL (end < start), dok su govorni tokeni
        # ispravni; zato vremensku strogost primenjujemo samo na target tokene.
        if token == "sil":
            continue
        if start < 0 or end <= start or start < previous_end:
            raise ValueError(f"{path}:{line_number}: neispravan vremenski interval govornog tokena")
        previous_end = end
        tokens.append(token)
    if not tokens:
        raise ValueError(f"{path}: prazan transkript")
    return " ".join(tokens)


def min_ctc_frames(transcript: str) -> int:
    """Minimum CTC time steps, including blanks between repeated labels."""
    return len(transcript) + sum(a == b for a, b in zip(transcript, transcript[1:]))


def probe_video(path: Path) -> tuple[int, float, float, int, int]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Instaliraj projektne zavisnosti komandom `uv sync`.") from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Video ne može da se otvori: {path}")
    try:
        frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    if frames <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Neispravni video metapodaci: {path}")
    return frames, fps, frames / fps, width, height


def discover_pairs(corpus_root: Path) -> list[tuple[str, Path, Path]]:
    videos = {p.stem: p for p in corpus_root.glob("spk*/ser/video_a/*.mp4")}
    alignments = {p.stem: p for p in corpus_root.glob("spk*/alignment/*.align")}
    if not videos:
        raise ValueError(f"Nema MP4 fajlova ispod {corpus_root}")
    missing_alignments = sorted(videos.keys() - alignments.keys())
    missing_videos = sorted(alignments.keys() - videos.keys())
    if missing_alignments or missing_videos:
        raise ValueError(
            f"Neupareni fajlovi: bez ALIGN={missing_alignments[:5]}, bez MP4={missing_videos[:5]}"
        )
    return [(sample_id, videos[sample_id], alignments[sample_id]) for sample_id in sorted(videos)]


def build_manifest(corpus_root: Path) -> list[ManifestRow]:
    corpus_root = corpus_root.resolve()
    rows: list[ManifestRow] = []
    for sample_id, video, alignment in discover_pairs(corpus_root):
        speaker_id = sample_id.split("_", 1)[0]
        if video.parents[2].name != speaker_id or alignment.parent.parent.name != speaker_id:
            raise ValueError(f"Speaker ID i putanja se ne slažu za {sample_id}")
        transcript = parse_alignment(alignment)
        frames, fps, duration, width, height = probe_video(video)
        rows.append(
            ManifestRow(
                sample_id=sample_id,
                speaker_id=speaker_id,
                video_path=video.relative_to(corpus_root).as_posix(),
                alignment_path=alignment.relative_to(corpus_root).as_posix(),
                transcript=transcript,
                duration_s=round(duration, 6),
                num_frames=frames,
                fps=round(fps, 6),
                width=width,
                height=height,
                target_length=len(transcript),
                min_ctc_frames=min_ctc_frames(transcript),
            )
        )
    validate_rows(rows)
    return rows


def validate_rows(rows: list[ManifestRow]) -> None:
    allowed = set(VOCAB_SYMBOLS[1:])
    ids: set[str] = set()
    for row in rows:
        if row.sample_id in ids:
            raise ValueError(f"Duplikat u manifestu: {row.sample_id}")
        ids.add(row.sample_id)
        unknown = set(row.transcript) - allowed
        if unknown:
            raise ValueError(f"{row.sample_id}: karakteri van vokabulara: {sorted(unknown)}")
        if row.min_ctc_frames > row.num_frames:
            raise ValueError(
                f"{row.sample_id}: CTC target traži {row.min_ctc_frames}, video ima {row.num_frames} frejmova"
            )


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def make_split(rows: list[ManifestRow], seed: int = 42) -> dict:
    speakers = sorted({row.speaker_id for row in rows})
    if len(speakers) < 3:
        raise ValueError("Speaker-disjoint split zahteva najmanje 3 govornika")
    random.Random(seed).shuffle(speakers)
    test_count = max(1, round(len(speakers) * 0.15))
    validation_count = max(1, round(len(speakers) * 0.15))
    split_speakers = {
        "train": sorted(speakers[test_count + validation_count :]),
        "validation": sorted(speakers[test_count : test_count + validation_count]),
        "test": sorted(speakers[:test_count]),
    }
    result: dict = {"seed": seed, "speakers": split_speakers}
    for partition, members in split_speakers.items():
        member_set = set(members)
        result[partition] = [row.sample_id for row in rows if row.speaker_id in member_set]
    validate_split(result)
    return result


def validate_split(split: dict) -> None:
    speaker_sets = [set(split["speakers"][name]) for name in ("train", "validation", "test")]
    if any(a & b for index, a in enumerate(speaker_sets) for b in speaker_sets[index + 1 :]):
        raise ValueError("Splitovi dele govornike")
    sample_sets = [set(split[name]) for name in ("train", "validation", "test")]
    if any(a & b for index, a in enumerate(sample_sets) for b in sample_sets[index + 1 :]):
        raise ValueError("Splitovi dele primere")


def write_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def vocab_payload() -> dict:
    return {
        "blank_id": 0,
        "space_id": 1,
        "symbols": VOCAB_SYMBOLS,
        "symbol_to_id": {symbol: index for index, symbol in enumerate(VOCAB_SYMBOLS)},
    }
