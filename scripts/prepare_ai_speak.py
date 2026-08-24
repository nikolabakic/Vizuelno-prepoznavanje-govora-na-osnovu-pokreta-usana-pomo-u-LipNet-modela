#!/usr/bin/env python3
"""Convert AI-SPEAK MP4 files to VIPL-compatible mouth JPEG folders.

This is intentionally a thin corpus loop around ``lipnet.demo``. It writes no
training manifest and no static ROI. Run it on a Colab GPU; the script itself
never selects or installs a CUDA runtime.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from lipnet.demo import make_face_aligner, preprocess_video, write_mouth_jpegs


@dataclass(frozen=True)
class ClipResult:
    sample_id: str
    speaker: str
    video: str
    output: str
    decoded_frames: int
    landmark_frames: int
    dropped_frames: int


def _numeric_frames(folder: Path) -> list[Path]:
    frames = [path for path in folder.glob("*.jpg") if path.stem.isdigit()]
    return sorted(frames, key=lambda path: int(path.stem))


def restore_checkpoint_archives(checkpoint_dir: Path, output_root: Path) -> int:
    """Restore frame blocks and their real preprocessing metadata."""
    archives = sorted(checkpoint_dir.glob("chunk_*.zip"))
    if not archives:
        return 0

    restored: dict[tuple[str, str], ClipResult] = {}
    previous_log = output_root / "preprocessing.jsonl"
    if previous_log.exists():
        for line in previous_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                result = ClipResult(**payload)
                restored[(result.speaker, result.sample_id)] = result

    output_root_resolved = output_root.resolve()
    for index, checkpoint in enumerate(archives, start=1):
        with zipfile.ZipFile(checkpoint) as archive:
            try:
                result_lines = archive.read("_checkpoint/results.jsonl").decode("utf-8")
            except KeyError as exc:
                raise ValueError(f"Checkpoint nema results.jsonl: {checkpoint}") from exc
            for line in result_lines.splitlines():
                if line.strip():
                    payload = json.loads(line)
                    result = ClipResult(**payload)
                    restored[(result.speaker, result.sample_id)] = result
            for member in archive.infolist():
                if member.is_dir() or member.filename.startswith("_checkpoint/"):
                    continue
                destination = (output_root / member.filename).resolve()
                if not destination.is_relative_to(output_root_resolved):
                    raise ValueError(f"Nebezbedna putanja u {checkpoint}: {member.filename}")
                archive.extract(member, output_root)
        if index % 25 == 0 or index == len(archives):
            print(f"Vraćeni checkpoint-i: {index}/{len(archives)}", flush=True)

    previous_log.write_text(
        "".join(
            json.dumps(asdict(restored[key]), ensure_ascii=False) + "\n"
            for key in sorted(restored)
        ),
        encoding="utf-8",
    )
    return len(restored)


def write_checkpoint(
    block: list[ClipResult],
    failures: list[str],
    output_root: Path,
    checkpoint_dir: Path,
    start_index: int,
    end_index: int,
) -> Path:
    """Persist one bounded block as a single Drive-friendly ZIP file."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_dir / f"chunk_{start_index:06d}_{end_index:06d}.zip"
    temporary = checkpoint_dir / f".{destination.name}.tmp"
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for result in block:
            sample_dir = output_root / result.speaker / "video" / "video_a" / result.sample_id
            for frame in sorted(sample_dir.glob("*.jpg"), key=lambda path: int(path.stem)):
                archive.write(frame, frame.relative_to(output_root))
        archive.writestr(
            "_checkpoint/results.jsonl",
            "".join(json.dumps(asdict(item), ensure_ascii=False) + "\n" for item in block),
        )
        archive.writestr(
            "_checkpoint/failures.log",
            "\n".join(failures) + ("\n" if failures else ""),
        )
    temporary.replace(destination)
    print(f"Drive checkpoint: {destination.name}", flush=True)
    return destination


def discover_pairs(corpus_root: Path) -> list[tuple[str, str, Path, Path]]:
    def keyed(paths: list[Path], speaker_parent: int) -> dict[tuple[str, str], Path]:
        values: dict[tuple[str, str], Path] = {}
        for path in paths:
            key = (path.parents[speaker_parent].name, path.stem)
            if key in values:
                raise ValueError(f"Dupliran uzorak {key}: {values[key]} i {path}")
            values[key] = path
        return values

    videos = keyed(list(corpus_root.glob("spk*/ser/video_a/*.mp4")), 2)
    annotations = keyed(list(corpus_root.glob("spk*/alignment/*.align")), 1)
    if not videos:
        raise ValueError(f"Nema spk*/ser/video_a/*.mp4 ispod {corpus_root}")
    missing_annotations = sorted(videos.keys() - annotations.keys())
    missing_videos = sorted(annotations.keys() - videos.keys())
    if missing_annotations or missing_videos:
        raise ValueError(
            f"Neupareni ulazi: bez ALIGN={missing_annotations[:5]}, bez MP4={missing_videos[:5]}"
        )
    pairs = []
    for speaker, sample_id in sorted(videos):
        key = (speaker, sample_id)
        pairs.append((sample_id, speaker, videos[key], annotations[key]))
    return pairs


def _qa_sheet(results: list[ClipResult], output_root: Path, destination: Path) -> None:
    """Show a normal and highest-drop clip per speaker, three frames per clip."""
    by_speaker: dict[str, list[ClipResult]] = defaultdict(list)
    for result in results:
        by_speaker[result.speaker].append(result)
    selected: list[ClipResult] = []
    for speaker in sorted(by_speaker):
        clips = by_speaker[speaker]
        selected.append(clips[0])
        borderline = max(clips, key=lambda item: (item.dropped_frames, item.sample_id))
        if borderline.sample_id != clips[0].sample_id:
            selected.append(borderline)
    rows: list[np.ndarray] = []
    for result in selected:
        folder = output_root / result.speaker / "video" / "video_a" / result.sample_id
        files = sorted(folder.glob("*.jpg"), key=lambda path: int(path.stem))
        indices = np.linspace(0, len(files) - 1, 3, dtype=int)
        tiles = [cv2.imread(str(files[index])) for index in indices]
        strip = np.hstack(tiles)
        canvas = np.zeros((92, strip.shape[1], 3), dtype=np.uint8)
        canvas[:64] = strip
        label = f"{result.sample_id} dropped={result.dropped_frames}/{result.decoded_frames}"
        cv2.putText(canvas, label, (4, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        rows.append(canvas)
    if not rows:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), np.vstack(rows)):
        raise RuntimeError(f"QA slika nije sačuvana: {destination}")


def run(args: argparse.Namespace) -> None:
    corpus_root = args.corpus.resolve()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    pairs = discover_pairs(corpus_root)
    if args.speakers:
        allowed = set(args.speakers)
        pairs = [pair for pair in pairs if pair[1] in allowed]
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise ValueError("Filteri nisu ostavili nijedan MP4/ALIGN par za obradu")
    print(f"Pronađeno {len(pairs)} uparenih klipova", flush=True)

    if args.resume and args.checkpoint_dir:
        restored = restore_checkpoint_archives(args.checkpoint_dir.resolve(), output_root)
        if restored:
            print(f"Vraćeno rezultata sa Drive checkpoint-a: {restored}", flush=True)

    previous_results: dict[tuple[str, str], ClipResult] = {}
    previous_log = output_root / "preprocessing.jsonl"
    if args.resume and previous_log.exists():
        for line in previous_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                result = ClipResult(**payload)
                previous_results[(result.speaker, result.sample_id)] = result

    aligner = make_face_aligner(device=args.device, face_detector=args.face_detector)
    results: list[ClipResult] = []
    checkpoint_results: list[ClipResult] = []
    failures: list[str] = []
    failure_keys: list[tuple[str, str]] = []
    checkpoint_failures: list[str] = []
    checkpoint_start = 1
    for index, (sample_id, speaker, video_path, _) in enumerate(pairs, start=1):
        destination = output_root / speaker / "video" / "video_a" / sample_id
        try:
            previous = previous_results.get((speaker, sample_id))
            frame_count = len(_numeric_frames(destination))
            if (
                args.resume
                and previous is not None
                and frame_count > 0
                and frame_count == previous.landmark_frames
                and previous.decoded_frames >= previous.landmark_frames
            ):
                result = previous
            else:
                processed = preprocess_video(video_path, aligner)
                write_mouth_jpegs(processed.frames, destination)
                result = ClipResult(
                    sample_id=sample_id,
                    speaker=speaker,
                    video=str(video_path),
                    output=str(destination),
                    decoded_frames=processed.decoded_frames,
                    landmark_frames=processed.landmark_frames,
                    dropped_frames=processed.dropped_frames,
                )
            results.append(result)
            checkpoint_results.append(result)
        except Exception as exc:  # keep the corpus job alive and make failures explicit
            message = f"{sample_id}\t{type(exc).__name__}: {exc}"
            failures.append(message)
            failure_keys.append((speaker, sample_id))
            checkpoint_failures.append(message)
            print(f"NEUSPEH {message}", flush=True)
        if index % args.report_every == 0 or index == len(pairs):
            print(f"Preprocessing {index}/{len(pairs)} | uspešno={len(results)} | neuspešno={len(failures)}", flush=True)
        if args.checkpoint_dir and (
            index % args.checkpoint_every == 0 or index == len(pairs)
        ):
            write_checkpoint(
                checkpoint_results,
                checkpoint_failures,
                output_root,
                args.checkpoint_dir,
                checkpoint_start,
                index,
            )
            checkpoint_results = []
            checkpoint_failures = []
            checkpoint_start = index + 1

    result_ids = [(result.speaker, result.sample_id) for result in results]
    failure_ids = failure_keys
    if len(result_ids) != len(set(result_ids)) or len(failure_ids) != len(set(failure_ids)):
        raise RuntimeError("Duplirani uzorci u završnom preprocessing auditu")
    expected_ids = {(speaker, sample_id) for sample_id, speaker, _, _ in pairs}
    if set(result_ids) | set(failure_ids) != expected_ids or set(result_ids) & set(failure_ids):
        raise RuntimeError("Završni preprocessing audit ne pokriva tačno sve ulazne klipove")

    # JSONL is a runtime/audit log only. Dataset discovery never reads it.
    log_path = output_root / "preprocessing.jsonl"
    log_path.write_text(
        "".join(json.dumps(asdict(result), ensure_ascii=False) + "\n" for result in results),
        encoding="utf-8",
    )
    (output_root / "failed_clips.log").write_text(
        "\n".join(failures) + ("\n" if failures else ""), encoding="utf-8"
    )
    _qa_sheet(results, output_root, output_root / "qa_mouth_crops.jpg")
    print(f"Frejmovi: {output_root}")
    print(f"QA: {output_root / 'qa_mouth_crops.jpg'}")
    print(f"Neuspeli klipovi: {output_root / 'failed_clips.log'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--face-detector", default="sfd", choices=("sfd", "blazeface", "retinaface"))
    parser.add_argument("--speakers", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
