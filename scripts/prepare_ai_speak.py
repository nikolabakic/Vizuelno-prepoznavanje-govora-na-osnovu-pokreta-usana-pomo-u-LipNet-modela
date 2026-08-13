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
    videos = {path.stem: path for path in corpus_root.glob("spk*/ser/video_a/*.mp4")}
    annotations = {path.stem: path for path in corpus_root.glob("spk*/alignment/*.align")}
    if not videos:
        raise ValueError(f"Nema spk*/ser/video_a/*.mp4 ispod {corpus_root}")
    missing_annotations = sorted(videos.keys() - annotations.keys())
    missing_videos = sorted(annotations.keys() - videos.keys())
    if missing_annotations or missing_videos:
        raise ValueError(
            f"Neupareni ulazi: bez ALIGN={missing_annotations[:5]}, bez MP4={missing_videos[:5]}"
        )
    pairs = []
    for sample_id in sorted(videos):
        speaker = videos[sample_id].parents[2].name
        if annotations[sample_id].parent.parent.name != speaker:
            raise ValueError(f"Speaker folderi se ne slažu za {sample_id}")
        pairs.append((sample_id, speaker, videos[sample_id], annotations[sample_id]))
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
    print(f"Pronađeno {len(pairs)} uparenih klipova", flush=True)

    previous_results: dict[str, ClipResult] = {}
    previous_log = output_root / "preprocessing.jsonl"
    if args.resume and previous_log.exists():
        for line in previous_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                previous_results[payload["sample_id"]] = ClipResult(**payload)

    aligner = make_face_aligner(device=args.device, face_detector=args.face_detector)
    results: list[ClipResult] = []
    checkpoint_results: list[ClipResult] = []
    failures: list[str] = []
    checkpoint_start = 1
    for index, (sample_id, speaker, video_path, _) in enumerate(pairs, start=1):
        destination = output_root / speaker / "video" / "video_a" / sample_id
        try:
            if args.resume and list(destination.glob("*.jpg")):
                result = previous_results.get(sample_id)
                if result is None:
                    frame_count = len(list(destination.glob("*.jpg")))
                    result = ClipResult(
                        sample_id, speaker, str(video_path), str(destination), frame_count,
                        frame_count, 0,
                    )
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
            print(f"NEUSPEH {message}", flush=True)
        if index % args.report_every == 0 or index == len(pairs):
            print(f"Preprocessing {index}/{len(pairs)} | uspešno={len(results)} | neuspešno={len(failures)}", flush=True)
        if args.checkpoint_dir and (
            index % args.checkpoint_every == 0 or index == len(pairs)
        ):
            write_checkpoint(
                checkpoint_results,
                failures,
                output_root,
                args.checkpoint_dir,
                checkpoint_start,
                index,
            )
            checkpoint_results = []
            checkpoint_start = index + 1

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
