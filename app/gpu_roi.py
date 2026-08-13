from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from app.phase2 import read_manifest
from app.preprocessing import RoiRow, _median_box, _normalize_box, write_qa_sheet, write_rois
from app.data import write_json


MOUTH_68 = slice(48, 68)


def sampled_frames(video_path: Path, samples: int, max_side: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    positions = [round(count * (0.15 + 0.7 * i / max(samples - 1, 1))) for i in range(samples)]
    frames: list[np.ndarray] = []
    try:
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, min(position, count - 1))
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            scale = min(1.0, max_side / max(height, width))
            if scale < 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames


def detect_box(detector, frames: list[np.ndarray]) -> tuple[tuple[float, float, float, float] | None, int]:
    boxes: list[tuple[float, float, float, float]] = []
    for frame in frames:
        predictions = detector.get_landmarks_from_image(frame)
        if predictions is None or len(predictions) == 0:
            continue
        # Korpus ima jednu frontalnu osobu; prvi rezultat je ciljno lice.
        mouth = np.asarray(predictions[0])[MOUTH_68]
        height, width = frame.shape[:2]
        boxes.append(
            _normalize_box(
                (mouth[:, 0] / width).tolist(),
                (mouth[:, 1] / height).tolist(),
            )
        )
    return (_median_box(boxes), len(boxes)) if boxes else (None, 0)


def run(args: argparse.Namespace) -> None:
    import face_alignment
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA nije dostupna. U Colab-u izaberi Runtime -> Change runtime type -> T4 GPU.")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    manifest = read_manifest(args.input / "manifest.csv")
    detector = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device="cuda",
        face_detector=args.face_detector,
        flip_input=False,
        compile=False,
        max_batch_size=args.batch_size,
    )

    detections: dict[str, tuple[float, float, float, float] | None] = {}
    counts: dict[str, int] = {}
    by_speaker: dict[str, list[tuple[float, float, float, float]]] = {}
    corpus_root = args.corpus.resolve()
    for index, row in enumerate(manifest, 1):
        box, count = detect_box(
            detector,
            sampled_frames(corpus_root / row.video_path, args.samples, args.max_side),
        )
        detections[row.sample_id], counts[row.sample_id] = box, count
        if box is not None:
            by_speaker.setdefault(row.speaker_id, []).append(box)
        if index % 100 == 0 or index == len(manifest):
            print(f"GPU ROI: {index}/{len(manifest)}", flush=True)

    valid = [box for box in detections.values() if box is not None]
    if not valid:
        raise RuntimeError("Detektor nije pronašao lice ni u jednom klipu.")
    global_box = _median_box(valid)
    speaker_boxes = {speaker: _median_box(boxes) for speaker, boxes in by_speaker.items()}
    rois: list[RoiRow] = []
    for row in manifest:
        box = detections[row.sample_id]
        source = "face_alignment_cuda"
        if box is None:
            box = speaker_boxes.get(row.speaker_id, global_box)
            source = "speaker_median" if row.speaker_id in speaker_boxes else "global_median"
        rois.append(RoiRow(row.sample_id, *box, counts[row.sample_id], args.samples, source))

    write_rois(rois, args.input / "roi.csv")
    qa_ids = write_qa_sheet(manifest, rois, corpus_root, args.input / "roi_qa.jpg", args.qa_count)
    fallback = sum(row.source != "face_alignment_cuda" for row in rois)
    write_json(
        {
            "backend": "face-alignment",
            "device": "cuda",
            "face_detector": args.face_detector,
            "samples_per_video": args.samples,
            "fallback_count": fallback,
            "sample_ids": qa_ids,
            "manual_review_complete": False,
        },
        args.input / "roi_qa.json",
    )
    print(f"Završeno: {len(rois)} ROI-ja, fallback={fallback}", flush=True)
    print(f"Ručno pregledaj: {args.input / 'roi_qa.jpg'}", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Faza 2 mouth ROI preko face-alignment CUDA backend-a")
    result.add_argument("--corpus", type=Path, default=Path("processed/processed"))
    result.add_argument("--input", type=Path, default=Path("artifacts/phase2"))
    result.add_argument("--samples", type=int, default=5)
    result.add_argument("--qa-count", type=int, default=30)
    result.add_argument("--max-side", type=int, default=640, help="Detekciona rezolucija; ROI ostaje normalizovan")
    result.add_argument("--face-detector", choices=("sfd", "blazeface", "retinaface"), default="sfd")
    result.add_argument("--batch-size", type=int, default=8)
    return result


if __name__ == "__main__":
    run(parser().parse_args())

