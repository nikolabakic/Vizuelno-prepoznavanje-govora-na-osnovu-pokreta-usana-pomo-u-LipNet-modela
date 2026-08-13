from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from app.data import ManifestRow


LIP_LANDMARKS = (0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91,
                 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311,
                 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415)


@dataclass(frozen=True)
class RoiRow:
    sample_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    valid_detections: int
    sampled_frames: int
    source: str


def _median_box(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return tuple(median(values) for values in zip(*boxes))  # type: ignore[return-value]


def _normalize_box(xs: list[float], ys: list[float]) -> tuple[float, float, float, float]:
    center_x, center_y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    width = max(max(xs) - min(xs), 0.04) * 1.55
    height = max(max(ys) - min(ys), 0.025) * 2.0
    width = max(width, height * 2.0)
    height = width / 2.0
    x1, x2 = center_x - width / 2, center_x + width / 2
    y1, y2 = center_y - height / 2, center_y + height / 2
    shift_x = max(0.0, -x1) - max(0.0, x2 - 1.0)
    shift_y = max(0.0, -y1) - max(0.0, y2 - 1.0)
    return x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y


def detect_video_roi(
    video_path: Path, samples: int = 5, detector=None
) -> tuple[tuple[float, float, float, float] | None, int]:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("ROI zahteva instalirane `opencv-python-headless` i `mediapipe` pakete.") from exc

    owns_detector = detector is None
    if owns_detector:
        detector = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5
        )
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    positions = [round(frame_count * (0.15 + 0.7 * i / max(samples - 1, 1))) for i in range(samples)]
    boxes: list[tuple[float, float, float, float]] = []
    try:
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, min(position, frame_count - 1))
            ok, frame = capture.read()
            if not ok:
                continue
            result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not result.multi_face_landmarks:
                continue
            landmarks = result.multi_face_landmarks[0].landmark
            boxes.append(_normalize_box([landmarks[i].x for i in LIP_LANDMARKS], [landmarks[i].y for i in LIP_LANDMARKS]))
    finally:
        capture.release()
        if owns_detector:
            detector.close()
    return (_median_box(boxes), len(boxes)) if boxes else (None, 0)


def build_rois(rows: list[ManifestRow], corpus_root: Path, samples: int = 5) -> list[RoiRow]:
    import mediapipe as mp

    detections: dict[str, tuple[float, float, float, float] | None] = {}
    counts: dict[str, int] = {}
    by_speaker: dict[str, list[tuple[float, float, float, float]]] = {}
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5
    ) as detector:
        for index, row in enumerate(rows, 1):
            box, count = detect_video_roi(corpus_root / row.video_path, samples=samples, detector=detector)
            detections[row.sample_id], counts[row.sample_id] = box, count
            if box is not None:
                by_speaker.setdefault(row.speaker_id, []).append(box)
            if index % 100 == 0 or index == len(rows):
                print(f"ROI detekcija: {index}/{len(rows)}", flush=True)
    all_boxes = [box for box in detections.values() if box is not None]
    if not all_boxes:
        raise RuntimeError("Landmark detekcija nije uspela ni na jednom videu; ROI nije generisan.")
    global_box = _median_box(all_boxes)
    speaker_boxes = {speaker: _median_box(boxes) for speaker, boxes in by_speaker.items()}
    output: list[RoiRow] = []
    for row in rows:
        box = detections[row.sample_id]
        source = "detected"
        if box is None:
            box = speaker_boxes.get(row.speaker_id, global_box)
            source = "speaker_median" if row.speaker_id in speaker_boxes else "global_median"
        output.append(RoiRow(row.sample_id, *box, counts[row.sample_id], samples, source))
    return output


def write_rois(rows: list[RoiRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(RoiRow.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: getattr(row, field) for field in fields} for row in rows)


def write_qa_sheet(
    manifest: list[ManifestRow], rois: list[RoiRow], corpus_root: Path, output_path: Path, count: int = 30
) -> list[str]:
    import cv2
    import numpy as np

    roi_by_id = {row.sample_id: row for row in rois}
    chosen: list[ManifestRow] = []
    seen_speakers: set[str] = set()
    for row in manifest:
        if row.speaker_id not in seen_speakers:
            chosen.append(row)
            seen_speakers.add(row.speaker_id)
    for row in sorted(manifest, key=lambda item: item.num_frames, reverse=True):
        if len(chosen) >= max(count, len(seen_speakers)):
            break
        if row not in chosen:
            chosen.append(row)
    tiles = []
    for row in chosen:
        capture = cv2.VideoCapture(str(corpus_root / row.video_path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, row.num_frames // 2)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            continue
        roi = roi_by_id[row.sample_id]
        height, width = frame.shape[:2]
        x1, x2 = int(roi.x1 * width), int(roi.x2 * width)
        y1, y2 = int(roi.y1 * height), int(roi.y2 * height)
        crop = cv2.resize(frame[y1:y2, x1:x2], (256, 128))
        canvas = np.zeros((154, 256, 3), dtype=np.uint8)
        canvas[:128] = crop
        cv2.putText(canvas, f"{row.sample_id} [{roi.source}]", (5, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        tiles.append(canvas)
    columns = 5
    blank = np.zeros_like(tiles[0])
    while len(tiles) % columns:
        tiles.append(blank)
    sheet = np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"QA slika nije sačuvana: {output_path}")
    return [row.sample_id for row in chosen]
