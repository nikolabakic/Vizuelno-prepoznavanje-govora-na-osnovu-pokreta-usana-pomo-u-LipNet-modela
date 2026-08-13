"""Modern Colab-safe form of VIPL's face alignment and mouth-crop demo.

Geometry and crop coordinates come from VIPL ``demo.py`` at commit
``40209e09c49553c00c25c7d41faa3706aea3c625``.  Compatibility changes use
``subprocess`` argument lists, the current face-alignment enum, explicit device
selection and reusable functions.  See ``LICENSE.vipl`` and
``docs/upstream-diff.md``.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch


_CANONICAL_X = np.asarray(
    [
        0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856,
        0.689483, 0.799124, 0.904991, 0.98004, 0.490127, 0.490127,
        0.490127, 0.490127, 0.36688, 0.426036, 0.490127, 0.554217,
        0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
        0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751,
        0.719335, 0.254149, 0.340985, 0.428858, 0.490127, 0.551395,
        0.639268, 0.726104, 0.642159, 0.556721, 0.490127, 0.423532,
        0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
        0.553364, 0.490127, 0.42689,
    ],
    dtype=np.float64,
)
_CANONICAL_Y = np.asarray(
    [
        0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906,
        0.0344891, 0.0187482, 0.038915, 0.106454, 0.203352, 0.307009,
        0.409805, 0.515625, 0.587326, 0.609345, 0.628106, 0.609345,
        0.587326, 0.216423, 0.178758, 0.179852, 0.231733, 0.245099,
        0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077,
        0.245099, 0.780233, 0.745405, 0.727388, 0.742578, 0.727388,
        0.745405, 0.780233, 0.864805, 0.902192, 0.909281, 0.902192,
        0.864805, 0.784792, 0.778746, 0.785343, 0.778746, 0.784792,
        0.824182, 0.831803, 0.824182,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PreprocessResult:
    frames: np.ndarray
    decoded_frames: int
    landmark_frames: int

    @property
    def dropped_frames(self) -> int:
        return self.decoded_frames - self.landmark_frames


def get_position(size: int, padding: float = 0.25) -> np.ndarray:
    """VIPL's canonical 51-point frontal face template."""
    x = (_CANONICAL_X + padding) / (2 * padding + 1) * size
    y = (_CANONICAL_Y + padding) / (2 * padding + 1) * size
    return np.column_stack((x, y))


def transformation_from_points(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """VIPL similarity transform, expressed with ndarray matrix products."""
    source = np.asarray(points1, dtype=np.float64).copy()
    target = np.asarray(points2, dtype=np.float64).copy()
    center_source = source.mean(axis=0)
    center_target = target.mean(axis=0)
    source -= center_source
    target -= center_target
    scale_source = source.std()
    scale_target = target.std()
    if scale_source == 0 or scale_target == 0:
        raise ValueError("Degenerisani landmark-i ne mogu da definišu transformaciju")
    source /= scale_source
    target /= scale_target
    left, _, right_t = np.linalg.svd(source.T @ target)
    rotation = (left @ right_t).T
    affine = np.zeros((2, 3), dtype=np.float64)
    affine[:, :2] = (scale_target / scale_source) * rotation
    affine[:, 2] = center_target - affine[:, :2] @ center_source
    return affine


def make_face_aligner(device: str = "cuda", face_detector: str = "sfd"):
    """Construct current ``face_alignment`` API without importing it on CPU-only setup."""
    import face_alignment

    landmark_type = getattr(face_alignment.LandmarksType, "TWO_D", None)
    if landmark_type is None:  # old API used by the pinned upstream version
        landmark_type = face_alignment.LandmarksType._2D
    return face_alignment.FaceAlignment(
        landmark_type,
        flip_input=False,
        device=device,
        face_detector=face_detector,
    )


def decode_video_25fps(video_path: str | Path, frame_dir: str | Path) -> list[np.ndarray]:
    """Decode with FFmpeg exactly once and return numerically ordered BGR frames."""
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path),
        "-qscale:v", "2", "-r", "25", str(frame_dir / "%06d.jpg"),
    ]
    subprocess.run(command, check=True)
    paths = sorted(frame_dir.glob("*.jpg"), key=lambda path: int(path.stem))
    frames = [cv2.imread(str(path)) for path in paths]
    frames = [frame for frame in frames if frame is not None]
    if not frames:
        raise RuntimeError(f"FFmpeg nije dekodirao frejmove iz {video_path}")
    return frames


def crop_aligned_mouth(scene: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Apply VIPL alignment and its fixed 160x80 mouth crop, then resize to 128x64."""
    front256 = get_position(256)
    shape = np.asarray(landmarks, dtype=np.float64)[:68, :2][17:]
    affine = transformation_from_points(shape, front256)
    aligned = cv2.warpAffine(scene, affine, (256, 256))
    x, y = front256[-20:].mean(axis=0).astype(np.int32)
    half_width = 160 // 2
    crop = aligned[y - half_width // 2 : y + half_width // 2, x - half_width : x + half_width]
    if crop.shape[:2] != (80, 160):
        raise RuntimeError(f"VIPL mouth crop ima neočekivan oblik {crop.shape}")
    return cv2.resize(crop, (128, 64), interpolation=cv2.INTER_LANCZOS4)


def preprocess_frames(
    frames: Sequence[np.ndarray],
    aligner,
    progress_every: int = 0,
) -> PreprocessResult:
    """Run the upstream per-frame landmark/alignment/crop path.

    Like VIPL, frames without landmarks are omitted. Their count is returned so
    the caller can log and review borderline clips instead of hiding failures.
    """
    mouths: list[np.ndarray] = []
    for index, scene in enumerate(frames, start=1):
        landmarks = aligner.get_landmarks(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
        if landmarks is not None and len(landmarks):
            mouths.append(crop_aligned_mouth(scene, np.asarray(landmarks[0])))
        if progress_every and (index % progress_every == 0 or index == len(frames)):
            print(f"Landmarks: {index}/{len(frames)}", flush=True)
    if not mouths:
        raise RuntimeError("Lice nije pronađeno ni na jednom frejmu")
    return PreprocessResult(
        frames=np.stack(mouths).astype(np.uint8),
        decoded_frames=len(frames),
        landmark_frames=len(mouths),
    )


def preprocess_video(
    video_path: str | Path,
    aligner,
    progress_every: int = 0,
) -> PreprocessResult:
    with tempfile.TemporaryDirectory(prefix="lipnet_frames_") as directory:
        frames = decode_video_25fps(video_path, directory)
        return preprocess_frames(frames, aligner, progress_every=progress_every)


def write_mouth_jpegs(frames: np.ndarray, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.jpg"):
        stale.unlink()
    for index, frame in enumerate(frames, start=1):
        destination = output_dir / f"{index:06d}.jpg"
        if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"JPEG nije sačuvan: {destination}")


def mouth_tensor(frames: np.ndarray) -> torch.Tensor:
    """Return VIPL normalized ``(C,T,64,128)`` tensor."""
    if frames.ndim != 4 or frames.shape[1:] != (64, 128, 3):
        raise ValueError(f"Očekivano (T,64,128,3), dobijeno {frames.shape}")
    normalized = frames.astype(np.float32) / 255.0
    return torch.from_numpy(normalized.transpose(3, 0, 1, 2)).float()
