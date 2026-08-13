"""Small video transforms adapted from VIPL ``cvtransforms.py``.

Source commit: 40209e09c49553c00c25c7d41faa3706aea3c625.
See ``LICENSE.vipl``. Arrays use VIPL's ``(T,H,W,C)`` BGR layout.
"""

from __future__ import annotations

import random

import numpy as np


def HorizontalFlip(batch_img: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Preserve the upstream probability convention and horizontal axis."""
    if random.random() > p:
        batch_img = batch_img[:, :, ::-1, ...]
    return batch_img


def ColorNormalize(batch_img: np.ndarray) -> np.ndarray:
    """VIPL baseline normalization: uint8-like values divided by 255."""
    return batch_img / 255.0
