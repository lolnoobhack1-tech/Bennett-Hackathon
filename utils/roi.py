"""Shared ROI utilities for efficient ANPR cropping."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def extract_bottom_roi(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    ratio: float = 0.4,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Extract bottom `ratio` region from a bounding box.

    Returns:
        Cropped ROI image and ROI bbox in global image coordinates.
    """
    if image is None or image.size == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8), (0, 0, 0, 0)

    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=np.uint8), (0, 0, 0, 0)

    box_height = y2 - y1
    bottom_height = max(1, int(box_height * float(ratio)))
    roi_y1 = max(y1, y2 - bottom_height)
    roi = image[roi_y1:y2, x1:x2]
    return roi, (x1, roi_y1, x2, y2)

