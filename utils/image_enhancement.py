"""ROI-only adaptive image enhancement utilities for ANPR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class EnhancementStats:
    """Container for enhancement diagnostics."""

    mean_intensity: float
    laplacian_variance: float
    clahe_applied: bool
    glare_reduction_applied: bool


def compute_roi_quality(roi_bgr: np.ndarray) -> Tuple[float, float]:
    """Return mean brightness and blur score (Laplacian variance) for an ROI."""
    if roi_bgr is None or roi_bgr.size == 0:
        return 0.0, 0.0

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    mean_intensity = float(np.mean(gray))
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return mean_intensity, laplacian_variance


def adaptive_roi_preprocess(roi_bgr: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, EnhancementStats]:
    """Apply config-driven, ROI-only adaptive preprocessing for OCR readiness."""
    if roi_bgr is None or roi_bgr.size == 0:
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        return empty, EnhancementStats(0.0, 0.0, False, False)

    pre_cfg = cfg["pipeline"]["preprocessing"]
    brightness_th = float(pre_cfg["brightness_threshold"])
    blur_th = float(pre_cfg["blur_threshold"])

    mean_intensity, laplacian_variance = compute_roi_quality(roi_bgr)
    enhanced = roi_bgr.copy()

    clahe_applied = False
    if mean_intensity < brightness_th:
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        clahe_cfg = pre_cfg["clahe"]
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_cfg["clip_limit"]),
            tileGridSize=(int(clahe_cfg["tile_grid_size"]), int(clahe_cfg["tile_grid_size"])),
        )
        gray = clahe.apply(gray)
        enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        clahe_applied = True

    glare_reduction_applied = False
    if laplacian_variance < blur_th:
        k = int(pre_cfg["glare_reduction"]["kernel_size"])
        k = max(1, k)
        kernel = np.ones((k, k), np.uint8)
        # Mild top-hat reduction helps suppress small bright artifacts before OCR.
        bgr = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
        enhanced = cv2.addWeighted(enhanced, 0.8, bgr, 0.2, 0.0)
        glare_reduction_applied = True

    stats = EnhancementStats(
        mean_intensity=mean_intensity,
        laplacian_variance=laplacian_variance,
        clahe_applied=clahe_applied,
        glare_reduction_applied=glare_reduction_applied,
    )
    return enhanced, stats

