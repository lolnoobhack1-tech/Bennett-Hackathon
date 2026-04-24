"""Lightweight ROI-constrained plate localization utilities."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


class PlateLocator:
    """Locate probable license plate regions within a vehicle bottom ROI."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        pl_cfg = cfg.get("plate_localization", {})
        self.enabled = bool(pl_cfg.get("enabled", True))
        self.min_aspect_ratio = float(pl_cfg.get("min_aspect_ratio", 2.0))
        self.max_aspect_ratio = float(pl_cfg.get("max_aspect_ratio", 6.5))
        self.min_area = int(pl_cfg.get("min_area", 250))
        self.min_width = int(pl_cfg.get("min_width", 24))
        self.min_height = int(pl_cfg.get("min_height", 10))
        self.canny_threshold1 = int(pl_cfg.get("canny_threshold1", 50))
        self.canny_threshold2 = int(pl_cfg.get("canny_threshold2", 150))
        self.fallback_to_bottom_roi = bool(pl_cfg.get("fallback_to_bottom_roi", True))

    def _fallback(self, vehicle_bottom_roi: np.ndarray) -> Dict[str, Any]:
        h, w = (0, 0)
        if vehicle_bottom_roi is not None and vehicle_bottom_roi.size > 0:
            h, w = vehicle_bottom_roi.shape[:2]
        return {
            "plate_roi": vehicle_bottom_roi,
            "plate_bbox_local": (0, 0, int(w), int(h)),
            "method": "BOTTOM_40_FALLBACK",
            "confidence": 0.0,
        }

    def localize(self, vehicle_bottom_roi: np.ndarray) -> Dict[str, Any]:
        """Return localized plate ROI inside the provided vehicle bottom ROI."""
        if (
            not self.enabled
            or vehicle_bottom_roi is None
            or vehicle_bottom_roi.size == 0
            or vehicle_bottom_roi.shape[0] < 2
            or vehicle_bottom_roi.shape[1] < 2
        ):
            return self._fallback(vehicle_bottom_roi)

        gray = cv2.cvtColor(vehicle_bottom_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
        edges = cv2.Canny(gray, self.canny_threshold1, self.canny_threshold2)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
        merged = cv2.bitwise_or(edges, thresh)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._fallback(vehicle_bottom_roi)

        target_aspect = (self.min_aspect_ratio + self.max_aspect_ratio) / 2.0
        roi_h, roi_w = vehicle_bottom_roi.shape[:2]
        max_area = float(max(1, roi_h * roi_w))

        best_bbox: Optional[Tuple[int, int, int, int]] = None
        best_score = float("-inf")

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 0 or h <= 0:
                continue
            area = w * h
            if area < self.min_area or w < self.min_width or h < self.min_height:
                continue
            aspect_ratio = float(w) / float(h)
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True) if peri > 0 else []
            rectangular_bonus = 0.25 if len(approx) == 4 else 0.0
            aspect_penalty = abs(aspect_ratio - target_aspect)
            area_score = float(area) / max_area
            score = area_score - (0.08 * aspect_penalty) + rectangular_bonus
            if score > best_score:
                best_score = score
                best_bbox = (x, y, x + w, y + h)

        if best_bbox is None:
            return self._fallback(vehicle_bottom_roi)

        x1, y1, x2, y2 = best_bbox
        x1 = max(0, min(roi_w - 1, x1))
        y1 = max(0, min(roi_h - 1, y1))
        x2 = max(x1 + 1, min(roi_w, x2))
        y2 = max(y1 + 1, min(roi_h, y2))
        plate_roi = vehicle_bottom_roi[y1:y2, x1:x2]
        if plate_roi.size == 0:
            return self._fallback(vehicle_bottom_roi)

        confidence = min(1.0, max(0.0, best_score))
        return {
            "plate_roi": plate_roi,
            "plate_bbox_local": (int(x1), int(y1), int(x2), int(y2)),
            "method": "CONTOUR_PLATE_CANDIDATE",
            "confidence": float(round(confidence, 4)),
        }

