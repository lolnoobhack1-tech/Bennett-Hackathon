"""Pipeline profiling and metric aggregation utilities."""

from __future__ import annotations

import os
from typing import Any, Dict, List


class Profiler:
    """Collects latency, FPS, FLOPs estimate, and enhancement diagnostics."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self._records: List[Dict[str, Any]] = []
        self._accurate_noisy = 0
        self._accurate_clean = 0
        self._yolo_flops_g = 3.2
        self._ocr_flops_g = 0.5

    @staticmethod
    def _file_size_mb(path: str) -> float:
        """Return file size in MB if path exists."""
        if not path:
            return 0.0
        if os.path.exists(path):
            return os.path.getsize(path) / (1024 * 1024)
        return 0.0

    def _model_size_breakdown(self) -> Dict[str, float]:
        """Compute real model footprint from filesystem."""
        yolo_path = str(self.cfg["models"]["vehicle_detector"]["name"])
        ocr_path = str(self.cfg["models"]["ocr"].get("model_path", ""))
        yolo_mb = self._file_size_mb(yolo_path)
        ocr_mb = self._file_size_mb(ocr_path)
        return {
            "yolo_model_size_mb": float(round(yolo_mb, 4)),
            "ocr_model_size_mb": float(round(ocr_mb, 4)),
            "total_model_size_mb": float(round(yolo_mb + ocr_mb, 4)),
        }

    def update(
        self,
        detection_ms: float,
        tracking_ms: float,
        ocr_ms: float,
        total_ms: float,
        enhancement_stats: Dict[str, Any],
        accurate_predictions_on_noisy_set: int = 0,
        accurate_predictions_on_clean_set: int = 0,
    ) -> Dict[str, Any]:
        """Store one frame metrics and return current snapshot."""
        self._accurate_noisy += int(accurate_predictions_on_noisy_set)
        self._accurate_clean += int(accurate_predictions_on_clean_set)

        robustness_retention = (
            float(self._accurate_noisy) / float(self._accurate_clean)
            if self._accurate_clean > 0
            else 0.0
        )
        fps = 1000.0 / float(total_ms) if total_ms > 0 else 0.0
        sizes = self._model_size_breakdown()
        estimated_flops_g = self._yolo_flops_g + self._ocr_flops_g

        row = {
            "detection_latency_ms": float(detection_ms),
            "tracking_latency_ms": float(tracking_ms),
            "ocr_latency_ms": float(ocr_ms),
            "total_latency_ms": float(total_ms),
            "fps": float(fps),
            "yolo_model_size_mb": sizes["yolo_model_size_mb"],
            "ocr_model_size_mb": sizes["ocr_model_size_mb"],
            "total_model_size_mb": sizes["total_model_size_mb"],
            "estimated_flops_g": float(estimated_flops_g),
            "robustness_retention": float(round(robustness_retention, 4)),
            "temporal_consistency_score": float(round(robustness_retention, 4)),
            "enhancement_stats": enhancement_stats,
        }
        self._records.append(row)
        return row

    def latest(self) -> Dict[str, Any]:
        """Get latest record or defaults."""
        if not self._records:
            return {
                "detection_latency_ms": 0.0,
                "tracking_latency_ms": 0.0,
                "ocr_latency_ms": 0.0,
                "total_latency_ms": 0.0,
                "fps": 0.0,
                "yolo_model_size_mb": 0.0,
                "ocr_model_size_mb": 0.0,
                "total_model_size_mb": 0.0,
                "estimated_flops_g": 3.7,
                "robustness_retention": 0.0,
                "temporal_consistency_score": 0.0,
                "enhancement_stats": {},
            }
        return self._records[-1]


