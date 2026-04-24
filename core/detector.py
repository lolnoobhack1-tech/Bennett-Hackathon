"""YOLO-based vehicle and plate detection utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
from utils.roi import extract_bottom_roi


class YoloDetector:
    """Config-driven detector wrapper for vehicles and plates."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self._vehicle_cfg = cfg["models"]["vehicle_detector"]
        self._plate_cfg = cfg["models"]["plate_detector"]
        self._vehicle_model = YOLO(str(self._vehicle_cfg["name"]))
        self._plate_model = YOLO(str(self._plate_cfg["name"]))
        self._vehicle_class_ids = set(int(x) for x in self._vehicle_cfg.get("vehicle_class_ids", [2, 3, 5, 7]))

    @staticmethod
    def _run_model(
        model: YOLO,
        image_bgr: np.ndarray,
        input_size: int,
        conf: float,
        iou: float,
        device: str,
    ) -> List[Tuple[Tuple[int, int, int, int], float, int]]:
        """Run YOLO model and return (bbox, confidence, class_id) tuples."""
        results = model.predict(
            source=image_bgr,
            imgsz=input_size,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
        detections: List[Tuple[Tuple[int, int, int, int], float, int]] = []
        if not results:
            return detections
        res = results[0]
        if res.boxes is None:
            return detections
        for box in res.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            score = float(box.conf[0].cpu().item())
            cls_id = int(box.cls[0].cpu().item())
            detections.append(((xyxy[0], xyxy[1], xyxy[2], xyxy[3]), score, cls_id))
        return detections

    def detect_vehicles(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect vehicle-like objects on a frame."""
        # Use original frame without resizing to preserve aspect ratio
        raw = self._run_model(
            model=self._vehicle_model,
            image_bgr=frame_bgr,
            input_size=int(self._vehicle_cfg["input_size"]),
            conf=float(self._vehicle_cfg["confidence"]),
            iou=float(self._vehicle_cfg["iou"]),
            device=str(self._vehicle_cfg["device"]),
        )
        outputs: List[Dict[str, Any]] = []
        print(f"Raw detections: {len(raw)}")
        for bbox, conf, cls_id in raw:
            print(f"Detection: bbox={bbox}, conf={conf:.3f}, class={cls_id}, allowed={cls_id in self._vehicle_class_ids}")
            if cls_id not in self._vehicle_class_ids:
                continue
            outputs.append({"bbox": bbox, "confidence": conf, "class_id": cls_id})
        print(f"Filtered vehicle detections: {len(outputs)}")
        return outputs

    def detect_plate_in_roi(
        self, roi_bgr: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Detect plate in ROI only. Returns local ROI bbox."""
        if roi_bgr is None or roi_bgr.size == 0:
            print("ROI is empty or None")
            return None
        print(f"ROI shape: {roi_bgr.shape}")
        raw = self._run_model(
            model=self._plate_model,
            image_bgr=roi_bgr,
            input_size=int(self._plate_cfg["input_size"]),
            conf=float(self._plate_cfg["confidence"]),
            iou=float(self._plate_cfg["iou"]),
            device=str(self._plate_cfg["device"]),
        )
        print(f"Plate detections in ROI: {len(raw)}")
        for bbox, conf, cls_id in raw:
            print(f"Plate candidate: bbox={bbox}, conf={conf:.3f}, class={cls_id}")
        if not raw:
            return None
        bbox, conf, cls_id = max(raw, key=lambda d: d[1])
        print(f"Selected plate: bbox={bbox}, conf={conf:.3f}, class={cls_id}")
        return {"bbox": bbox, "confidence": conf, "class_id": cls_id}

    def extract_vehicle_bottom_roi(self, frame_bgr: np.ndarray, vehicle_bbox: Tuple[int, int, int, int]):
        """Shared ROI utility wrapper used by pipeline."""
        ratio = float(self.cfg["pipeline"]["roi"]["vehicle_bottom_ratio"])
        return extract_bottom_roi(frame_bgr, vehicle_bbox, ratio=ratio)


