"""Lightweight centroid-based tracker for stable vehicle IDs."""

from __future__ import annotations

from typing import Dict, List, Tuple


class CentroidTracker:
    """Centroid-distance tracker with simple aging."""

    def __init__(self, max_disappeared: int = 20, max_distance: float = 80.0) -> None:
        self.next_id = 1
        self.objects: Dict[int, Tuple[int, int, int, int]] = {}
        self.centroids: Dict[int, Tuple[float, float]] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = int(max_disappeared)
        self.max_distance = float(max_distance)

    @staticmethod
    def _centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _register(self, bbox: Tuple[int, int, int, int]) -> int:
        object_id = self.next_id
        self.next_id += 1
        self.objects[object_id] = bbox
        self.centroids[object_id] = self._centroid(bbox)
        self.disappeared[object_id] = 0
        return object_id

    def _deregister(self, object_id: int) -> None:
        self.objects.pop(object_id, None)
        self.centroids.pop(object_id, None)
        self.disappeared.pop(object_id, None)

    def update(self, detections: List[Tuple[int, int, int, int]]) -> List[Dict]:
        """Update tracker with detection bboxes and return tracked objects."""
        if not detections:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            return [{"vehicle_id": obj_id, "bbox": bbox} for obj_id, bbox in self.objects.items()]

        input_centroids = [self._centroid(bbox) for bbox in detections]

        if not self.objects:
            for bbox in detections:
                self._register(bbox)
            return [{"vehicle_id": obj_id, "bbox": bbox} for obj_id, bbox in self.objects.items()]

        object_ids = list(self.objects.keys())
        unmatched_detections = set(range(len(detections)))

        for object_id in object_ids:
            prev_centroid = self.centroids[object_id]
            best_idx = None
            best_dist = float("inf")
            for idx in list(unmatched_detections):
                cx, cy = input_centroids[idx]
                dist = ((prev_centroid[0] - cx) ** 2 + (prev_centroid[1] - cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx is not None and best_dist <= self.max_distance:
                self.objects[object_id] = detections[best_idx]
                self.centroids[object_id] = input_centroids[best_idx]
                self.disappeared[object_id] = 0
                unmatched_detections.discard(best_idx)
            else:
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)

        for idx in unmatched_detections:
            self._register(detections[idx])

        return [{"vehicle_id": obj_id, "bbox": bbox} for obj_id, bbox in self.objects.items()]


