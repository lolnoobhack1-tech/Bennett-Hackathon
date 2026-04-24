"""Multi-frame OCR stabilization for ANPR plate finalization."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import re
from typing import Any, Deque, Dict, List, Optional


@dataclass(frozen=True)
class OCRCandidate:
    """Single OCR candidate for stabilization voting."""

    text: str
    confidence: float
    frame_id: int


class PlateStabilizer:
    """Per-vehicle OCR stabilizer with validation and fallback handling."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        stab_cfg = cfg["pipeline"]["stabilization"]
        self._history_size = int(stab_cfg["history_size"])
        self._sliding_window_frames = int(
            stab_cfg.get("sliding_window_frames", self._history_size)
        )
        self._min_frames_before_unknown = int(stab_cfg["min_frames_before_unknown"])
        self._min_conf = float(stab_cfg["min_confidence"])
        self._plate_regex = re.compile(str(stab_cfg["regex"]))
        self._substitutions = {str(k): str(v) for k, v in stab_cfg["substitutions"].items()}
        self._history: Dict[int, Deque[OCRCandidate]] = defaultdict(
            lambda: deque(maxlen=self._history_size)
        )

    def _normalize(self, text: str) -> str:
        normalized = (text or "").strip().upper().replace(" ", "").replace("-", "")
        for src, dst in self._substitutions.items():
            normalized = normalized.replace(src, dst)
        return normalized

    def _is_valid(self, text: str) -> bool:
        return bool(self._plate_regex.match(text))

    def update(
        self, vehicle_id: int, raw_text: str, confidence: float, frame_id: int
    ) -> Dict[str, Any]:
        """
        Add a new OCR observation and compute stabilized output.

        Always returns a structured JSON-compatible dict and never an empty response.
        """
        normalized = self._normalize(raw_text)
        confidence = float(confidence)
        self._history[int(vehicle_id)].append(
            OCRCandidate(
                text=normalized,
                confidence=confidence,
                frame_id=int(frame_id),
            )
        )
        return self.get_output(vehicle_id, frame_id=frame_id)

    def get_output(self, vehicle_id: int, frame_id: Optional[int] = None) -> Dict[str, Any]:
        """Return stabilized plate state for a vehicle with UNKNOWN fallback."""
        vid = int(vehicle_id)
        history_all = list(self._history.get(vid, []))
        if frame_id is None:
            frame_id = history_all[-1].frame_id if history_all else 0
        frame_id = int(frame_id)
        history = [
            item
            for item in history_all
            if item.frame_id >= frame_id - self._sliding_window_frames + 1
        ]

        if not history:
            return {
                "vehicle_id": vid,
                "plate": "UNKNOWN",
                "valid": False,
                "confidence": 0.0,
            }

        valid_items = [
            it for it in history if self._is_valid(it.text) and it.confidence >= self._min_conf
        ]
        if valid_items:
            freq = Counter(item.text for item in valid_items)
            groups = {}
            for plate_text in freq:
                group = [x for x in valid_items if x.text == plate_text]
                avg_conf = sum(x.confidence for x in group) / len(group)
                recent_frame = max(x.frame_id for x in group)
                groups[plate_text] = (freq[plate_text], avg_conf, recent_frame)
            best_plate = max(groups, key=lambda k: (groups[k][0], groups[k][1], groups[k][2]))
            best_count, best_avg_conf, _ = groups[best_plate]
            # Frequency dominates; confidence breaks ties through highest-confidence read.
            support = best_count / max(1, len(history))

            return {
                "vehicle_id": vid,
                "plate": best_plate,
                "valid": True,
                "confidence": float(round(best_avg_conf, 4)),
                "support": float(round(support, 4)),
                "raw_candidates": [item.text for item in history if item.text],
            }

        if len(history) >= self._min_frames_before_unknown:
            return {
                "vehicle_id": vid,
                "plate": "UNKNOWN",
                "valid": False,
                "confidence": 0.0,
                "raw_candidates": [item.text for item in history if item.text],
            }

        best_unvalidated: Optional[OCRCandidate] = None
        for item in history:
            if not item.text:
                continue
            if best_unvalidated is None or item.confidence > best_unvalidated.confidence:
                best_unvalidated = item

        if best_unvalidated is None:
            return {
                "vehicle_id": vid,
                "plate": "UNKNOWN",
                "valid": False,
                "confidence": 0.0,
                "raw_candidates": [],
            }

        return {
            "vehicle_id": vid,
            "plate": best_unvalidated.text,
            "valid": False,
            "confidence": float(round(best_unvalidated.confidence, 4)),
            "raw_candidates": [item.text for item in history if item.text],
        }

    def snapshot(self) -> Dict[int, List[Dict[str, Any]]]:
        """Return internal state as a JSON-serializable snapshot for debugging."""
        return {
            vehicle_id: [
                {"text": item.text, "confidence": item.confidence, "frame_id": item.frame_id}
                for item in items
            ]
            for vehicle_id, items in self._history.items()
        }

