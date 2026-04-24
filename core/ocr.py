"""EasyOCR wrapper for ROI-only text recognition."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np
import easyocr


class OCREngine:
    """Config-driven OCR engine. Accepts ROI only."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.ocr = None
        try:
            # Initialize EasyOCR with English language
            self.ocr = easyocr.Reader(['en'])
            print("OCR: EasyOCR initialized successfully")
        except Exception as err:
            # Keep pipeline alive even if OCR backend cannot initialize
            self.ocr = None
            self._init_error = str(err)
            print(f"OCR initialization failed: {self._init_error}")

    def read_text(self, roi_bgr: np.ndarray) -> Tuple[str, float]:
        """Return (text, confidence) from ROI. Never pass full frame here."""
        if roi_bgr is None or roi_bgr.size == 0:
            print("OCR: ROI is empty or None")
            return "", 0.0
        if self.ocr is None:
            print("OCR: OCR engine not initialized")
            return "", 0.0

        print(f"OCR: Processing ROI shape: {roi_bgr.shape}")
        
        # EasyOCR expects RGB, convert BGR to RGB
        if len(roi_bgr.shape) == 3 and roi_bgr.shape[2] == 3:
            roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        else:
            roi_rgb = roi_bgr
            
        result = self.ocr.readtext(roi_rgb)
        print(f"OCR: Raw result: {result}")
        
        if not result:
            print("OCR: No text detected in ROI")
            return "", 0.0

        texts = []
        confidences = []
        for detection in result:
            # EasyOCR format: (bbox, text, confidence)
            if len(detection) >= 3:
                bbox, txt, conf = detection[0], detection[1], detection[2]
                txt = str(txt).strip()
                print(f"OCR: Found text: '{txt}' with confidence {conf:.3f}")
                if txt:
                    texts.append(txt)
                    confidences.append(conf)

        if not texts:
            print("OCR: No valid text extracted")
            return "", 0.0
        merged_text = "".join(texts)
        avg_conf = sum(confidences) / len(confidences)
        print(f"OCR: Final result: '{merged_text}' with avg confidence {avg_conf:.3f}")
        return merged_text, float(avg_conf)


