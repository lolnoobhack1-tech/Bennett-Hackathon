# KnightSight ANPR EdgeVision

Deployment-oriented, ROI-first ANPR pipeline optimized for edge demos and defensible judging.

## Architecture

Input -> Adaptive Frame Skipping -> YOLOv8n Vehicle Detection -> Centroid Tracking -> Vehicle ROI (Bottom 40%)
-> Plate Localization (ROI-constrained OpenCV contour candidate, cached ROI, fallback to bottom ROI)
-> ROI-only Enhancement -> EasyOCR/PaddleOCR backend -> Multi-frame Stabilization -> JSON Output

## Detection and OCR Stack

- Vehicle detection: YOLOv8n
- Plate localization: lightweight ROI-constrained OpenCV contouring + bottom-40% fallback
- OCR: EasyOCR/PaddleOCR (depends on configured backend)
- Stabilization: multi-frame voting, regex validation, and character substitutions

## Efficiency Metrics

- YOLOv8n: ~3.2 GFLOPs @ 640x640
- OCR: ~0.4-0.6 GFLOPs (approx)
- Total pipeline: <= 4 GFLOPs
- Real model footprint is computed at runtime from filesystem paths.

## Robustness Metrics

Live demo proxy is reported as:

`temporal_consistency_score` (derived from stabilized consistency over incoming frames).

Benchmark-compatible robustness is reported as:

`robustness_retention = accurate_predictions_on_noisy_set / accurate_predictions_on_clean_set`

Use `robustness_retention` as a true benchmark metric only when clean/noisy validation counts are computed on separate validation subsets. Both fields are exposed in profiler output, JSON payloads, and Streamlit labels.

## Output Format

```json
{
  "vehicle_id": 1,
  "plate": "MH12AB1234",
  "valid": true,
  "confidence": 0.91,
  "latency_ms": 42.7
}
```

## Run

```bash
python main.py --video path/to/video.mp4 --config configs/config.yaml
streamlit run app/streamlit_app.py
```

