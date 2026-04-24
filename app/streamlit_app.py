"""KnightSight ANPR EdgeVision — Industry-Grade Dashboard."""

from __future__ import annotations

import json
import tempfile
from typing import Any, Dict, List

import cv2
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import ANPRPipeline

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KnightSight ANPR EdgeVision",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Orbitron:wght@700;900&display=swap');

/* ── Base Reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #090c10 !important;
    color: #c8d6e5 !important;
    font-family: 'Rajdhani', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1f2937 !important;
}
[data-testid="stSidebar"] * { color: #c8d6e5 !important; }
section[data-testid="stSidebarContent"] { padding-top: 1.5rem !important; }

/* ── Header ── */
.ks-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 0 10px 0;
    border-bottom: 1px solid #1a2236;
    margin-bottom: 20px;
}
.ks-logo {
    font-family: 'Orbitron', monospace;
    font-size: 1.65rem;
    font-weight: 900;
    letter-spacing: 2px;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.ks-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #4b6076;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-top: 2px;
}
.ks-badge {
    margin-left: auto;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 2px;
    padding: 4px 10px;
    border: 1px solid #1e3a5f;
    border-radius: 3px;
    color: #38bdf8;
    background: #0c1929;
}

/* ── Gate Status ── */
.gate-granted {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-align: center;
    color: #22c55e;
    background: linear-gradient(135deg, #052e16 0%, #0a1f0e 100%);
    border: 2px solid #15803d;
    border-radius: 6px;
    padding: 22px 16px;
    box-shadow: 0 0 24px #15803d55;
    animation: pulse-green 2s infinite;
}
.gate-denied {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-align: center;
    color: #ef4444;
    background: linear-gradient(135deg, #2d0a0a 0%, #1a0808 100%);
    border: 2px solid #991b1b;
    border-radius: 6px;
    padding: 22px 16px;
    box-shadow: 0 0 24px #991b1b55;
    animation: pulse-red 2s infinite;
}
@keyframes pulse-green { 0%,100%{box-shadow:0 0 18px #15803d55} 50%{box-shadow:0 0 36px #22c55e88} }
@keyframes pulse-red   { 0%,100%{box-shadow:0 0 18px #991b1b55} 50%{box-shadow:0 0 36px #ef444488} }
.gate-idle {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-align: center;
    color: #4b6076;
    background: #0d1117;
    border: 2px solid #1f2937;
    border-radius: 6px;
    padding: 22px 16px;
}

/* ── Plate Display ── */
.plate-valid {
    font-family: 'Orbitron', monospace;
    font-size: 3rem;
    font-weight: 900;
    text-align: center;
    letter-spacing: 6px;
    color: #f0f9ff;
    background: linear-gradient(135deg, #0c1929 0%, #0f2540 100%);
    border: 2px solid #38bdf8;
    border-radius: 8px;
    padding: 28px 20px;
    box-shadow: 0 0 32px #38bdf822;
}
.plate-unknown {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 700;
    text-align: center;
    letter-spacing: 4px;
    color: #f87171;
    background: #1a0808;
    border: 2px solid #7f1d1d;
    border-radius: 8px;
    padding: 28px 20px;
}
.plate-reason {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 2px;
    color: #6b7280;
    text-align: center;
    margin-top: 8px;
}

/* ── Raw OCR Panel ── */
.ocr-panel {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    background: #0d1117;
    border: 1px solid #1f2937;
    border-radius: 6px;
    padding: 16px;
    min-height: 130px;
    line-height: 1.8;
    color: #7dd3fc;
    overflow-y: auto;
}
.ocr-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #4b6076;
    margin-bottom: 8px;
}

/* ── Section Labels ── */
.section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 10px;
    border-bottom: 1px solid #1a2236;
    padding-bottom: 6px;
}

/* ── Pipeline Viz ── */
.pipeline-wrap {
    display: flex;
    align-items: center;
    gap: 0;
    background: #0d1117;
    border: 1px solid #1f2937;
    border-radius: 6px;
    padding: 18px 20px;
    overflow-x: auto;
}
.pip-stage {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 5px;
    min-width: 90px;
}
.pip-stage.active .pip-icon { background: #0c2d4a; border-color: #38bdf8; color: #38bdf8; box-shadow: 0 0 14px #38bdf855; }
.pip-stage.done   .pip-icon { background: #052e16; border-color: #22c55e; color: #22c55e; }
.pip-stage.idle   .pip-icon { background: #111827; border-color: #374151; color: #4b5563; }
.pip-icon {
    width: 48px; height: 48px;
    border-radius: 50%;
    border: 2px solid #374151;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    background: #111827;
    transition: all .3s;
}
.pip-name {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 1px;
    color: #6b7280;
    text-align: center;
}
.pip-arrow {
    font-size: 1.1rem;
    color: #1f2937;
    padding: 0 2px;
    margin-bottom: 22px;
}

/* ── Metric Cards ── */
[data-testid="stMetric"] {
    background: #0d1117 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 6px !important;
    padding: 14px 18px !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #4b6076 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.45rem !important;
    color: #e2e8f0 !important;
}

/* ── Event Log Table ── */
.event-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
}
.event-table th {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.68rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #4b6076;
    text-align: left;
    padding: 8px 12px;
    border-bottom: 1px solid #1f2937;
}
.event-table td {
    padding: 8px 12px;
    border-bottom: 1px solid #111827;
    color: #94a3b8;
    vertical-align: middle;
}
.event-table tr:hover td { background: #0d1117; }
.badge-valid { color: #22c55e; font-weight: 700; }
.badge-invalid { color: #ef4444; }
.badge-roi { color: #818cf8; font-size: 0.72rem; }
.conf-bar-wrap { background: #1f2937; border-radius: 4px; height: 6px; width: 80px; }
.conf-bar { height: 6px; border-radius: 4px; background: linear-gradient(90deg, #38bdf8, #818cf8); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #1f2937; border-radius: 3px; }

/* ── Upload Area ── */
[data-testid="stFileUploadDropzone"] {
    background: #0d1117 !important;
    border: 1px dashed #1f2937 !important;
    border-radius: 6px !important;
}
[data-testid="stFileUploadDropzone"] * { color: #4b6076 !important; }

/* ── Streamlit elements cleanup ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #0c2d4a 0%, #0f1f35 100%) !important;
    color: #38bdf8 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 4px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 1px !important;
}
.stDownloadButton > button:hover {
    background: #0c2d4a !important;
    box-shadow: 0 0 12px #38bdf833 !important;
}
div[data-testid="stImage"] img { border-radius: 6px; border: 1px solid #1f2937; }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ────────────────────────────────────────────────────────────────
def _draw_tracks(frame, results: List[Dict[str, Any]]):
    out = frame.copy()
    for item in results:
        vid = item["vehicle_id"]
        label = f"ID {vid} | {item['plate']} ({item['confidence']:.2f})"
        cv2.putText(out, label, (12, 24 + vid * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 1, cv2.LINE_AA)
    return out


def _gate_html(results: List[Dict[str, Any]], idle: bool = False) -> str:
    if idle:
        return '<div class="gate-idle">◈ AWAITING SCAN</div>'
    granted = any(r.get("valid") for r in results)
    if granted:
        return '<div class="gate-granted">✔ ACCESS GRANTED</div>'
    return '<div class="gate-denied">✖ ACCESS DENIED</div>'


def _plate_html(results: List[Dict[str, Any]]) -> str:
    valid = [r for r in results if r.get("valid") and r["plate"] != "UNKNOWN"]
    if not valid:
        unknowns = [r for r in results if r["plate"] == "UNKNOWN"]
        reason = unknowns[0].get("failure_reason", "OCR_INVALID") if unknowns else "NO_DETECTION"
        return (
            f'<div class="plate-unknown">◈ UNKNOWN</div>'
            f'<div class="plate-reason">REASON · {reason}</div>'
        )
    best = max(valid, key=lambda r: r["confidence"])
    return f'<div class="plate-valid">{best["plate"]}</div>'


def _raw_ocr_html(results: List[Dict[str, Any]]) -> str:
    lines = ""
    for r in results:
        raw = r.get("raw_ocr", "") or "&lt;empty&gt;"
        lines += f'<div><span style="color:#4b6076;">ID {r["vehicle_id"]} ›</span> {raw}</div>'
    if not lines:
        lines = '<div style="color:#374151;">— no data —</div>'
    return f'<div class="ocr-label">RAW OCR STREAM</div><div class="ocr-panel">{lines}</div>'


PIPELINE_STAGES = [
    ("🎯", "YOLO", "vehicle detection"),
    ("🔗", "TRACKER", "centroid tracking"),
    ("✂️", "ROI", "plate localization"),
    ("🔍", "OCR", "text extraction"),
    ("🧲", "STABILIZER", "temporal smoothing"),
    ("🚦", "GATE", "access control"),
]


def _pipeline_html(active_stage: int = -1) -> str:
    parts = []
    for i, (icon, name, _) in enumerate(PIPELINE_STAGES):
        if active_stage < 0:
            cls = "idle"
        elif i < active_stage:
            cls = "done"
        elif i == active_stage:
            cls = "active"
        else:
            cls = "idle"
        parts.append(
            f'<div class="pip-stage {cls}">'
            f'<div class="pip-icon">{icon}</div>'
            f'<div class="pip-name">{name}</div>'
            f'</div>'
        )
        if i < len(PIPELINE_STAGES) - 1:
            parts.append('<div class="pip-arrow">⟶</div>')
    return f'<div class="pipeline-wrap">{"".join(parts)}</div>'


def _event_table_html(all_results: List[Dict[str, Any]]) -> str:
    if not all_results:
        return '<div style="color:#374151;font-family:Share Tech Mono,monospace;font-size:.8rem;padding:12px;">— no events —</div>'

    rows = ""
    for r in reversed(all_results[-50:]):
        plate = r["plate"]
        valid = r.get("valid", False)
        conf = float(r.get("confidence", 0.0))
        lat = float(r.get("latency_ms", 0.0))
        roi = r.get("roi_strategy", "—")
        badge = f'<span class="badge-valid">✔ {plate}</span>' if valid else f'<span class="badge-invalid">✖ {plate}</span>'
        bar_w = int(conf * 80)
        conf_cell = f'<div class="conf-bar-wrap"><div class="conf-bar" style="width:{bar_w}px"></div></div><span style="font-size:.72rem;color:#64748b;margin-left:6px">{conf:.2f}</span>'
        rows += (
            f"<tr>"
            f"<td style='color:#38bdf8'>#{r['vehicle_id']}</td>"
            f"<td>{badge}</td>"
            f"<td><div style='display:flex;align-items:center'>{conf_cell}</div></td>"
            f"<td style='color:#818cf8'>{lat:.1f} ms</td>"
            f"<td><span class='badge-roi'>{roi}</span></td>"
            f"</tr>"
        )
    return (
        "<table class='event-table'>"
        "<thead><tr>"
        "<th>VID</th><th>PLATE</th><th>CONFIDENCE</th><th>LATENCY</th><th>ROI STRATEGY</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
    )


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-family:Orbitron,monospace;font-size:1rem;font-weight:900;'
        'letter-spacing:2px;color:#38bdf8;margin-bottom:18px">⬡ KNIGHTSIGHT</div>',
        unsafe_allow_html=True,
    )
    edge_mode = st.toggle("Edge Mode", value=True)
    debug_mode = st.toggle("Debug Mode", value=False)
    st.markdown("---")
    st.markdown(
        '<div style="font-family:Share Tech Mono,monospace;font-size:.65rem;'
        'letter-spacing:3px;color:#4b6076;margin-bottom:8px">SYSTEM INFO</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div style="font-family:Share Tech Mono,monospace;font-size:.72rem;line-height:2;color:#374151">
YOLOv8n · 3.2 GFLOPs<br>
OCR (PaddleOCR) · 0.5G<br>
Target Latency · 150ms<br>
Hard Cap · 250ms
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    uploaded = st.file_uploader(
        "UPLOAD MEDIA", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"], label_visibility="visible"
    )

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="ks-header">'
    '<div><div class="ks-logo">KnightSight</div>'
    '<div class="ks-sub">ANPR EdgeVision · v0.1.0</div></div>'
    '<div class="ks-badge">LIVE SYSTEM</div>'
    "</div>",
    unsafe_allow_html=True,
)

# ─── Idle state ─────────────────────────────────────────────────────────────
if uploaded is None:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown('<div class="section-label">VIDEO FEED</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="background:#0d1117;border:1px dashed #1f2937;border-radius:6px;'
            'height:260px;display:flex;align-items:center;justify-content:center;'
            'font-family:Share Tech Mono,monospace;font-size:.8rem;color:#1f2937;">'
            "NO SIGNAL</div>",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown('<div class="section-label">GATE STATUS</div>', unsafe_allow_html=True)
        st.markdown(_gate_html([], idle=True), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_c, col_d = st.columns([1, 1])
    with col_c:
        st.markdown('<div class="section-label">PLATE</div>', unsafe_allow_html=True)
        st.markdown(_plate_html([]), unsafe_allow_html=True)
    with col_d:
        st.markdown(_raw_ocr_html([]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">PIPELINE</div>', unsafe_allow_html=True)
    st.markdown(_pipeline_html(-1), unsafe_allow_html=True)
    st.stop()

# ─── Pipeline init ──────────────────────────────────────────────────────────
pipeline = ANPRPipeline("configs/config.yaml")
if edge_mode:
    pipeline.frame_skip = max(pipeline.frame_skip, 2)

with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded.name}") as tmp:
    tmp.write(uploaded.read())
    media_path = tmp.name

is_image = uploaded.type.startswith("image")

# ═══════════════════════════════════════════════════════════════════
# IMAGE MODE
# ═══════════════════════════════════════════════════════════════════
if is_image:
    frame = cv2.imread(media_path)
    payload = pipeline.process_frame(frame, frame_id=1)
    results = payload["results"]
    m = payload["metrics"]
    vis = _draw_tracks(frame, payload.get("valid_results", results))

    # ── Row 1: Video + Gate ──────────────────────────────────────
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown('<div class="section-label">VIDEO FEED</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col_b:
        st.markdown('<div class="section-label">GATE STATUS</div>', unsafe_allow_html=True)
        st.markdown(_gate_html(results), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if edge_mode:
            st.markdown(
                '<div style="font-family:Share Tech Mono,monospace;font-size:.68rem;'
                'letter-spacing:2px;color:#22c55e;padding:8px 12px;background:#052e16;'
                'border:1px solid #15803d;border-radius:4px">'
                "⚡ EDGE MODE ACTIVE<br>"
                '<span style="color:#4b6076">ROI-ONLY · SKIP ENABLED</span>'
                "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Plate + OCR ──────────────────────────────────────
    col_c, col_d = st.columns([1, 1])
    with col_c:
        st.markdown('<div class="section-label">STABILIZED PLATE</div>', unsafe_allow_html=True)
        st.markdown(_plate_html(results), unsafe_allow_html=True)
    with col_d:
        st.markdown(_raw_ocr_html(results), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pipeline Viz ─────────────────────────────────────────────
    st.markdown('<div class="section-label">PIPELINE</div>', unsafe_allow_html=True)
    st.markdown(_pipeline_html(active_stage=5), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metrics Row ──────────────────────────────────────────────
    st.markdown('<div class="section-label">PERFORMANCE METRICS</div>', unsafe_allow_html=True)
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("LATENCY (ms)", f"{m['total_latency_ms']:.1f}")
    mc2.metric("FPS", f"{m['fps']:.2f}")
    mc3.metric("MODEL SIZE (MB)", f"{m['total_model_size_mb']:.2f}")
    mc4.metric("ROBUSTNESS", f"{m['robustness_retention'] * 100:.1f}%")
    mc5.metric("CONSISTENCY", f"{m['temporal_consistency_score'] * 100:.1f}%")

    if debug_mode:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">DEBUG · LATENCY BREAKDOWN</div>', unsafe_allow_html=True)
        db1, db2, db3 = st.columns(3)
        db1.metric("DETECTION (ms)", f"{m['detection_latency_ms']:.1f}")
        db2.metric("OCR (ms)", f"{m['ocr_latency_ms']:.1f}")
        db3.metric("TRACKING (ms)", f"{m['tracking_latency_ms']:.1f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Event Log ────────────────────────────────────────────────
    st.markdown('<div class="section-label">EVENT LOG</div>', unsafe_allow_html=True)
    st.markdown(_event_table_html(results), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        "⬇ DOWNLOAD JSON REPORT",
        data=json.dumps(payload, indent=2),
        file_name="anpr_output.json",
        mime="application/json",
    )

# ═══════════════════════════════════════════════════════════════════
# VIDEO MODE
# ═══════════════════════════════════════════════════════════════════
else:
    cap = cv2.VideoCapture(media_path)
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown('<div class="section-label">VIDEO FEED</div>', unsafe_allow_html=True)
        frame_slot = st.empty()
    with col_b:
        st.markdown('<div class="section-label">GATE STATUS</div>', unsafe_allow_html=True)
        gate_slot = st.empty()

    st.markdown("<br>", unsafe_allow_html=True)
    col_c, col_d = st.columns([1, 1])
    with col_c:
        st.markdown('<div class="section-label">STABILIZED PLATE</div>', unsafe_allow_html=True)
        plate_slot = st.empty()
    with col_d:
        ocr_slot = st.empty()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">PIPELINE</div>', unsafe_allow_html=True)
    pip_slot = st.empty()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">PERFORMANCE METRICS</div>', unsafe_allow_html=True)
    m_cols = st.columns(5)
    metric_slots = [c.empty() for c in m_cols]
    metric_labels = ["LATENCY (ms)", "FPS", "MODEL SIZE (MB)", "ROBUSTNESS", "CONSISTENCY"]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">EVENT LOG</div>', unsafe_allow_html=True)
    table_slot = st.empty()

    all_payloads: List[Dict[str, Any]] = []
    all_flat_results: List[Dict[str, Any]] = []
    frame_id = 0

    pip_slot.markdown(_pipeline_html(active_stage=0), unsafe_allow_html=True)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        if frame_id % pipeline.frame_skip != 0:
            continue

        pip_slot.markdown(_pipeline_html(active_stage=1), unsafe_allow_html=True)
        payload = pipeline.process_frame(frame, frame_id=frame_id)
        all_payloads.append(payload)
        results = payload["results"]
        all_flat_results.extend(results)
        m = payload["metrics"]

        vis = _draw_tracks(frame, payload.get("valid_results", results))
        frame_slot.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_id}", use_container_width=True)

        gate_slot.markdown(_gate_html(results), unsafe_allow_html=True)
        plate_slot.markdown(_plate_html(results), unsafe_allow_html=True)
        ocr_slot.markdown(_raw_ocr_html(results), unsafe_allow_html=True)
        pip_slot.markdown(_pipeline_html(active_stage=5), unsafe_allow_html=True)

        metric_vals = [
            f"{m['total_latency_ms']:.1f}",
            f"{m['fps']:.2f}",
            f"{m['total_model_size_mb']:.2f}",
            f"{m['robustness_retention'] * 100:.1f}%",
            f"{m['temporal_consistency_score'] * 100:.1f}%",
        ]
        for slot, lbl, val in zip(metric_slots, metric_labels, metric_vals):
            slot.metric(lbl, val)

        table_slot.markdown(_event_table_html(all_flat_results), unsafe_allow_html=True)

        if debug_mode:
            with st.expander(f"[DEBUG] Frame {frame_id}", expanded=False):
                st.json({"enhancement_stats": m.get("enhancement_stats", {}), "results": results})

    cap.release()

    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        "⬇ DOWNLOAD JSON REPORT",
        data=json.dumps(all_payloads, indent=2),
        file_name="anpr_video_output.json",
        mime="application/json",
    )

