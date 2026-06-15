"""
Support Integrity Auditor — Streamlit Web Application
=====================================================
Premium dark-mode dashboard for detecting priority mismatches in
customer-support tickets using self-supervised severity inference.

Features:
  • Single-ticket form input or batch CSV upload
  • Binary judgment + full Evidence Dossier per ticket
  • Priority Mismatch Dashboard: flagged-ticket distribution,
    mismatch types, and top contributing signals
  • Severity-delta heatmap across categories × channels
"""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from predict import (
    apply_pseudo_labeler,
    build_classifier_features,
    load_and_prepare_for_prediction,
    make_dossier,
)
from train_pipeline import SEVERITY_INV, SEVERITY_MAP

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Support Integrity Auditor · SIA",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_DIR = Path("models")

# ── Premium colour palette ───────────────────────────────────────────────────
_BG         = "#0B0F19"
_CARD       = "#111827"
_CARD_EDGE  = "#1E293B"
_ACCENT     = "#6366F1"   # indigo-500
_ACCENT2    = "#8B5CF6"   # violet-500
_SUCCESS    = "#10B981"
_WARNING    = "#F59E0B"
_DANGER     = "#EF4444"
_TEXT       = "#E2E8F0"
_TEXT_DIM   = "#94A3B8"
_GRADIENT   = "linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #A78BFA 100%)"

# ── Plotly shared template ───────────────────────────────────────────────────
_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color=_TEXT, size=13),
    margin=dict(l=40, r=20, t=50, b=40),
    colorway=["#6366F1", "#8B5CF6", "#A78BFA", "#C4B5FD",
              "#10B981", "#F59E0B", "#EF4444", "#EC4899"],
    xaxis=dict(gridcolor="#1E293B", zerolinecolor="#1E293B"),
    yaxis=dict(gridcolor="#1E293B", zerolinecolor="#1E293B"),
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CSS — dark glassmorphism theme
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── global ──────────────────────────────────── */
    html, body, [data-testid="stApp"] {
        background-color: """ + _BG + """;
        color: """ + _TEXT + """;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #0F172A; }

    /* ── glass cards ─────────────────────────────── */
    .glass-card {
        background: rgba(17,24,39,0.72);
        backdrop-filter: blur(16px) saturate(1.6);
        -webkit-backdrop-filter: blur(16px) saturate(1.6);
        border: 1px solid rgba(99,102,241,0.18);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 16px;
        transition: box-shadow .3s, transform .25s;
    }
    .glass-card:hover {
        box-shadow: 0 8px 32px rgba(99,102,241,0.18);
        transform: translateY(-2px);
    }

    /* ── hero banner ─────────────────────────────── */
    .hero {
        background: linear-gradient(135deg,
            rgba(99,102,241,0.22) 0%,
            rgba(139,92,246,0.16) 50%,
            rgba(167,139,250,0.10) 100%);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 20px;
        padding: 40px 44px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -60%; left: -30%;
        width: 200%; height: 200%;
        background: radial-gradient(
            ellipse at 30% 40%,
            rgba(99,102,241,0.12) 0%,
            transparent 60%);
        pointer-events: none;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 800;
        background: """ + _GRADIENT + """;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
        letter-spacing: -0.03em;
    }
    .hero p {
        color: """ + _TEXT_DIM + """;
        font-size: 1.08rem;
        margin: 0;
        max-width: 680px;
    }

    /* ── KPI pill ────────────────────────────────── */
    .kpi-pill {
        background: rgba(17,24,39,0.70);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(99,102,241,0.20);
        border-radius: 14px;
        padding: 22px 24px;
        text-align: center;
        transition: all .3s;
    }
    .kpi-pill:hover {
        border-color: rgba(99,102,241,0.50);
        box-shadow: 0 4px 24px rgba(99,102,241,0.14);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.04em;
    }
    .kpi-label {
        font-size: 0.82rem;
        color: """ + _TEXT_DIM + """;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }

    /* ── judgment badges ─────────────────────────── */
    .badge-mismatch {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
    }
    .badge-hidden-crisis {
        background: rgba(239,68,68,0.18);
        color: #F87171;
        border: 1px solid rgba(239,68,68,0.30);
    }
    .badge-false-alarm {
        background: rgba(245,158,11,0.18);
        color: #FBBF24;
        border: 1px solid rgba(245,158,11,0.30);
    }
    .badge-consistent {
        background: rgba(16,185,129,0.18);
        color: #34D399;
        border: 1px solid rgba(16,185,129,0.30);
    }

    /* ── dossier accordion ───────────────────────── */
    .dossier-box {
        background: rgba(17,24,39,0.60);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }
    .dossier-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #C4B5FD;
    }
    .evidence-tag {
        display: inline-block;
        background: rgba(99,102,241,0.14);
        color: #A5B4FC;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 500;
        margin: 2px 4px 2px 0;
    }

    /* ── section headers ─────────────────────────── */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: """ + _TEXT + """;
        margin: 32px 0 16px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-header .icon {
        font-size: 1.4rem;
    }

    /* ── streamlit overrides ─────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(17,24,39,0.50);
        border-radius: 12px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: """ + _TEXT_DIM + """;
        font-weight: 600;
        padding: 10px 24px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99,102,241,0.22) !important;
        color: #A5B4FC !important;
    }

    /* form styling */
    .stTextInput input, .stNumberInput input, .stTextArea textarea,
    .stSelectbox > div > div {
        background: rgba(17,24,39,0.70) !important;
        border: 1px solid rgba(99,102,241,0.20) !important;
        border-radius: 10px !important;
        color: """ + _TEXT + """ !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus,
    .stTextArea textarea:focus {
        border-color: """ + _ACCENT + """ !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.18) !important;
    }

    /* primary button */
    .stFormSubmitButton button, .stDownloadButton button {
        background: """ + _GRADIENT + """ !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 10px 28px !important;
        letter-spacing: 0.02em;
        transition: opacity .2s, transform .15s;
    }
    .stFormSubmitButton button:hover, .stDownloadButton button:hover {
        opacity: 0.90;
        transform: translateY(-1px);
    }

    /* file uploader */
    [data-testid="stFileUploader"] {
        background: rgba(17,24,39,0.50);
        border: 2px dashed rgba(99,102,241,0.30);
        border-radius: 14px;
        padding: 20px;
    }

    /* dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 12px;
        overflow: hidden;
    }

    /* expander override */
    .stExpander {
        background: rgba(17,24,39,0.50);
        border: 1px solid rgba(99,102,241,0.14);
        border-radius: 12px;
    }

    /* metric override */
    [data-testid="stMetric"] {
        background: rgba(17,24,39,0.55);
        border: 1px solid rgba(99,102,241,0.16);
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="stMetricValue"] {
        font-weight: 800 !important;
    }

    /* scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0B0F19; }
    ::-webkit-scrollbar-thumb {
        background: #1E293B;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover { background: #334155; }

    /* hide default streamlit footer/menu */
    #MainMenu, footer { visibility: hidden; }

    /* fade-in animation */
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .fade-in { animation: fadeSlideUp 0.55s ease-out both; }
    </style>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_resource
def load_artifacts(model_dir: Path):
    """Load trained pseudo-label and classifier artefacts."""
    pseudo_path = model_dir / "pseudo_label_artifacts.joblib"
    clf_path    = model_dir / "classifier_artifacts.joblib"
    if not pseudo_path.exists() or not clf_path.exists():
        return None, None
    return joblib.load(pseudo_path), joblib.load(clf_path)


def run_sia(df_input: pd.DataFrame, pseudo, clf_artifacts):
    """Run the full SIA inference pipeline on *df_input*."""
    with NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        df_input.to_csv(tmp.name, index=False)
        prepared = load_and_prepare_for_prediction(tmp.name)

    prepared = apply_pseudo_labeler(prepared, pseudo)
    x = build_classifier_features(prepared, clf_artifacts)
    probs = clf_artifacts["classifier"].predict_proba(x)[:, 1]
    preds = (probs >= clf_artifacts["threshold"]).astype(int)

    result = prepared.copy()
    result["predicted_mismatch"]    = preds
    result["mismatch_probability"]  = probs
    result["assigned_priority"]     = result["assigned_ordinal"].map(SEVERITY_INV)
    result["inferred_severity"]     = result["inferred_severity_ordinal"].map(SEVERITY_INV)
    result["mismatch_type"] = result["severity_delta"].apply(
        lambda d: "Hidden Crisis" if d > 0 else ("False Alarm" if d < 0 else "Consistent")
    )

    dossiers = []
    for i, row in result.iterrows():
        if int(row["predicted_mismatch"]) == 1 and int(row["severity_delta"]) != 0:
            dossiers.append(make_dossier(row, probs[i], pseudo["keywords"]))
    return result, dossiers


def _badge_class(mtype: str) -> str:
    if mtype == "Hidden Crisis":
        return "badge-hidden-crisis"
    if mtype == "False Alarm":
        return "badge-false-alarm"
    return "badge-consistent"


def _delta_color(val: float) -> str:
    if val > 0.5:
        return _DANGER
    if val < -0.5:
        return _WARNING
    return _SUCCESS


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  KPI row
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_kpi_row(result: pd.DataFrame):
    total   = len(result)
    flagged = int(result["predicted_mismatch"].sum())
    rate    = flagged / max(total, 1)
    hidden  = int((result["mismatch_type"] == "Hidden Crisis").sum())
    false_a = int((result["mismatch_type"] == "False Alarm").sum())
    avg_conf = result.loc[result["predicted_mismatch"] == 1, "mismatch_probability"].mean()
    avg_conf = avg_conf if not np.isnan(avg_conf) else 0.0

    cols = st.columns(6)
    kpis = [
        ("🎫", str(total),           "Tickets Audited",  _ACCENT),
        ("🚩", str(flagged),          "Flagged Mismatches", _DANGER),
        ("📊", f"{rate:.1%}",         "Flag Rate",          _ACCENT2),
        ("🔴", str(hidden),           "Hidden Crises",      _DANGER),
        ("🟡", str(false_a),          "False Alarms",       _WARNING),
        ("🎯", f"{avg_conf:.1%}",     "Avg. Confidence",    _SUCCESS),
    ]
    for col, (icon, val, lbl, color) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-pill fade-in">
                <div class="kpi-value" style="color:{color}">{icon} {val}</div>
                <div class="kpi-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Charts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def chart_mismatch_distribution(result: pd.DataFrame):
    """Donut chart: Consistent vs Mismatch."""
    counts = (
        result["predicted_mismatch"]
        .map({0: "Consistent", 1: "Mismatch"})
        .value_counts()
        .reset_index()
    )
    counts.columns = ["Judgment", "Count"]
    fig = go.Figure(go.Pie(
        labels=counts["Judgment"],
        values=counts["Count"],
        hole=0.55,
        marker=dict(colors=[_SUCCESS, _DANGER],
                    line=dict(color=_BG, width=3)),
        textfont=dict(size=14, color="white"),
        hoverinfo="label+value+percent",
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text="Mismatch Distribution", font=dict(size=16)),
        showlegend=True,
        legend=dict(font=dict(size=12)),
        height=360,
    )
    return fig


def chart_mismatch_types(result: pd.DataFrame):
    """Bar chart of mismatch types for flagged tickets only."""
    flagged = result[result["predicted_mismatch"] == 1]
    if flagged.empty:
        return None
    counts = flagged["mismatch_type"].value_counts().reset_index()
    counts.columns = ["Mismatch Type", "Count"]

    color_map = {"Hidden Crisis": _DANGER, "False Alarm": _WARNING, "Consistent": _SUCCESS}
    colors = [color_map.get(t, _ACCENT) for t in counts["Mismatch Type"]]

    fig = go.Figure(go.Bar(
        x=counts["Mismatch Type"],
        y=counts["Count"],
        marker=dict(
            color=colors,
            line=dict(width=0),
            cornerradius=6,
        ),
        text=counts["Count"],
        textposition="outside",
        textfont=dict(color=_TEXT, size=14, family="Inter"),
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text="Mismatch Types Breakdown", font=dict(size=16)),
        yaxis_title="Count",
        height=360,
        bargap=0.35,
    )
    return fig


def chart_top_signals(result: pd.DataFrame):
    """Horizontal bar chart of average signal severity values."""
    signal_map = {
        "sig_a_severity":   "🧠 SBERT Semantic Anchors",
        "sig_b_rt_proxy":   "⏱️ Resolution-Time Proxy",
        "sig_c_rules":      "📝 Rule-Based NLP",
        "sig_d_cluster":    "🔗 Embedding Clusters",
    }
    signal_cols = list(signal_map.keys())
    avgs = result[signal_cols].mean()
    df_sig = pd.DataFrame({
        "Signal": [signal_map[c] for c in signal_cols],
        "Avg Severity": avgs.values,
    }).sort_values("Avg Severity", ascending=True)

    fig = go.Figure(go.Bar(
        y=df_sig["Signal"],
        x=df_sig["Avg Severity"],
        orientation="h",
        marker=dict(
            color=["#6366F1", "#8B5CF6", "#A78BFA", "#C4B5FD"],
            cornerradius=6,
        ),
        text=[f"{v:.2f}" for v in df_sig["Avg Severity"]],
        textposition="outside",
        textfont=dict(color=_TEXT, size=13, family="Inter"),
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text="Top Contributing Signals", font=dict(size=16)),
        xaxis_title="Average Severity (0-3 ordinal)",
        height=340,
        bargap=0.30,
    )
    return fig


def chart_severity_heatmap(result: pd.DataFrame):
    """Heatmap: average severity_delta by Issue_Category × Ticket_Channel."""
    heat = result.pivot_table(
        values="severity_delta",
        index="Issue_Category",
        columns="Ticket_Channel",
        aggfunc="mean",
        fill_value=0,
    )
    fig = go.Figure(go.Heatmap(
        z=heat.values,
        x=heat.columns.tolist(),
        y=heat.index.tolist(),
        colorscale=[
            [0.0, "#3B82F6"],   # negative delta = blue
            [0.5, "#1E293B"],   # zero = dark
            [1.0, "#EF4444"],   # positive delta = red
        ],
        zmid=0,
        text=np.round(heat.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=13, color="white"),
        hovertemplate=(
            "<b>Category:</b> %{y}<br>"
            "<b>Channel:</b> %{x}<br>"
            "<b>Avg Δ:</b> %{z:.2f}<extra></extra>"
        ),
        colorbar=dict(
            title="Avg Δ",
            tickfont=dict(color=_TEXT),
            titlefont=dict(color=_TEXT),
        ),
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text="Severity Delta Heatmap · Category × Channel",
                   font=dict(size=16)),
        xaxis_title="Ticket Channel",
        yaxis_title="Issue Category",
        height=max(340, 60 * len(heat.index)),
        yaxis=dict(autorange="reversed", gridcolor="#1E293B"),
    )
    return fig


def chart_priority_flow(result: pd.DataFrame):
    """Sankey diagram: Assigned priority → Inferred severity flow."""
    flagged = result[result["predicted_mismatch"] == 1]
    if flagged.empty:
        return None

    levels = ["Low", "Medium", "High", "Critical"]
    src_labels = [f"Assigned: {l}" for l in levels]
    tgt_labels = [f"Inferred: {l}" for l in levels]
    all_labels = src_labels + tgt_labels

    source, target, value = [], [], []
    for i, al in enumerate(levels):
        for j, il in enumerate(levels):
            cnt = int(((flagged["assigned_priority"] == al) &
                       (flagged["inferred_severity"] == il)).sum())
            if cnt > 0:
                source.append(i)
                target.append(len(levels) + j)
                value.append(cnt)

    if not value:
        return None

    node_colors = ["#6366F1", "#8B5CF6", "#F59E0B", "#EF4444",
                   "#6366F1", "#8B5CF6", "#F59E0B", "#EF4444"]

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=20,
            thickness=28,
            line=dict(color=_BG, width=1),
            label=all_labels,
            color=node_colors,
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color="rgba(99,102,241,0.25)",
        ),
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text="Priority Re-Classification Flow (Flagged Tickets)",
                   font=dict(size=16)),
        height=400,
    )
    return fig


def chart_confidence_distribution(result: pd.DataFrame):
    """Histogram of mismatch_probability for all tickets."""
    fig = go.Figure(go.Histogram(
        x=result["mismatch_probability"],
        nbinsx=30,
        marker=dict(
            color=_ACCENT,
            line=dict(color=_ACCENT2, width=1),
            opacity=0.80,
        ),
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text="Mismatch Confidence Distribution", font=dict(size=16)),
        xaxis_title="Mismatch Probability",
        yaxis_title="Ticket Count",
        height=320,
        bargap=0.06,
    )
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Evidence Dossier renderer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_dossier(dossier: dict):
    """Render a single evidence dossier as rich HTML inside an expander."""
    mt = dossier.get("mismatch_type", "")
    badge = _badge_class(mt)
    delta = dossier.get("severity_delta", "0")
    conf  = dossier.get("confidence", 0)

    st.markdown(f"""
    <div class="dossier-box fade-in">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
            <span class="dossier-title">🔍 {dossier['ticket_id']}</span>
            <span class="badge-mismatch {badge}">{mt}</span>
        </div>
        <div style="display:grid; grid-template-columns: repeat(4,1fr); gap:12px; margin-bottom:14px;">
            <div>
                <div style="color:{_TEXT_DIM}; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em;">Assigned</div>
                <div style="font-weight:700; font-size:1.05rem;">{dossier.get("assigned_priority","—")}</div>
            </div>
            <div>
                <div style="color:{_TEXT_DIM}; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em;">Inferred</div>
                <div style="font-weight:700; font-size:1.05rem;">{dossier.get("inferred_severity","—")}</div>
            </div>
            <div>
                <div style="color:{_TEXT_DIM}; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em;">Delta</div>
                <div style="font-weight:700; font-size:1.05rem; color:{"#F87171" if str(delta).startswith("+") else "#FBBF24"}">{delta}</div>
            </div>
            <div>
                <div style="color:{_TEXT_DIM}; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em;">Confidence</div>
                <div style="font-weight:700; font-size:1.05rem; color:{_SUCCESS}">{conf:.1%}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Evidence details
    evidence = dossier.get("feature_evidence", [])
    if evidence:
        ev_html = "".join(
            f'<span class="evidence-tag">{e["signal"]}: {e.get("value","")}</span>'
            for e in evidence
        )
        st.markdown(f'<div style="margin:-8px 0 8px 0;">{ev_html}</div>', unsafe_allow_html=True)

    analysis = dossier.get("constraint_analysis", "")
    if analysis:
        st.markdown(f"> {analysis}")


def render_dossiers_section(dossiers: list):
    """Render the full dossier section with count header."""
    if not dossiers:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:40px;">
            <div style="font-size:2rem; margin-bottom:8px;">✅</div>
            <div style="font-size:1.1rem; font-weight:600;">No Priority Mismatches Detected</div>
            <div style="color:#94A3B8; margin-top:6px;">All tickets appear to be correctly prioritised.</div>
        </div>""", unsafe_allow_html=True)
        return

    st.markdown(f"""
    <div class="section-header fade-in">
        <span class="icon">📋</span> Evidence Dossiers
        <span style="background:rgba(239,68,68,0.20); color:#F87171; padding:4px 14px;
              border-radius:999px; font-size:0.82rem; font-weight:700;">
            {len(dossiers)} flagged
        </span>
    </div>""", unsafe_allow_html=True)

    for d in dossiers:
        with st.expander(f"🔎  {d['ticket_id']}  ·  {d['mismatch_type']}  ·  Δ {d['severity_delta']}"):
            render_dossier(d)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Single-ticket form
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def single_ticket_form():
    """Render single-ticket input form and return DataFrame or None."""
    st.markdown("""
    <div class="section-header fade-in">
        <span class="icon">✏️</span> Single Ticket Audit
    </div>""", unsafe_allow_html=True)

    with st.form("single_ticket_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            ticket_id = st.text_input("Ticket ID", "TKT-DEMO-001",
                                      help="Unique ticket identifier")
            subject = st.text_input("Ticket Subject",
                                    "Dashboard not loading data")
            priority = st.selectbox("Assigned Priority",
                                    ["Low", "Medium", "High", "Critical"],
                                    index=1)
            channel = st.selectbox("Ticket Channel",
                                   ["Email", "Chat", "Web Form", "Phone",
                                    "Social Media"], index=1)
        with col2:
            category = st.selectbox("Issue Category",
                                    ["Technical", "Billing", "Account",
                                     "General Inquiry", "Fraud",
                                     "Refund Request"])
            email = st.text_input("Customer Email",
                                  "customer@example.com")
            resolution_time = st.number_input(
                "Resolution Time (hours)",
                min_value=0.0, value=48.0, step=1.0)
            product = st.text_input("Product Purchased",
                                    "Unknown Product",
                                    help="Leave as default if unknown")
        description = st.text_area(
            "Ticket Description",
            "Hi Support, The dashboard is not loading any data, just a "
            "spinning wheel. This is blocking our team from accessing "
            "critical reports.",
            height=130)
        submitted = st.form_submit_button("🛡️  Audit This Ticket")

    if not submitted:
        return None

    return pd.DataFrame([{
        "Ticket_ID":             ticket_id,
        "Customer_Email":        email,
        "Ticket_Subject":        subject,
        "Ticket_Description":    description,
        "Issue_Category":        category,
        "Priority_Level":        priority,
        "Ticket_Channel":        channel,
        "Resolution_Time_Hours": resolution_time,
        "Product_Purchased":     product,
    }])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Single-ticket result display
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_single_result(result: pd.DataFrame, dossiers: list):
    row = result.iloc[0]
    is_mismatch = int(row["predicted_mismatch"]) == 1
    mtype = row["mismatch_type"]
    badge = _badge_class(mtype)
    conf = float(row["mismatch_probability"])

    # Big judgment card
    judgment_label = "⚠️ MISMATCH DETECTED" if is_mismatch else "✅ CONSISTENT"
    judgment_color = _DANGER if is_mismatch else _SUCCESS
    st.markdown(f"""
    <div class="glass-card fade-in" style="text-align:center; border-color:{"rgba(239,68,68,0.35)" if is_mismatch else "rgba(16,185,129,0.35)"};">
        <div style="font-size:1.8rem; font-weight:800; color:{judgment_color};
                    margin-bottom:8px;">{judgment_label}</div>
        <span class="badge-mismatch {badge}" style="font-size:1rem; padding:8px 24px;">{mtype}</span>
        <div style="margin-top:14px; color:{_TEXT_DIM};">
            Confidence: <span style="color:{_SUCCESS}; font-weight:700;">{conf:.1%}</span>
        </div>
    </div>""", unsafe_allow_html=True)

    # Detail grid
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Assigned Priority", row["Priority_Level"])
    with c2:
        st.metric("Inferred Severity", row["inferred_severity"])
    with c3:
        st.metric("Severity Delta", int(row["severity_delta"]),
                   delta=f"{'Over' if row['severity_delta'] > 0 else 'Under'}-prioritised"
                         if row["severity_delta"] != 0 else "Aligned")
    with c4:
        st.metric("Resolution Time", f"{row['Resolution_Time_Hours']:.0f}h")

    # Signal breakdown
    st.markdown("""
    <div class="section-header fade-in">
        <span class="icon">📡</span> Signal Breakdown
    </div>""", unsafe_allow_html=True)

    sig_names = {
        "sig_a_severity":  "🧠 Semantic Anchors",
        "sig_b_rt_proxy":  "⏱️ RT Proxy",
        "sig_c_rules":     "📝 Rule NLP",
        "sig_d_cluster":   "🔗 Clusters",
    }
    sig_cols = st.columns(4)
    for col, (key, label) in zip(sig_cols, sig_names.items()):
        val = int(row[key])
        sev = SEVERITY_INV.get(val, f"Ord {val}")
        with col:
            st.metric(label, sev, delta=f"ordinal {val}")

    # Dossier
    render_dossiers_section(dossiers)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Batch dashboard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_batch_dashboard(result: pd.DataFrame, dossiers: list):
    """Full Priority Mismatch Dashboard for batch CSV."""

    # ── KPIs ──
    render_kpi_row(result)

    st.markdown("---")

    # ── Row 1: distribution + types ──
    st.markdown("""
    <div class="section-header fade-in">
        <span class="icon">📊</span> Priority Mismatch Dashboard
    </div>""", unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(chart_mismatch_distribution(result),
                        use_container_width=True, key="dist_chart")
    with r1c2:
        fig = chart_mismatch_types(result)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="types_chart")
        else:
            st.info("No mismatches flagged — all tickets appear consistent.")

    # ── Row 2: signals + confidence ──
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(chart_top_signals(result),
                        use_container_width=True, key="signals_chart")
    with r2c2:
        st.plotly_chart(chart_confidence_distribution(result),
                        use_container_width=True, key="conf_chart")

    st.markdown("---")

    # ── Severity delta heatmap ──
    st.markdown("""
    <div class="section-header fade-in">
        <span class="icon">🗺️</span> Severity Delta Heatmap
    </div>""", unsafe_allow_html=True)
    st.plotly_chart(chart_severity_heatmap(result),
                    use_container_width=True, key="heatmap_chart")

    # ── Sankey flow ──
    sankey_fig = chart_priority_flow(result)
    if sankey_fig:
        st.markdown("""
        <div class="section-header fade-in">
            <span class="icon">🔀</span> Priority Re-Classification Flow
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(sankey_fig, use_container_width=True,
                        key="sankey_chart")

    st.markdown("---")

    # ── Predictions table ──
    st.markdown("""
    <div class="section-header fade-in">
        <span class="icon">📋</span> All Predictions
    </div>""", unsafe_allow_html=True)

    view_cols = [
        "Ticket_ID", "Priority_Level", "Ticket_Channel",
        "Issue_Category", "Resolution_Time_Hours",
        "inferred_severity", "severity_delta",
        "mismatch_type", "predicted_mismatch", "mismatch_probability",
    ]
    st.dataframe(
        result[view_cols].style.map(
            lambda v: f"color: {_DANGER}" if v == "Hidden Crisis"
                      else (f"color: {_WARNING}" if v == "False Alarm"
                            else f"color: {_SUCCESS}" if v == "Consistent"
                            else ""),
            subset=["mismatch_type"],
        ),
        use_container_width=True,
        height=440,
    )

    # ── Downloads ──
    dl1, dl2, _ = st.columns([1, 1, 2])
    with dl1:
        csv_bytes = result[view_cols].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️  Download Predictions CSV",
                           csv_bytes, "sia_predictions.csv", "text/csv")
    with dl2:
        st.download_button("⬇️  Download Dossiers JSON",
                           json.dumps(dossiers, indent=2).encode("utf-8"),
                           "sia_dossiers.json", "application/json")

    # ── Dossiers ──
    render_dossiers_section(dossiers)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    inject_css()

    # ── Hero banner ──
    st.markdown("""
    <div class="hero fade-in">
        <h1>🛡️ Support Integrity Auditor</h1>
        <p>
            Self-supervised priority mismatch detection with grounded evidence dossiers.
            Upload a single ticket or a batch CSV to identify <b>Hidden Crises</b>
            and <b>False Alarms</b> across your support queue.
        </p>
    </div>""", unsafe_allow_html=True)

    # ── Load model artifacts ──
    pseudo, clf_artifacts = load_artifacts(MODEL_DIR)
    if pseudo is None or clf_artifacts is None:
        st.markdown(f"""
        <div class="glass-card" style="border-color:rgba(239,68,68,0.40); text-align:center; padding:40px;">
            <div style="font-size:2rem; margin-bottom:10px;">⚠️</div>
            <div style="font-size:1.15rem; font-weight:700; color:#F87171; margin-bottom:10px;">
                Model Artifacts Not Found
            </div>
            <div style="color:{_TEXT_DIM}; max-width:600px; margin:0 auto;">
                Run the training pipeline first:<br>
                <code style="background:#1E293B; padding:6px 14px; border-radius:8px; margin-top:8px; display:inline-block;">
                python train_pipeline.py --data customer_support_tickets.csv --model-dir models --outputs-dir outputs
                </code>
            </div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    # ── Tabs ──
    tab_single, tab_batch = st.tabs(["✏️  Single Ticket", "📁  Batch CSV Upload"])

    with tab_single:
        df_single = single_ticket_form()
        if df_single is not None:
            with st.spinner("🔍  Running SIA audit pipeline…"):
                result, dossiers = run_sia(df_single, pseudo, clf_artifacts)
            render_single_result(result, dossiers)

    with tab_batch:
        st.markdown("""
        <div class="section-header fade-in">
            <span class="icon">📁</span> Batch CSV Upload
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.20);
                    border-radius:12px; padding:16px 20px; margin-bottom:16px; font-size:0.9rem; color:{_TEXT_DIM};">
            <b style="color:{_TEXT};">Required columns:</b>
            Ticket_Subject · Ticket_Description · Customer_Email · Priority_Level
            · Ticket_Channel · Resolution_Time_Hours · Issue_Category<br>
            <b style="color:{_TEXT};">Optional:</b> Ticket_ID · Product_Purchased
        </div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload your support tickets CSV",
                                    type=["csv"],
                                    label_visibility="collapsed")
        if uploaded is not None:
            df_batch = pd.read_csv(uploaded)
            with st.spinner("🔍  Auditing CSV — this may take a moment for large files…"):
                result, dossiers = run_sia(df_batch, pseudo, clf_artifacts)
            render_batch_dashboard(result, dossiers)

    # ── Footer ──
    st.markdown(f"""
    <div style="text-align:center; padding:40px 0 20px 0; color:{_TEXT_DIM};
                font-size:0.78rem; border-top:1px solid #1E293B; margin-top:40px;">
        <b>Support Integrity Auditor</b> · Self-supervised priority mismatch detection ·
        Built with Streamlit & Plotly<br>
        Powered by SBERT semantic anchors · XGBoost classification · Rule-based NLP
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
