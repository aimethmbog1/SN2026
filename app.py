# =============================================================================
# StockSight — Application Streamlit de Prédiction d'Actions par IA
# =============================================================================
# Requirements: streamlit, yfinance, torch, prophet, neuralprophet,
#               scikit-learn, pandas, numpy, plotly, matplotlib
#
# Lancer avec: streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import warnings

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NEURALPROPHET_DISABLE_CUDA"] = "1"
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSight — AI Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS + Animated AI Background ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');

/* ═══════════════════════════════════════════════
   ANIMATED NEURAL NETWORK BACKGROUND CANVAS
═══════════════════════════════════════════════ */
#ai-bg-canvas {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: 0;
    pointer-events: none;
}

/* ─── Base reset ─── */
html, body {
    margin: 0; padding: 0;
    background: #020818 !important;
    font-family: 'Exo 2', sans-serif;
    color: #e0eaff;
}

/* Make Streamlit containers float above canvas */
[data-testid="stAppViewContainer"] {
    background: transparent !important;
    position: relative;
    z-index: 1;
}
[data-testid="stHeader"] {
    background: rgba(2,8,24,0.85) !important;
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(66,165,245,0.15);
    z-index: 100;
}
[data-testid="stMain"] {
    background: transparent !important;
}
.main .block-container {
    background: transparent !important;
    padding-top: 2rem;
}
section[data-testid="stSidebar"] {
    background: rgba(2, 8, 24, 0.92) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(66,165,245,0.2) !important;
    box-shadow: 4px 0 40px rgba(0,0,0,0.5);
    z-index: 50;
    max-height: 100vh;
    overflow-y: auto;
    overflow-x: hidden;
    padding-right: 6px;
}
section[data-testid="stSidebar"]::-webkit-scrollbar {
    width: 8px;
}
section[data-testid="stSidebar"]::-webkit-scrollbar-track {
    background: rgba(2,8,24,0.5);
}
section[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
    background: rgba(66,165,245,0.4);
    border-radius: 999px;
}
section[data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover {
    background: rgba(66,165,245,0.65);
}
[data-testid="stSidebar"] * { color: #c8deff !important; }
[data-testid="stSidebar"] .stButton > button { color: white !important; }

/* Glassmorphism content panels */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    background: rgba(4, 15, 40, 0.55);
    backdrop-filter: blur(8px);
    border-radius: 16px;
}

/* ─── Tabs ─── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(2,8,24,0.75);
    backdrop-filter: blur(16px);
    border-radius: 14px;
    padding: 6px;
    border: 1px solid rgba(66,165,245,0.2);
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 10px 22px;
    font-family: 'Exo 2', sans-serif;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.5px;
    color: #5a8fbf !important;
    border: none !important;
    transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
    background: transparent !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: #90caf9 !important;
    background: rgba(13,71,161,0.2) !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, #0d47a1 0%, #1565c0 50%, #1e88e5 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 0 24px rgba(21,101,192,0.6), inset 0 1px 0 rgba(255,255,255,0.15);
}

/* ─── Buttons ─── */
.stButton > button {
    background: linear-gradient(135deg, #0a2a5e 0%, #0d47a1 40%, #1976d2 80%, #42a5f5 100%);
    color: white !important;
    border: 1px solid rgba(66,165,245,0.4) !important;
    border-radius: 12px;
    padding: 12px 32px;
    font-family: 'Exo 2', sans-serif;
    font-weight: 700;
    font-size: 14px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    box-shadow: 0 0 30px rgba(13,71,161,0.5), inset 0 1px 0 rgba(255,255,255,0.1);
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    position: relative;
    overflow: hidden;
}
.stButton > button::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
    transition: left 0.5s ease;
}
.stButton > button:hover::before { left: 100%; }
.stButton > button:hover {
    box-shadow: 0 0 50px rgba(66,165,245,0.7), 0 0 100px rgba(66,165,245,0.2);
    transform: translateY(-3px) scale(1.02);
    border-color: rgba(66,165,245,0.8) !important;
}
.stButton > button:active { transform: translateY(-1px); }

/* ─── Model choice buttons ─── */
.model-selection-card {
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
}
.model-selection-card:hover {
    transform: translateY(-3px) scale(1.02);
    border-color: rgba(66,165,245,0.6) !important;
    box-shadow: 0 8px 25px rgba(66,165,245,0.3) !important;
}
.model-selection-card.active:hover {
    border-color: rgba(255,255,255,0.9) !important;
    box-shadow: 0 8px 30px rgba(66,165,245,0.6) !important;
}

/* ─── Metric cards ─── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(13,71,161,0.22) 0%, rgba(5,20,50,0.7) 100%);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(66,165,245,0.25);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35),
                inset 0 1px 0 rgba(255,255,255,0.06),
                0 0 0 1px rgba(66,165,245,0.05);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(13,71,161,0.4), 0 0 30px rgba(66,165,245,0.15);
}
[data-testid="metric-container"] label {
    color: #5a8fbf !important;
    font-size: 11px !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-family: 'Share Tech Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #42a5f5 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 20px !important;
    font-weight: 700;
    text-shadow: 0 0 20px rgba(66,165,245,0.5);
}
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* ─── Inputs ─── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: rgba(4,14,38,0.85) !important;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(66,165,245,0.25) !important;
    border-radius: 10px !important;
    color: #c8deff !important;
    transition: border-color 0.3s;
}
.stSelectbox > div > div:focus-within,
.stMultiSelect > div > div:focus-within {
    border-color: rgba(66,165,245,0.6) !important;
    box-shadow: 0 0 15px rgba(66,165,245,0.2) !important;
}
.stSlider [data-baseweb="slider"] { padding: 0 4px; }
.stSlider [data-baseweb="thumb"] {
    background: linear-gradient(135deg, #1565c0, #42a5f5) !important;
    box-shadow: 0 0 12px rgba(66,165,245,0.6) !important;
}
.stSlider [data-baseweb="track-fill"] {
    background: linear-gradient(90deg, #0d47a1, #42a5f5) !important;
}
.stRadio > div label { color: #8aabcc !important; }
.stRadio > div [aria-checked="true"] { color: #42a5f5 !important; }

/* ─── Expander ─── */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
    background: rgba(13,71,161,0.18) !important;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(66,165,245,0.2) !important;
    border-radius: 10px !important;
    color: #7aabd4 !important;
    font-weight: 600;
    font-family: 'Exo 2', sans-serif;
    transition: all 0.3s;
}
.streamlit-expanderHeader:hover { background: rgba(13,71,161,0.3) !important; }

/* ─── Dataframe ─── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
[data-testid="stDataFrame"] thead th {
    background: rgba(13,71,161,0.4) !important;
    color: #42a5f5 !important;
    font-family: 'Exo 2', sans-serif;
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* ═══════════════════════════════════════════════
   CUSTOM COMPONENTS
═══════════════════════════════════════════════ */

/* Hero section */
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 56px;
    font-weight: 900;
    background: linear-gradient(90deg, #0d47a1, #42a5f5, #7c4dff, #00b4d8, #42a5f5, #0d47a1);
    background-size: 400% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 5s linear infinite;
    text-align: center;
    line-height: 1.05;
    margin-bottom: 6px;
    text-shadow: none;
    filter: drop-shadow(0 0 30px rgba(66,165,245,0.4));
}
.hero-subtitle {
    color: #5a8fbf;
    text-align: center;
    font-size: 13px;
    font-weight: 400;
    letter-spacing: 5px;
    text-transform: uppercase;
    margin-bottom: 8px;
    font-family: 'Share Tech Mono', monospace;
}
.hero-tagline {
    text-align: center;
    color: #7aabd4;
    font-size: 15px;
    line-height: 1.7;
    max-width: 680px;
    margin: 0 auto 36px;
}

/* Feature cards */
.hero-card {
    background: linear-gradient(145deg,
        rgba(13,71,161,0.28) 0%,
        rgba(5,20,60,0.65) 50%,
        rgba(13,71,161,0.15) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(66,165,245,0.22);
    border-radius: 20px;
    padding: 32px 24px;
    text-align: center;
    box-shadow: 0 12px 40px rgba(0,0,0,0.45),
                0 0 0 1px rgba(66,165,245,0.08),
                inset 0 1px 0 rgba(255,255,255,0.06);
    transition: transform 0.4s cubic-bezier(0.4,0,0.2,1),
                box-shadow 0.4s cubic-bezier(0.4,0,0.2,1),
                border-color 0.4s;
    height: 100%;
    position: relative;
    overflow: hidden;
}
.hero-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(66,165,245,0.5), transparent);
}
.hero-card::after {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 60% 0%, rgba(66,165,245,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-card:hover {
    transform: translateY(-8px) scale(1.01);
    box-shadow: 0 24px 60px rgba(13,71,161,0.5),
                0 0 50px rgba(66,165,245,0.18),
                0 0 0 1px rgba(66,165,245,0.25);
    border-color: rgba(66,165,245,0.45);
}
.hero-card .icon {
    font-size: 42px;
    margin-bottom: 12px;
    display: block;
    filter: drop-shadow(0 0 12px rgba(66,165,245,0.5));
}
.hero-card h3 {
    color: #42a5f5;
    font-family: 'Orbitron', monospace;
    font-size: 14px;
    letter-spacing: 1px;
    margin: 0 0 10px;
    text-shadow: 0 0 20px rgba(66,165,245,0.4);
}
.hero-card p { color: #6a90b4; font-size: 13px; line-height: 1.6; margin: 0; }

/* Ticker cards */
.ticker-card {
    background: linear-gradient(145deg,
        rgba(4,14,38,0.85) 0%,
        rgba(13,71,161,0.18) 100%);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(66,165,245,0.2);
    border-radius: 16px;
    padding: 18px 14px;
    text-align: center;
    margin-bottom: 8px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3),
                inset 0 1px 0 rgba(255,255,255,0.04);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    position: relative;
    overflow: hidden;
}
.ticker-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent, #42a5f5), transparent);
}
.ticker-card:hover { transform: translateY(-3px); box-shadow: 0 16px 48px rgba(0,0,0,0.4); }
.ticker-card .ticker-name {
    font-family: 'Orbitron', monospace;
    color: #42a5f5;
    font-size: 17px;
    font-weight: 700;
    text-shadow: 0 0 16px rgba(66,165,245,0.4);
}
.ticker-card .ticker-price { color: #e0eaff; font-size: 21px; font-weight: 600; margin: 6px 0 4px; }
.ticker-card .ticker-delta-pos { color: #00e676; font-size: 13px; font-family: 'Share Tech Mono', monospace; }
.ticker-card .ticker-delta-neg { color: #ff5252; font-size: 13px; font-family: 'Share Tech Mono', monospace; }

/* Section titles */
.section-title {
    font-family: 'Orbitron', monospace;
    font-size: 22px;
    font-weight: 700;
    background: linear-gradient(90deg, #42a5f5, #7c4dff, #00b4d8, #42a5f5);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
    margin: 24px 0 18px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(66,165,245,0.4), transparent);
    margin-left: 12px;
    display: block;
    -webkit-background-clip: unset;
    -webkit-text-fill-color: unset;
    background-clip: unset;
}

/* Animated shimmer */
@keyframes shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 300% center; }
}

/* Glowing divider */
.glow-divider {
    height: 1px;
    background: linear-gradient(90deg,
        transparent 0%,
        rgba(13,71,161,0.5) 15%,
        rgba(66,165,245,0.8) 50%,
        rgba(13,71,161,0.5) 85%,
        transparent 100%);
    border: none;
    margin: 28px 0;
    box-shadow: 0 0 12px rgba(66,165,245,0.3);
}

/* Info boxes */
.info-box {
    background: linear-gradient(135deg,
        rgba(13,71,161,0.18) 0%,
        rgba(5,20,50,0.6) 100%);
    backdrop-filter: blur(8px);
    border-left: 3px solid #42a5f5;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin: 14px 0;
    font-size: 14px;
    color: #8aabcc;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2),
                inset 4px 0 0 rgba(66,165,245,0.1);
    line-height: 1.6;
}

/* Sidebar ticker badge */
.sb-ticker-card {
    background: linear-gradient(135deg,
        rgba(13,71,161,0.3) 0%,
        rgba(5,20,50,0.75) 100%);
    border: 1px solid rgba(66,165,245,0.25);
    border-radius: 12px;
    padding: 14px 16px;
    margin: 8px 0;
    position: relative;
    overflow: hidden;
}
.sb-ticker-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #0d47a1, #42a5f5, #7c4dff);
}

/* Footer */
.footer {
    text-align: center;
    padding: 28px;
    margin-top: 48px;
    border-top: 1px solid rgba(66,165,245,0.12);
    color: #2d5a8e;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-family: 'Share Tech Mono', monospace;
}
.footer span { color: #42a5f5; }

/* Scanning line effect on main area */
[data-testid="stMain"]::after {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg,
        transparent,
        rgba(66,165,245,0.6),
        rgba(124,77,255,0.4),
        transparent);
    animation: scanline 8s linear infinite;
    z-index: 9999;
    pointer-events: none;
}
@keyframes scanline {
    0%   { top: 0%; opacity: 1; }
    80%  { opacity: 0.6; }
    100% { top: 100%; opacity: 0; }
}

/* Floating data numbers decoration */
.data-stream {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: rgba(66,165,245,0.35);
    letter-spacing: 1px;
    line-height: 2;
    text-align: right;
    pointer-events: none;
    animation: fadeInNumbers 2s ease-in;
}
@keyframes fadeInNumbers { from {opacity:0} to {opacity:1} }

/* Pulse dot indicator */
.pulse-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #00e676;
    box-shadow: 0 0 0 0 rgba(0,230,118,0.4);
    animation: pulse 2s infinite;
    margin-right: 6px;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(0,230,118,0.4); }
    70%  { box-shadow: 0 0 0 8px rgba(0,230,118,0); }
    100% { box-shadow: 0 0 0 0 rgba(0,230,118,0); }
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: rgba(2,8,24,0.5); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #0d47a1, #42a5f5);
    border-radius: 3px;
    box-shadow: 0 0 8px rgba(66,165,245,0.4);
}

/* Warning / error / info boxes */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    backdrop-filter: blur(8px);
    border-left-width: 4px !important;
}
</style>

<!-- ═══ BLUEPRINT GRID BACKGROUND CANVAS ═══ -->
<canvas id="ai-bg-canvas"></canvas>
<script>
(function () {
  const canvas = document.getElementById('ai-bg-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H, t = 0;

  /* ── Floating dust particles ── */
  let dust = [];
  function initDust() {
    dust = Array.from({ length: 55 }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      r: 0.5 + Math.random() * 1.6,
      vx: (Math.random() - 0.5) * 0.18,
      vy: -(0.08 + Math.random() * 0.22),
      alpha: 0.04 + Math.random() * 0.18,
      phase: Math.random() * Math.PI * 2,
    }));
  }

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
    initDust();
  }

  function draw() {
    t++;
    ctx.clearRect(0, 0, W, H);

    /* ── 1. Base background: deep navy blue ── */
    const bg = ctx.createRadialGradient(W * 0.5, H * 0.48, 0, W * 0.5, H * 0.48, Math.max(W, H) * 0.75);
    bg.addColorStop(0,   '#0d2a45');   /* centre légèrement plus clair */
    bg.addColorStop(0.45,'#0a2038');
    bg.addColorStop(0.75,'#071828');
    bg.addColorStop(1,   '#050f1c');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    /* ── 2. Very subtle central glow (warm dark-blue) ── */
    const glow = ctx.createRadialGradient(W*0.5, H*0.5, 0, W*0.5, H*0.5, Math.min(W,H)*0.55);
    glow.addColorStop(0,   'rgba(18,55,90,0.35)');
    glow.addColorStop(0.6, 'rgba(10,32,58,0.12)');
    glow.addColorStop(1,   'rgba(0,0,0,0)');
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, W, H);

    /* ── 3. Blueprint grid ── */
    const CELL  = 36;           /* main cell size in px  */
    const CELL5 = CELL * 5;     /* every 5th line = accent */

    /* scroll offset — very slow drift */
    const ox = (t * 0.08) % CELL;
    const oy = (t * 0.06) % CELL;

    /* minor grid lines */
    ctx.save();
    ctx.strokeStyle = 'rgba(80,160,220,0.10)';
    ctx.lineWidth   = 0.5;
    for (let x = -CELL + ox; x < W + CELL; x += CELL) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let y = -CELL + oy; y < H + CELL; y += CELL) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    /* major accent lines (every 5 cells) */
    ctx.strokeStyle = 'rgba(100,185,240,0.20)';
    ctx.lineWidth   = 0.8;
    const ox5 = (t * 0.08) % CELL5;
    const oy5 = (t * 0.06) % CELL5;
    for (let x = -CELL5 + ox5; x < W + CELL5; x += CELL5) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let y = -CELL5 + oy5; y < H + CELL5; y += CELL5) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }
    ctx.restore();

    /* ── 4. Cross hair dots at major intersections ── */
    ctx.save();
    for (let x = -CELL5 + ox5; x < W + CELL5; x += CELL5) {
      for (let y = -CELL5 + oy5; y < H + CELL5; y += CELL5) {
        ctx.strokeStyle = 'rgba(120,200,255,0.30)';
        ctx.lineWidth   = 0.7;
        const cs = 5;   /* crosshair arm length */
        ctx.beginPath(); ctx.moveTo(x - cs, y); ctx.lineTo(x + cs, y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x, y - cs); ctx.lineTo(x, y + cs); ctx.stroke();
      }
    }
    ctx.restore();

    /* ── 5. Subtle vignette overlay (edges darker) ── */
    const vig = ctx.createRadialGradient(W/2, H/2, Math.min(W,H)*0.3, W/2, H/2, Math.max(W,H)*0.75);
    vig.addColorStop(0,   'rgba(0,0,0,0)');
    vig.addColorStop(0.65,'rgba(0,0,0,0.08)');
    vig.addColorStop(1,   'rgba(0,0,0,0.52)');
    ctx.fillStyle = vig;
    ctx.fillRect(0, 0, W, H);

    /* ── 6. Floating dust motes ── */
    dust.forEach(p => {
      p.x += p.vx; p.y += p.vy; p.phase += 0.012;
      if (p.y < -4)  { p.y = H + 4;  p.x = Math.random() * W; }
      if (p.x < -4)  { p.x = W + 4; }
      if (p.x > W+4) { p.x = -4; }
      const a = p.alpha * (0.5 + 0.5 * Math.sin(p.phase));
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(160,220,255,${a})`;
      ctx.fill();
    });

    requestAnimationFrame(draw);
  }

  resize();
  draw();
  window.addEventListener('resize', resize);
})();
</script>
""", unsafe_allow_html=True)

# ── Company meta ─────────────────────────────────────────────────────────────
# ── Official company logos via Clearbit Logo API & Wikipedia SVGs ───────────
# Uses official brand CDN URLs — real logos, no hand-drawn SVGs
LOGOS = {
    "NVDA": """<img src="data:image/png;base64,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"
        style="width:90px; height:36px; object-fit:contain; filter:drop-shadow(0 0 6px rgba(118,185,0,0.5));">""",

    "ORCL": """<img src="data:image/png;base64,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"
        style="width:90px; height:36px; object-fit:contain; filter:drop-shadow(0 0 6px rgba(248,0,0,0.5));">""",

    "IBM": """<img src="data:image/png;base64,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"
        style="width:80px; height:36px; object-fit:contain; filter:drop-shadow(0 0 6px rgba(31,112,193,0.6));">""",

    "CSCO": """<img src="data:image/png;base64,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"
        style="width:90px; height:36px; object-fit:contain; filter:drop-shadow(0 0 6px rgba(4,159,217,0.5));">""",
}

COMPANIES = {
    "NVIDIA (NVDA)": {
        "ticker": "NVDA",
        "emoji": "🟢",
        "color": "#76b900",
        "full_name": "NVIDIA Corporation",
        "desc": "Leader mondial des GPU et de l'IA",
    },
    "Oracle (ORCL)": {
        "ticker": "ORCL",
        "emoji": "🔴",
        "color": "#f80000",
        "full_name": "Oracle Corporation",
        "desc": "Solutions cloud & bases de données",
    },
    "IBM (IBM)": {
        "ticker": "IBM",
        "emoji": "🔵",
        "color": "#1f70c1",
        "full_name": "International Business Machines",
        "desc": "IA hybride et solutions cloud",
    },
    "Cisco (CSCO)": {
        "ticker": "CSCO",
        "emoji": "🟦",
        "color": "#049fd9",
        "full_name": "Cisco Systems, Inc.",
        "desc": "Réseaux, sécurité et IoT",
    },
}
ALL_TICKERS = ["NVDA", "ORCL", "IBM", "CSCO"]

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(2,8,24,0)",
    plot_bgcolor="rgba(4,21,48,0.4)",
    font=dict(family="Exo 2, sans-serif", color="#c8deff"),
    xaxis=dict(gridcolor="#0d2a4d", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#0d2a4d", showgrid=True, zeroline=False),
)

# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def download_all_data():
    data = yf.download(ALL_TICKERS, start="2015-01-01", end="2023-12-31",
                       progress=False, auto_adjust=True)["Close"]
    data = data[ALL_TICKERS].dropna()
    # st.table(data)
    return data

@st.cache_data(ttl=1800, show_spinner=False)
def get_live_quote(ticker):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="35d")
        if hist.empty:
            return None, None, None
        price = hist["Close"].iloc[-1]
        prev  = hist["Close"].iloc[-2]
        chg   = (price - prev) / prev * 100
        spark = hist["Close"].tail(30)
        return round(price, 2), round(chg, 2), spark
    except:
        return None, None, None

def make_sparkline(series, color_pos=True):
    color = "#00e676" if (color_pos and series.iloc[-1] >= series.iloc[0]) else "#ff5252"
    fig = go.Figure(go.Scatter(
        x=list(range(len(series))), y=series.values,
        mode="lines", line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=color.replace(")", ",0.1)").replace("rgb", "rgba") if color.startswith("rgb") else
                   ("rgba(0,230,118,0.08)" if "#00e676" in color else "rgba(255,82,82,0.08)"),
    ))
    fig.update_layout(
        height=70, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig

# ─── LSTM ────────────────────────────────────────────────────────────────────
# ─── LSTM : chargement du modèle pré-entraîné ────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_lstm_model(ticker):
    """Charge le modèle LSTM pré-entraîné depuis le dossier models/."""
    import torch
    import torch.nn as nn

    SEQ_LEN = 60
    HIDDEN  = 64
    LAYERS  = 2

    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=4, hidden_size=HIDDEN,
                                num_layers=LAYERS, batch_first=True)
            self.fc   = nn.Linear(HIDDEN, 1)
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            return self.fc(out[:, -1, :])

    model_path = f"models/lstm_model_{ticker}.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modèle LSTM introuvable : {model_path}\n"
            "Assurez-vous que le dossier 'models/' est présent dans le répertoire de l'application."
        )

    model = LSTMModel()
    try:
        state_dict = torch.load(model_path, map_location="cpu")
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
        else:
            model = state_dict
    except:
        model = torch.jit.load(model_path, map_location="cpu")

    if hasattr(model, "eval"):
        model.eval()

    return model, SEQ_LEN


def predict_lstm(data, ticker, n_days):
    """Prédit avec le modèle LSTM pré-entraîné (pas d'entraînement)."""
    import torch
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    print("before loading model")

    model, SEQ_LEN = load_lstm_model(ticker)
    
    print("after loading model")

    arr = data[ticker].values.astype(np.float32).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(arr)
    print("after normalisation")
    # Construction des séquences
    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i - SEQ_LEN:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    print("after sequences contruction")
    split = int(len(X) * 0.8)
    X_test = torch.tensor(X[split:]).float()
    y_test = torch.tensor(y[split:]).float()

    idx = ALL_TICKERS.index(ticker)
    print("after split data")
    with torch.no_grad(): # on désactive le calcul de gradiants pendant cette opération
        test_pred = model(X_test).cpu().numpy()
    test_pred_real = scaler.inverse_transform(test_pred).flatten()
    test_real      = scaler.inverse_transform(y_test.cpu().numpy()).flatten()
    test_dates     = data.index[SEQ_LEN + split:]
    print("after inverse transforme value")
    # Prédiction future (boucle auto-régressive)
    last_seq = scaled[-SEQ_LEN:].copy()
    future_preds = []
    with torch.no_grad():
        for _ in range(n_days):
            inp  = torch.tensor(last_seq[np.newaxis]).float()
            pred = model(inp).cpu().numpy()[0]
            future_preds.append(pred)
            last_seq = np.vstack([last_seq[1:], pred])

    future_arr   = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
    future_dates = pd.bdate_range(start=data.index[-1] + pd.Timedelta(days=1), periods=n_days)

    rmse = np.sqrt(mean_squared_error(test_real, test_pred_real))
    mae  = mean_absolute_error(test_real, test_pred_real)
    mape = np.mean(np.abs((test_real - test_pred_real) / (test_real + 1e-9))) * 100
    print("at the end")
    return {
        "test_dates":   test_dates,
        "test_real":    test_real,
        "test_pred":    test_pred_real,
        "future_dates": future_dates,
        "future_pred":  future_arr,
        "rmse": rmse, "mae": mae, "mape": mape,
    }


# ─── Prophet : chargement du modèle pré-entraîné ─────────────────────────────
@st.cache_resource(show_spinner=False)
def load_prophet_model(ticker):
    """Charge le modèle Prophet pré-entraîné depuis le dossier models/."""
    import joblib
    # Les noms de fichiers utilisent le ticker en minuscules
    model_path = f"models/prophet_model_{ticker}.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modèle Prophet introuvable : {model_path}\n"
            "Assurez-vous que le dossier 'models/' est présent dans le répertoire de l'application."
        )
    model = joblib.load(model_path)
    return model


def predict_prophet(data, ticker, n_days):
    """Prédit avec le modèle Prophet pré-entraîné (pas d'entraînement)."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    m = load_prophet_model(ticker)

    series = data[[ticker]].reset_index()
    series.columns = ["ds", "y"]
    series["ds"] = pd.to_datetime(series["ds"]).dt.tz_localize(None)

    split = int(len(series) * 0.8)
    test_df = series.iloc[split:]

    # Prédiction sur le jeu de test
    test_future = m.make_future_dataframe(periods=len(test_df), freq="B")
    test_fc     = m.predict(test_future)
    test_pred   = test_fc.set_index("ds").reindex(test_df["ds"])["yhat"].values
    test_real   = test_df["y"].values

    # Prédiction future
    future_df = m.make_future_dataframe(periods=n_days, freq="B")
    forecast  = m.predict(future_df)
    fut_mask  = forecast["ds"] > series["ds"].max()
    fut_fc    = forecast[fut_mask]

    rmse = np.sqrt(mean_squared_error(test_real, test_pred))
    mae  = mean_absolute_error(test_real, test_pred)
    mape = np.mean(np.abs((test_real - test_pred) / (test_real + 1e-9))) * 100

    return {
        "test_dates":   test_df["ds"].values,
        "test_real":    test_real,
        "test_pred":    test_pred,
        "future_dates": fut_fc["ds"].values,
        "future_pred":  fut_fc["yhat"].values,
        "future_lower": fut_fc["yhat_lower"].values,
        "future_upper": fut_fc["yhat_upper"].values,
        "full_fc": forecast,
        "series": series,
        "rmse": rmse, "mae": mae, "mape": mape,
    }


# ─── NeuralProphet : chargement du modèle pré-entraîné ───────────────────────
@st.cache_resource(show_spinner=False)
def load_neuralprophet_model(ticker):
    """Charge le modèle NeuralProphet pré-entraîné depuis le dossier models/."""
    import torch
    # Les noms de fichiers utilisent le ticker en majuscules (ex: neuralprophet_NVDA.pkl)
    model_path = f"models/neuralprophet_{ticker}.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modèle NeuralProphet introuvable : {model_path}\n"
            "Assurez-vous que le dossier 'models/' est présent dans le répertoire de l'application."
        )
    model = torch.load(model_path, map_location=torch.device('cpu'),weights_only=False)
    # 2. Désactiver explicitement le Trainer (souvent la cause de l'erreur)
    if hasattr(model, 'trainer'):
        model.trainer = None 
        
    # 3. Forcer le flag device
    model.device = torch.device('cpu')
    return model

def predict_neuralprophet(data, ticker, n_days):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np
    import pandas as pd

    m = load_neuralprophet_model(ticker)
     # ✅ Forcer CPU sur le modèle NeuralProphet
    m.restore_trainer() # Réinitialise le trainer proprement sur l'appareil actuel (CPU)

    series = data[[ticker]].copy().reset_index()
    series.columns = ["ds", "y"]
    series["ds"] = pd.to_datetime(series["ds"]).dt.tz_localize(None)
    series = series.dropna(subset=["y"])
    last_hist_date = series["ds"].max()

    print(f"✅ Historique: {series['ds'].min()} → {last_hist_date} ({len(series)} pts)")

    # --- PARTIE A : TEST ---
    future_hist   = m.make_future_dataframe(series, periods=0, n_historic_predictions=True)
    forecast_test = m.predict(future_hist)
    test_results  = forecast_test.dropna(subset=['yhat1'])
    test_real     = test_results['y'].values
    test_pred     = test_results['yhat1'].values
    test_dates    = test_results['ds'].values

    # --- PARTIE B : FUTUR ---
    fut_df       = m.make_future_dataframe(series, periods=n_days, n_historic_predictions=False)
    forecast_fut = m.predict(fut_df)
    forecast_fut['ds'] = pd.to_datetime(forecast_fut['ds']).dt.tz_localize(None)

    yhat_cols = sorted(
        [c for c in forecast_fut.columns if str(c).startswith('yhat')],
        key=lambda x: int(x.replace('yhat', ''))
    )

    fut_only = forecast_fut[forecast_fut['ds'] > last_hist_date].copy()

    # ✅ CE BLOC REMPLACE tout ce qui venait après fut_only dans la version précédente
    if len(fut_only) == 0:
        anchor_row = forecast_fut.iloc[[-1]]
    else:
        anchor_row = fut_only.iloc[[0]]

    n_forecasts_model = len(yhat_cols)
    print(f"n_forecasts du modèle: {n_forecasts_model}")

    fut_prices = anchor_row[yhat_cols].values[0].astype(float)

    if n_days > n_forecasts_model and len(fut_only) > 1:
        all_prices = list(fut_prices)
        step = n_forecasts_model
        for i in range(step, len(fut_only), step):
            if len(all_prices) >= n_days:
                break
            row = fut_only.iloc[[i]]
            prices = row[yhat_cols].values[0].astype(float)
            all_prices.extend(prices)
        fut_prices = np.array(all_prices[:n_days])
    else:
        fut_prices = fut_prices[:n_days]

    fut_dates = pd.bdate_range(
        start=last_hist_date + pd.Timedelta(days=1),
        periods=len(fut_prices)
    )

    min_len    = min(len(fut_dates), len(fut_prices))
    fut_dates  = fut_dates[:min_len]
    fut_prices = fut_prices[:min_len]

    print(f"✅ Prédiction finale: {fut_dates[0]} → {fut_dates[-1]} ({min_len} pts)")

    rmse = np.sqrt(mean_squared_error(test_real, test_pred))
    mae  = mean_absolute_error(test_real, test_pred)
    mape = np.mean(np.abs((test_real - test_pred) / (test_real + 1e-9))) * 100

    return {
        "test_dates":   test_dates,
        "test_real":    test_real,
        "test_pred":    test_pred,
        "future_dates": np.array(fut_dates),
        "future_pred":  fut_prices,
        "rmse": rmse, "mae": mae, "mape": mape,
    }
  
# ─── Chart builder ───────────────────────────────────────────────────────────
def build_prediction_chart(result, ticker, model_name, color_pred, color_fut,
                            show_ci=False):
    print("in build prediction chart")
    all_dates = result["test_dates"]
    all_real  = result["test_real"]
    # print("after asign all_real prediction chart", all_dates)
    # Zoom: last 90 calendar days of history
    cutoff_date = pd.Timestamp(all_dates[-1])
    zoom_start  = cutoff_date - pd.Timedelta(days=90)
    mask = pd.DatetimeIndex(all_dates) >= zoom_start
    print(f"Zoom start: {zoom_start}")
    hist_dates = pd.DatetimeIndex(all_dates)[mask]
    hist_real  = all_real[mask]
    hist_pred  = result["test_pred"][mask]
    print("after zoom data prediction chart")
    fut_dates  = pd.DatetimeIndex(result["future_dates"])
    fut_pred   = result["future_pred"]

    fig = go.Figure()
    print("after create fig prediction chart")
    # Real history
    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_real,
        name="Prix réel", mode="lines",
        line=dict(color="#42a5f5", width=2.5),
    ))
    print("after adding real history trace")
    # Model fit
    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_pred,
        name=f"Ajustement {model_name}", mode="lines",
        line=dict(color=color_pred, width=1.8, dash="dash"),
    ))
    print("after adding model fit trace")
    # CI band (Prophet)
    if show_ci and "future_lower" in result and "future_upper" in result:
        fig.add_trace(go.Scatter(
            x=list(fut_dates) + list(fut_dates[::-1]),
            y=list(result["future_upper"]) + list(result["future_lower"][::-1]),
            fill="toself",
            fillcolor="rgba(66,165,245,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Intervalle de confiance",
            showlegend=True,
        ))
    elif not show_ci:
        band = fut_pred * 0.05
        fig.add_trace(go.Scatter(
            x=list(fut_dates) + list(fut_dates[::-1]),
            y=list(fut_pred + band) + list((fut_pred - band)[::-1]),
            fill="toself",
            fillcolor="rgba(0,230,118,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Bande ±5%",
            showlegend=True,
        ))
    print("after adding CI band trace")
    # Future projection
    connect_dates  = [pd.Timestamp(all_dates[-1])] + list(fut_dates)
    connect_prices = [float(all_real[-1])]         + list(fut_pred)

    fig.add_trace(go.Scatter(
        x=connect_dates, y=connect_prices,
        name="Projection future", mode="lines+markers",
        line=dict(color=color_fut, width=2.5, dash="dash"),
        marker=dict(size=5, color=color_fut),
    ))
    print("after adding future projection trace")
    # Vertical cutoff
    x_value = cutoff_date.timestamp() * 1000 

    fig.add_vline(
        x=x_value, 
        line=dict(color="rgba(255,255,255,0.4)", width=1.5, dash="dot"),
        annotation_text="Prédiction",
        annotation_font_color="#7aabd4",
    )
    print("after adding vertical cutoff line")
    n_months = round(len(fut_dates) / 21)
    print(f"n_months: {n_months}")
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(
            text=f"<b>{model_name}</b> — {ticker} | Horizon : {n_months} mois",
            font=dict(family="Orbitron", size=16, color="#42a5f5"),
        ),
        xaxis_title="Date",
        yaxis_title="Prix de clôture ($)",
        legend=dict(bgcolor="rgba(2,8,24,0.6)", bordercolor="#1a3a6e", borderwidth=1),
        height=420,
        hovermode="x unified",
    )
    print("at the end of build prediction chart")
    return fig

def build_compare_chart(results_dict, ticker, n_months):
    model_colors = {"LSTM": "#7c4dff", "Prophet": "#ff9800", "NeuralProphet": "#00e676"}
    fig = go.Figure()

    # Real prices (common)
    first = list(results_dict.values())[0]
    all_dates = pd.DatetimeIndex(first["test_dates"])
    cutoff = all_dates[-1]
    zoom   = cutoff - pd.Timedelta(days=90)
    mask   = all_dates >= zoom
    fig.add_trace(go.Scatter(
        x=all_dates[mask], y=first["test_real"][mask],
        name="Prix réel", mode="lines",
        line=dict(color="#42a5f5", width=2.5),
    ))

    for name, res in results_dict.items():
        col = model_colors.get(name, "#fff")
        fut_dates = pd.DatetimeIndex(res["future_dates"])
        fut_pred  = res["future_pred"]
        connect_d = [cutoff] + list(fut_dates)
        connect_p = [float(res["test_real"][-1])] + list(fut_pred)
        fig.add_trace(go.Scatter(
            x=connect_d, y=connect_p,
            name=name, mode="lines+markers",
            line=dict(color=col, width=2.5, dash="dash"),
            marker=dict(size=4),
        ))

    fig.add_vline(x=str(cutoff.date()), line=dict(color="rgba(255,255,255,0.3)", dash="dot"))
    fig.update_layout(
        **PLOTLY_THEME,
        title=dict(
            text=f"<b>Comparaison des modèles</b> — {ticker} | {n_months} mois",
            font=dict(family="Orbitron", size=16, color="#42a5f5"),
        ),
        xaxis_title="Date", yaxis_title="Prix ($)",
        height=480, hovermode="x unified",
        legend=dict(bgcolor="rgba(2,8,24,0.6)", bordercolor="#1a3a6e", borderwidth=1),
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 24px 0 12px; position:relative;'>
        <div style='font-family: Share Tech Mono, monospace; font-size:9px;
                    color: rgba(66,165,245,0.4); letter-spacing:3px; margin-bottom:8px;'>
            ◈ SYSTEM ONLINE ◈
        </div>
        <div style='font-family: Orbitron, monospace; font-size: 24px; font-weight:900;
                    background: linear-gradient(135deg,#42a5f5,#7c4dff,#00b4d8);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    filter: drop-shadow(0 0 12px rgba(66,165,245,0.4));'>
            📈 StockSight
        </div>
        <div style='color:#2d5a8e; font-size:10px; letter-spacing:3px; margin-top:4px;
                    font-family: Share Tech Mono, monospace;'>
            AI STOCK PREDICTION
        </div>
        <div style='margin-top:10px; display:flex; justify-content:center; align-items:center; gap:6px;'>
            <span class='pulse-dot'></span>
            <span style='font-size:10px; color:#00e676; font-family: Share Tech Mono, monospace;'>
                LIVE DATA
            </span>
        </div>
    </div>
    <div style='height:1px; background:linear-gradient(90deg,transparent,rgba(66,165,245,0.4),transparent); margin:8px 0 16px;'></div>
    """, unsafe_allow_html=True)

    st.markdown("**🔍 Sélectionnez une action :**")
    company_choice = st.selectbox("", list(COMPANIES.keys()), label_visibility="collapsed")
    meta = COMPANIES[company_choice]
    ticker_sel = meta["ticker"]

    st.markdown(f"""
    <div class='sb-ticker-card'>
        <div style='display:flex; align-items:center; gap:12px; margin-bottom:8px;'>
            <div style='background:rgba(4,14,38,0.8); border:1px solid rgba(66,165,245,0.2);
                        border-radius:8px; padding:6px 10px; display:inline-flex; align-items:center;'>
                {LOGOS[ticker_sel]}
            </div>
        </div>
        <div style='color:#7aabd4; font-size:12px; margin-bottom:3px;'>{meta['full_name']}</div>
        <div style='color:#2d5a8e; font-size:11px; font-family: Share Tech Mono, monospace;'>{meta['desc']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>**📅 Horizon de prédiction :**", unsafe_allow_html=True)
    n_months = st.slider("", min_value=3, max_value=12, value=3,
                         step=1, label_visibility="collapsed")
    n_days = n_months * 21
    st.markdown(f"""
    <div class='info-box'>
        <b>Horizon sélectionné :</b><br>
        {n_months} mois (~{n_days} jours de bourse)
    </div>
    """, unsafe_allow_html=True)

    if n_months > 9:
        st.warning("⚠️ Horizons longs (>9 mois) réduisent la fiabilité des prédictions.")

    st.markdown("<hr style='border-color:#1a3a6e;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:20px; padding: 12px; background: rgba(13,71,161,0.15);
                border-radius: 8px; border: 1px solid #1a3a6e;'>
        <div style='color:#3d6b9e; font-size:11px; text-align:center;'>
            📅 Données : 2015 – 2023<br>
            🔄 Mise à jour : en temps réel
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠  Accueil",
    "👥  À Propos",
    "📚  Documentation",
    "📈  Prédiction",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — HOME
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("""
    <div style='padding: 40px 20px 20px; position: relative;'>
        <div style='text-align:center; margin-bottom:6px;'>
            <span style='font-family: Share Tech Mono, monospace; font-size:11px;
                         color: rgba(66,165,245,0.5); letter-spacing:4px;'>
                ◈ NEURAL INTELLIGENCE PLATFORM ◈
            </span>
        </div>
        <div class='hero-title'>StockSight</div>
        <div class='hero-subtitle'>AI-POWERED STOCK PRICE PREDICTION</div>
        <div style='display:flex; justify-content:center; gap:32px; margin:12px 0 24px;'>
            <div style='font-family: Share Tech Mono, monospace; font-size:11px; color:rgba(66,165,245,0.5);'>▸ LSTM DEEP LEARNING</div>
            <div style='font-family: Share Tech Mono, monospace; font-size:11px; color:rgba(124,77,255,0.5);'>▸ FACEBOOK PROPHET</div>
            <div style='font-family: Share Tech Mono, monospace; font-size:11px; color:rgba(0,180,216,0.5);'>▸ NEURAL PROPHET</div>
        </div>
        <p class='hero-tagline'>
            StockSight exploite la puissance de trois modèles d'IA de pointe —
            <b style='color:#42a5f5'>LSTM</b>, <b style='color:#ab47bc'>Facebook Prophet</b>
            et <b style='color:#00b4d8'>NeuralProphet</b> — pour analyser et prédire les cours boursiers
            de NVIDIA, Oracle, IBM et Cisco avec une précision de niveau institutionnel.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='hero-card'>
            <div class='icon'>🧠</div>
            <h3>LSTM</h3>
            <p>Réseau de neurones récurrent spécialisé dans l'apprentissage
            de séquences temporelles complexes sur 60 jours de lookback.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='hero-card'>
            <div class='icon'>📊</div>
            <h3>Facebook Prophet</h3>
            <p>Modèle de régression additive bayésienne qui décompose les tendances,
            saisonnalités et effets calendaires automatiquement.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='hero-card'>
            <div class='icon'>🔮</div>
            <h3>Neural Prophet</h3>
            <p>Hybride combinant la décomposition Prophet avec un réseau
            auto-régressif neuronal pour une précision maximale.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)

    # Live market overview
    st.markdown("<div class='section-title'>📡 Aperçu du Marché en Temps Réel</div>",
                unsafe_allow_html=True)

    mcols = st.columns(4)
    ticker_info = [
        ("NVDA", "#76b900", "NVIDIA"),
        ("ORCL", "#f80000", "Oracle"),
        ("IBM",  "#1f70c1", "IBM"),
        ("CSCO", "#049fd9", "Cisco"),
    ]
    for col, (tk, color, name) in zip(mcols, ticker_info):
        with col:
            price, chg, spark = get_live_quote(tk)
            if price:
                delta_class = "ticker-delta-pos" if chg >= 0 else "ticker-delta-neg"
                delta_sign  = "▲" if chg >= 0 else "▼"
                logo_svg = LOGOS[tk]
                st.markdown(f"""
                <div class='ticker-card'>
                    <div style='display:flex; justify-content:center; margin-bottom:8px;
                                padding:8px; background:rgba(4,14,38,0.7);
                                border-radius:8px; border:1px solid rgba(66,165,245,0.12);'>
                        {logo_svg}
                    </div>
                    <div class='ticker-name' style='font-size:14px;'>{tk}</div>
                    <div style='color:#7aabd4; font-size:10px; margin-bottom:4px;'>{name}</div>
                    <div class='ticker-price'>${price:,.2f}</div>
                    <div class='{delta_class}'>{delta_sign} {abs(chg):.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
                if spark is not None and len(spark) > 1:
                    st.plotly_chart(make_sparkline(spark, chg >= 0),
                                    use_container_width=True, config={"displayModeBar": False})
            else:
                st.markdown(f"""
                <div class='ticker-card'>
                    <div style='display:flex; justify-content:center; margin-bottom:8px; padding:8px;
                                background:rgba(4,14,38,0.7); border-radius:8px;'>
                        {LOGOS[tk]}
                    </div>
                    <div class='ticker-name'>{tk}</div>
                    <div style='color:#3d6b9e;'>Données indisponibles</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; margin: 20px 0;'>
        <p style='color:#7aabd4;'>Prêt à analyser les marchés avec l'IA ?</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='footer'>© 2026 <span>StockSight By EFEMBA, HAMAD, KAMNO, MBOG</span> &nbsp;|&nbsp; Projet Académique &nbsp;|&nbsp; <span>Pas de conseil financier</span></div>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-title'>👥 À Propos du Projet</div>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("""
        <div style='background: rgba(13,71,161,0.15); border: 1px solid #1a3a6e;
                    border-radius: 12px; padding: 24px;'>
            <h4 style='color:#42a5f5; font-family: Orbitron, monospace;'>Contexte Académique</h4>
            <p style='color:#8aabcc; line-height:1.8;'>
                Ce projet s'inscrit dans le cadre d'un cours avancé de
                <b style='color:#c8deff'>Machine Learning et Deep Learning</b> appliqué aux séries temporelles financières.
                L'objectif est de comparer différentes architectures de modèles sur leur capacité
                à prédire des cours boursiers réels.
            </p>
            <p style='color:#8aabcc; line-height:1.8;'>
                Nous utilisons des données historiques réelles (2015–2023) des quatre grandes
                entreprises technologiques : <b style='color:#76b900'>NVIDIA</b>,
                <b style='color:#f80000'>Oracle</b>, <b style='color:#1f70c1'>IBM</b>
                et <b style='color:#049fd9'>Cisco</b>.
            </p>
            <p style='color:#8aabcc; line-height:1.8;'>
                L'évaluation s'appuie sur trois métriques : RMSE, MAE et MAPE,
                appliquées sur un ensemble de test représentant 20% des données.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div style='background: rgba(13,71,161,0.15); border: 1px solid #1a3a6e;
                    border-radius: 12px; padding: 24px;'>
            <h4 style='color:#42a5f5; font-family: Orbitron, monospace;'>Stack Technologique</h4>
        """, unsafe_allow_html=True)
        techs = [
            ("🐍", "Python 3.10", "#ffd54f"),
            ("🔥", "PyTorch", "#ef5350"),
            ("📊", "Prophet", "#42a5f5"),
            ("🔮", "NeuralProphet", "#00e676"),
            ("📈", "yfinance", "#7c4dff"),
            ("🌊", "Streamlit", "#ff4b4b"),
            ("🔬", "scikit-learn", "#f57c00"),
            ("📉", "Plotly", "#00b4d8"),
        ]
        for icon, name, color in techs:
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:10px; margin:8px 0;
                        padding: 8px 12px; background: rgba(4,21,48,0.5);
                        border-radius: 6px; border-left: 3px solid {color};'>
                <span>{icon}</span>
                <span style='color:#c8deff; font-size:14px;'>{name}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>⚖️ Comparaison des Modèles</div>", unsafe_allow_html=True)

    comparison_data = {
        "Modèle":       ["LSTM",                "Facebook Prophet",       "NeuralProphet"],
        "Type":         ["Deep Learning",        "Bayésien additif",       "Hybride NN+Prophet"],
        "Points forts": ["Capture non-linéaire","Gère tendances/saisons","Précision + interprétabilité"],
        "Limites":      ["Boîte noire, lent",   "Linéaire par morceaux", "Complexe à tuner"],
        "Idéal pour":   ["Patterns complexes",  "Tendances long terme",  "Séries structurées"],
    }
    df_compare = pd.DataFrame(comparison_data)
    st.dataframe(
        df_compare,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<div class='footer'>© 2026 <span>StockSight By EFEMBA, HAMAD, KAMNO, MBOG</span> &nbsp;|&nbsp; Projet Académique &nbsp;|&nbsp; <span>Pas de conseil financier</span></div>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — DOCUMENTATION
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-title'>📚 Documentation Technique</div>", unsafe_allow_html=True)

    with st.expander("📊 Source de données & Prétraitement", expanded=False):
        st.markdown("""
        **Source :** Yahoo Finance via `yfinance`

        | Paramètre | Valeur |
        |-----------|--------|
        | Tickers | NVDA, ORCL, IBM, CSCO |
        | Période d'entraînement | 2015-01-01 → 2023-12-31 |
        | Normalisation | MinMaxScaler [0, 1] (LSTM) |
        | Fréquence | Jours ouvrés (Business days) |
        | Séparation Train/Test | 80% / 20% |

        Les données sont téléchargées une fois et mises en cache pour optimiser les performances.
        """)

    with st.expander("🧠 Modèle LSTM", expanded=False):
        st.markdown("""
        **Architecture :** 2 couches LSTM empilées

        | Hyperparamètre | Valeur |
        |----------------|--------|
        | `input_size` | 4 (multivarié : NVDA, ORCL, IBM, CSCO) |
        | `hidden_size` | 64 |
        | `num_layers` | 2 |
        | `sequence_length` | 60 jours |
        | `dropout` | 0.2 |
        | Optimiseur | Adam (lr=0.001) |
        | Fonction de perte | MSELoss |
        | Epochs | 2000 (150 en mode rapide Streamlit) |

        **Prédiction future :** boucle auto-régressive — chaque prédiction devient l'entrée suivante.
        """)

    with st.expander("📈 Facebook Prophet", expanded=False):
        st.markdown("""
        **Type :** Modèle de régression additive bayésienne (Stan)

        | Paramètre | Valeur |
        |-----------|--------|
        | `changepoint_prior_scale` | 0.05 |
        | Saisonnalité hebdomadaire | ✅ |
        | Saisonnalité annuelle | ✅ |
        | Sortie | ŷ + intervalle de confiance [yhat_lower, yhat_upper] |

        **Formule :** `y(t) = trend(t) + seasonality(t) + ε`
        """)

    with st.expander("🔮 NeuralProphet", expanded=False):
        st.markdown("""
        **Type :** Hybride réseau neuronal + décomposition Prophet

        | Paramètre | Valeur |
        |-----------|--------|
        | `n_lags` | 60 |
        | `n_forecasts` | horizon sélectionné (3–12 mois) |
        | `ar_reg` | 0.1 |
        | `trend_reg` | 0.1 |
        | Fréquence | Jours ouvrés ('B') |
        | Epochs | 50 |
        """)

    with st.expander("📐 Métriques d'évaluation", expanded=False):
        st.markdown("""
        | Métrique | Formule | Interprétation |
        |----------|---------|----------------|
        | **RMSE** | √(Σ(yᵢ−ŷᵢ)²/n) | Pénalise fortement les grandes erreurs |
        | **MAE** | Σ|yᵢ−ŷᵢ|/n | Écart absolu moyen en dollars |
        | **MAPE** | Σ|yᵢ−ŷᵢ|/yᵢ × 100 | Erreur relative en pourcentage |

        Un MAPE < 5% indique une excellente précision ; > 20% indique un modèle peu fiable sur cet horizon.
        """)

    with st.expander("⚠️ Avertissement", expanded=False):
        st.error("""
        **Cette application est destinée à des fins académiques et éducatives uniquement.**

        - Les prédictions générées ne constituent en aucun cas des conseils financiers.
        - Les performances passées ne garantissent pas les résultats futurs.
        - Les modèles d'IA peuvent être influencés par des événements imprévus (crises, annonces).
        - Ne prenez jamais de décisions d'investissement basées uniquement sur cet outil.
        """)

    st.markdown("<div class='footer'>© 2026 <span>StockSight By EFEMBA, HAMAD, KAMNO, MBOG</span> &nbsp;|&nbsp; Projet Académique &nbsp;|&nbsp; <span>Pas de conseil financier</span></div>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("<div class='section-title'>📈 Module de Prédiction IA</div>", unsafe_allow_html=True)

    # Choix du modèle sous forme de liste
    st.markdown("**🤖 Sélectionnez un modèle :**")
    
    cols = st.columns(4)

    # Définition des modèles avec icônes
    model_info = {
        "LSTM": "🧠 Deep Learning",
        "Facebook Prophet": "📈 Stats (FB)",
        "NeuralProphet": "🚀 Hybrid AI",
        "🔀 Comparer tous": "📊 Benchmark"
    }

    # Création de boutons stylisés
    selected_model = None
    for i, (name, desc) in enumerate(model_info.items()):
        with cols[i]:
            if st.button(f"{name}\n\n{desc}", use_container_width=True, key=f"btn_{i}"):
                st.session_state.model_choice = name

    # On récupère le choix (par défaut LSTM)
    model_choice = st.session_state.get('model_choice', 'LSTM')

    run_btn = st.button("🚀 Lancer la Prédiction", use_container_width=True)

    # Status bar
    company_meta = COMPANIES[company_choice]
    info_cols = st.columns(4)
    with info_cols[0]:
        logo_html = LOGOS.get(ticker_sel, ticker_sel)
    st.metric("Action sélectionnée", ticker_sel)
    with info_cols[1]:
        st.metric("Horizon", f"{n_months} mois")
    with info_cols[2]:
        st.metric("Jours de bourse", f"~{n_days} jours")
    with info_cols[3]:
        model_display = model_choice.replace("🔀 ", "")
        st.metric("Modèle", model_display)

    st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)

    if not run_btn:
        st.markdown("""
        <div style='text-align:center; padding: 60px 20px;
                    background: rgba(13,71,161,0.08); border: 1px dashed #1a3a6e;
                    border-radius: 16px; margin: 20px 0;'>
            <div style='font-size:56px; margin-bottom:16px;'>🚀</div>
            <div style='font-family: Orbitron, monospace; color:#42a5f5; font-size:20px; margin-bottom:8px;'>
                Prêt à analyser
            </div>
            <p style='color:#3d6b9e; max-width:400px; margin:0 auto; font-size:14px;'>
                Configurez vos paramètres dans le panneau gauche et cliquez sur
                <b>"🚀 Lancer la Prédiction"</b> pour démarrer l'analyse IA.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Load data
        with st.spinner("⬇️ Chargement des données historiques..."):
            try:
                data_all = download_all_data()
            except Exception as e:
                st.error(f"Erreur lors du chargement des données : {e}")
                st.stop()

        # Determine which models to run
        compare_all = model_choice == "🔀 Comparer tous"
        models_to_run = (
            ["LSTM", "Prophet", "NeuralProphet"] if compare_all
            else [model_choice.replace("Facebook ", "")]
        )

        results = {}
        progress = st.progress(0, text="Chargement des modèles pré-entraînés...")
        # print(f"data received: {data_all}")
        for i, mdl in enumerate(models_to_run):
            
            progress.progress((i) / len(models_to_run),
                              text=f"⚙️ Chargement et prédiction {mdl} en cours...")
            try:
                if mdl == "LSTM":
                    with st.spinner(f"🧠 Chargement LSTM et génération des prédictions..."):
                        results["LSTM"] = predict_lstm(data_all, ticker_sel, n_days)
                elif mdl in ("Prophet", "Facebook Prophet"):
                    with st.spinner(f"📊 Chargement Facebook Prophet et génération des prédictions..."):
                        results["Prophet"] = predict_prophet(data_all, ticker_sel, n_days)
                elif mdl == "NeuralProphet":
                    with st.spinner(f"🔮 Chargement NeuralProphet et génération des prédictions..."):
                        results["NeuralProphet"] = predict_neuralprophet(data_all, ticker_sel, n_days)
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement / prédiction de {mdl} : {e}")

        progress.progress(1.0, text="✅ Prédictions générées !")
        import time; time.sleep(0.5)
        progress.empty()

        if not results:
            st.error("Aucun modèle n'a pu être entraîné. Vérifiez les paramètres.")
            st.stop()

        # ── [A] Metrics ──────────────────────────────────────────────────────
        st.markdown("### 📊 Métriques de Performance (ensemble de test)")
        for mdl_name, res in results.items():
            model_color = {"LSTM": "#7c4dff", "Prophet": "#ff9800", "NeuralProphet": "#00e676"}.get(mdl_name, "#42a5f5")
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:10px; margin:16px 0 8px;'>
                <div style='width:12px; height:12px; border-radius:50%; background:{model_color};'></div>
                <span style='font-family:Orbitron,monospace; color:{model_color}; font-size:14px;'>{mdl_name}</span>
            </div>
            """, unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("RMSE", f"${res['rmse']:.2f}", delta=f"±{res['rmse']:.2f}$", delta_color="inverse")
            with m2:
                st.metric("MAE",  f"${res['mae']:.2f}",  delta=f"±{res['mae']:.2f}$",  delta_color="inverse")
            with m3:
                st.metric("MAPE", f"{res['mape']:.2f}%", delta=f"{res['mape']:.2f}%", delta_color="inverse")

        st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)

        # ── [E] Compare All ───────────────────────────────────────────────────
        if compare_all and len(results) >= 2:
            st.markdown("### 🔀 Comparaison de tous les modèles")
            fig_compare = build_compare_chart(results, ticker_sel, n_months)
            st.plotly_chart(fig_compare, use_container_width=True)

            # Comparison table
            comp_rows = []
            for mn, r in results.items():
                comp_rows.append({
                    "Modèle": mn,
                    "RMSE ($)": f"{r['rmse']:.2f}",
                    "MAE ($)":  f"{r['mae']:.2f}",
                    "MAPE (%)": f"{r['mape']:.2f}",
                })
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
            st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)

        # ── [B] Individual charts ────────────────────────────────────────────
        MODEL_CFG = {
            "LSTM":          ("#7c4dff", "#00e676", False),
            "Prophet":       ("#ff9800", "#ff9800", True),
            "NeuralProphet": ("#00e676", "#00e676", False),
        }
        # print(results)
        for mdl_name, res in results.items():
            # print(results.items())
            c_pred, c_fut, show_ci = MODEL_CFG.get(mdl_name, ("#42a5f5", "#42a5f5", False))
            st.markdown(f"### {mdl_name} — Prédiction {ticker_sel}")

            fig = build_prediction_chart(res, ticker_sel, mdl_name,
                                         c_pred, c_fut, show_ci)
            st.plotly_chart(fig, use_container_width=True)

            # Insight box
            trend = "hausse" if res["future_pred"][-1] > res["future_pred"][0] else "baisse"
            delta_pct = abs((res["future_pred"][-1] - res["future_pred"][0]) / (res["future_pred"][0]+1e-9) * 100)
            st.markdown(f"""
            <div class='info-box'>
                <b>💡 Analyse {mdl_name} :</b> Le modèle projette une <b>{trend}</b>
                de <b>{delta_pct:.1f}%</b> sur {n_months} mois pour {ticker_sel}.
                Prix projeté à l'horizon : <b>${res['future_pred'][-1]:.2f}</b>
                (contre ${res['future_pred'][0]:.2f} au départ de la projection).
            </div>
            """, unsafe_allow_html=True)

            # ── [C] Full history toggle ───────────────────────────────────────
            with st.expander(f"📜 Afficher l'historique complet 2015–2023 ({mdl_name})"):
                fig_full = go.Figure()
                test_dates = pd.DatetimeIndex(res["test_dates"])
                fig_full.add_trace(go.Scatter(
                    x=test_dates, y=res["test_real"],
                    name="Prix réel", mode="lines",
                    line=dict(color="#42a5f5", width=1.8),
                ))
                fig_full.add_trace(go.Scatter(
                    x=test_dates, y=res["test_pred"],
                    name=f"Prédit ({mdl_name})", mode="lines",
                    line=dict(color=c_pred, width=1.5, dash="dash"),
                ))
                fig_full.update_layout(
                    **PLOTLY_THEME,
                    title=dict(text=f"Historique complet — {ticker_sel} | {mdl_name}",
                               font=dict(family="Orbitron", size=14, color="#42a5f5")),
                    xaxis_title="Date", yaxis_title="Prix ($)",
                    height=360, hovermode="x unified",
                )
                st.plotly_chart(fig_full, use_container_width=True)

            # ── [D] Raw prediction table ──────────────────────────────────────
            with st.expander(f"📋 Tableau des prédictions futures ({mdl_name})"):
                fut_dates_list = pd.DatetimeIndex(res["future_dates"])
                fut_pred_list  = res["future_pred"]

                lower = fut_pred_list * 0.95
                upper = fut_pred_list * 1.05
                if show_ci and "future_lower" in res:
                    lower = res["future_lower"]
                    upper = res["future_upper"]

                df_table = pd.DataFrame({
                    "Date":               [d.strftime("%Y-%m-%d") for d in fut_dates_list],
                    "Prix prédit ($)":    [f"{p:.2f}" for p in fut_pred_list],
                    "Borne inférieure ($)": [f"{p:.2f}" for p in lower],
                    "Borne supérieure ($)": [f"{p:.2f}" for p in upper],
                })
                st.dataframe(df_table, use_container_width=True, hide_index=True)

                csv = df_table.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"⬇️ Télécharger les prédictions {mdl_name} (CSV)",
                    data=csv,
                    file_name=f"predictions_{ticker_sel}_{mdl_name}_{n_months}mois.csv",
                    mime="text/csv",
                )

            st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)

    st.markdown("<div class='footer'>© 2026 <span>StockSight By EFEMBA, HAMAD, KAMNO, MBOG</span> &nbsp;|&nbsp; Projet Académique &nbsp;|&nbsp; <span>Pas de conseil financier</span></div>",
                unsafe_allow_html=True)
