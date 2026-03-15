"""
app.py
======
Streamlit dashboard for the AI-Powered A/B Test Analyzer.

Tells the story of Evergreen Home & Outdoor's Spring Sale banner
personalization experiment through four progressively detailed views:
  1. Executive Summary — the business story and bottom-line recommendation
  2. Detailed Results — full statistical evidence per metric
  3. Who Does It Work For? — segment breakdowns and anomaly detection
  4. AI Summary — Claude-powered executive narrative

Author: Brendan Hoffman
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_PROJECT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from stats_engine import (
    METRIC_REGISTRY,
    format_lift,
    format_value,
    get_data_summary,
    prepare_data,
    run_all_tests,
)
from segment_analyzer import (
    COMPOUND_SEGMENTS,
    SEGMENT_REGISTRY,
    build_summary_payload,
    full_scan_all_metrics,
)
from ai_summary import preview_prompt, PROMPT_TECHNIQUES


# ──────────────────────────────────────────────────────────────
# COLOR PALETTE
# Cerulean for Control A, forest green for Variant B.
# Green/red for outcome direction; warm neutrals for infrastructure.
# ──────────────────────────────────────────────────────────────

COLORS = {
    # Variant identity — cerulean for A (control), forest green for B (variant)
    "control":  "#2B6CB0",   # deep cerulean — stable, established
    "variant":  "#276749",   # forest green — growth, natural

    # Outcome colors — used for lifts and badges
    "positive": "#276749",   # deep forest green
    "negative": "#C0392B",   # terracotta rust

    # Neutral / borderline
    "neutral":  "#7A7268",   # warm gray
    "amber":    "#B7791F",   # aged gold

    # Chart infrastructure (light botanical theme)
    "grid":     "rgba(26,43,34,0.06)",
    "zero":     "rgba(26,43,34,0.20)",
    "text_dim": "rgba(26,43,34,0.50)",
    "text":     "rgba(26,43,34,0.85)",
}


# ──────────────────────────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Evergreen A/B Test Results",
    page_icon="\u2618",  # shamrock
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ─────────────────────────────────────────────
       FONTS  (Playfair Display · JetBrains Mono · Lato)
    ───────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;0,800;1,400;1,600&family=JetBrains+Mono:wght@400;500;600&family=Lato:ital,wght@0,300;0,400;0,600;0,700;1,400&display=swap');

    /* ─────────────────────────────────────────────
       GLOBAL TYPOGRAPHY
    ───────────────────────────────────────────── */
    html, body,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    .stMarkdown {
        font-family: 'Lato', 'Helvetica Neue', sans-serif;
    }
    h1, [data-testid="stMarkdownContainer"] h1 {
        font-family: 'Playfair Display', Georgia, serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
        color: #1A2B22 !important;
    }
    h2, h3,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        font-family: 'Playfair Display', Georgia, serif !important;
        font-weight: 600 !important;
        color: #1A2B22 !important;
    }

    /* ─────────────────────────────────────────────
       SIDEBAR — deep forest green
    ───────────────────────────────────────────── */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        background: #1C2B22 !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #B8D4BE !important;
        font-family: 'Lato', sans-serif;
    }
    /* Color-only on spans — do NOT override font-family or it breaks Material Icons */
    [data-testid="stSidebar"] span:not([class*="material"]):not([data-baseweb]) {
        color: #B8D4BE !important;
    }
    [data-testid="stSidebar"] strong {
        color: #D8EDDA !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.10) !important;
    }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: rgba(184,212,190,0.6) !important;
    }
    /* Preserve Material Icons font — must come before any overrides */
    .material-icons,
    .material-symbols-rounded,
    .material-symbols-outlined,
    [class*="material-symbols"],
    [class*="material-icons"] {
        font-family: 'Material Symbols Rounded', 'Material Icons', 'Material Icons Outlined' !important;
        color: #C8DFC2 !important;
    }

    /* Sidebar expander — header text and SVG arrow */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] p,
    [data-testid="stSidebar"] [data-testid="stExpander"] div {
        color: #C8DFC2 !important;
    }
    /* Color spans but never touch font-family — Material Icons needs its own font */
    [data-testid="stSidebar"] [data-testid="stExpander"] span:not([class*="material"]) {
        color: #C8DFC2 !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] svg,
    [data-testid="stSidebar"] [data-testid="stExpander"] svg path {
        fill: #C8DFC2 !important;
        stroke: #C8DFC2 !important;
    }

    /* ─────────────────────────────────────────────
       TABS
    ───────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid rgba(26,43,34,0.12);
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Lato', sans-serif;
        font-weight: 700;
        font-size: 0.74rem;
        letter-spacing: 0.9px;
        text-transform: uppercase;
        padding: 10px 20px;
        color: rgba(26,43,34,0.45);
        border-bottom: 2px solid transparent;
        margin-bottom: -2px;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: #276749 !important;
        border-bottom: 2px solid #276749 !important;
        background: transparent !important;
    }

    /* ─────────────────────────────────────────────
       METRIC WIDGETS
    ───────────────────────────────────────────── */
    [data-testid="metric-container"] {
        background: #FFFFFF;
        border: 1px solid rgba(26,43,34,0.09);
        border-radius: 8px;
        padding: 16px 18px;
        box-shadow: 0 1px 4px rgba(26,43,34,0.06);
    }
    [data-testid="metric-container"] [data-testid="stMetricLabel"] {
        font-family: 'Lato', sans-serif !important;
        font-size: 0.68rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.9px !important;
        color: rgba(26,43,34,0.45) !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', 'Consolas', monospace !important;
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        color: #1A2B22 !important;
    }

    /* ─────────────────────────────────────────────
       EXPANDERS
    ───────────────────────────────────────────── */
    [data-testid="stExpander"] {
        border: 1px solid rgba(26,43,34,0.10) !important;
        border-radius: 8px !important;
        background: #FFFFFF !important;
        box-shadow: 0 1px 3px rgba(26,43,34,0.04);
        margin-bottom: 8px;
    }
    [data-testid="stExpander"] summary {
        font-family: 'Lato', sans-serif;
        font-weight: 600;
        color: #1A2B22;
        padding: 12px 16px;
    }

    /* ─────────────────────────────────────────────
       BADGE PILLS
    ───────────────────────────────────────────── */
    .badge-green {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        background: rgba(39,103,73,0.10);
        color: #276749;
        font-weight: 700;
        font-size: 0.79em;
        letter-spacing: 0.2px;
        white-space: nowrap;
        border: 1px solid rgba(39,103,73,0.25);
        font-family: 'Lato', sans-serif;
    }
    .badge-red {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        background: rgba(192,57,43,0.08);
        color: #C0392B;
        font-weight: 700;
        font-size: 0.79em;
        letter-spacing: 0.2px;
        white-space: nowrap;
        border: 1px solid rgba(192,57,43,0.20);
        font-family: 'Lato', sans-serif;
    }
    .badge-yellow {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        background: rgba(183,121,31,0.10);
        color: #B7791F;
        font-weight: 700;
        font-size: 0.79em;
        letter-spacing: 0.2px;
        white-space: nowrap;
        border: 1px solid rgba(183,121,31,0.22);
        font-family: 'Lato', sans-serif;
    }
    .badge-gray {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        background: rgba(122,114,104,0.08);
        color: #7A7268;
        font-weight: 600;
        font-size: 0.79em;
        letter-spacing: 0.2px;
        white-space: nowrap;
        border: 1px solid rgba(122,114,104,0.18);
        font-family: 'Lato', sans-serif;
    }

    /* ─────────────────────────────────────────────
       VARIANT LETTER BADGES
    ───────────────────────────────────────────── */
    .variant-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 24px;
        height: 24px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.82em;
        margin-right: 8px;
        vertical-align: middle;
        font-family: 'Lato', sans-serif;
    }
    .variant-badge-a {
        background: rgba(43,108,176,0.12);
        color: #2B6CB0;
        border: 1px solid rgba(43,108,176,0.25);
    }
    .variant-badge-b {
        background: rgba(39,103,73,0.12);
        color: #276749;
        border: 1px solid rgba(39,103,73,0.25);
    }

    /* ─────────────────────────────────────────────
       CALLOUT BOXES  (left-border accent style)
    ───────────────────────────────────────────── */
    .callout-green {
        border-left: 4px solid #276749;
        background: rgba(39,103,73,0.06);
        padding: 1.1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-family: 'Lato', sans-serif;
    }
    .callout-yellow {
        border-left: 4px solid #B7791F;
        background: rgba(183,121,31,0.06);
        padding: 1.1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-family: 'Lato', sans-serif;
    }
    .callout-red {
        border-left: 4px solid #C0392B;
        background: rgba(192,57,43,0.05);
        padding: 1.1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-family: 'Lato', sans-serif;
    }
    .callout-blue {
        border-left: 4px solid #2B6CB0;
        background: rgba(43,108,176,0.05);
        padding: 1.1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-family: 'Lato', sans-serif;
    }
    .callout-title {
        font-weight: 700;
        font-size: 0.72em;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.5rem;
        font-family: 'Lato', sans-serif;
    }
    .callout-green .callout-title  { color: #276749; }
    .callout-yellow .callout-title { color: #B7791F; }
    .callout-red .callout-title    { color: #C0392B; }
    .callout-blue .callout-title   { color: #2B6CB0; }

    /* ─────────────────────────────────────────────
       EXPERIMENT CARDS  (control vs. variant)
    ───────────────────────────────────────────── */
    .experiment-card {
        background: #FFFFFF;
        border: 1px solid rgba(26,43,34,0.10);
        border-radius: 10px;
        padding: 1.5rem 1.7rem;
        height: 100%;
        box-shadow: 0 2px 8px rgba(26,43,34,0.05);
        transition: box-shadow 0.2s, border-color 0.2s;
        font-family: 'Lato', sans-serif;
    }
    .experiment-card:hover {
        border-color: rgba(26,43,34,0.18);
        box-shadow: 0 4px 16px rgba(26,43,34,0.09);
    }
    .experiment-card h4 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        font-size: 1.05em;
        font-weight: 700;
        font-family: 'Playfair Display', Georgia, serif;
        color: #1A2B22;
    }
    .experiment-card p {
        color: rgba(26,43,34,0.78);
        line-height: 1.6;
        margin-bottom: 0.5rem;
        font-size: 0.94em;
    }
    .experiment-card .card-label {
        font-size: 0.69em;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: rgba(26,43,34,0.38);
        margin-bottom: 0.6rem;
        font-weight: 700;
    }

    /* ─────────────────────────────────────────────
       RESULTS TABLE  (Tab 1 and Tab 3)
    ───────────────────────────────────────────── */
    .results-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.9em;
        margin: 0.75rem 0 1.25rem 0;
        font-family: 'Lato', sans-serif;
        background: #FFFFFF;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(26,43,34,0.10);
        box-shadow: 0 1px 4px rgba(26,43,34,0.05);
    }
    .results-table th {
        text-align: left;
        padding: 0.7rem 1rem;
        background: #EDE8DC;
        border-bottom: 2px solid rgba(26,43,34,0.12);
        font-weight: 700;
        font-size: 0.71em;
        text-transform: uppercase;
        letter-spacing: 0.9px;
        color: rgba(26,43,34,0.50);
        white-space: nowrap;
    }
    .results-table td {
        padding: 0.85rem 1rem;
        border-bottom: 1px solid rgba(26,43,34,0.06);
        vertical-align: middle;
        color: #1A2B22;
    }
    .results-table tbody tr:last-child td {
        border-bottom: none;
    }
    .results-table tbody tr:hover td {
        background: rgba(39,103,73,0.025);
    }
    /* Highlight entire row for significant winners */
    .results-table tr.row-winner td {
        background: rgba(39,103,73,0.04);
    }
    .results-table tr.row-loser td {
        background: rgba(192,57,43,0.035);
    }
    .results-table .metric-name {
        font-weight: 600;
        font-size: 0.93em;
        color: #1A2B22;
    }
    .results-table .metric-value {
        font-family: 'JetBrains Mono', 'Consolas', monospace;
        font-size: 0.89em;
        white-space: nowrap;
        color: rgba(26,43,34,0.70);
    }
    .results-table .lift-positive {
        font-family: 'JetBrains Mono', 'Consolas', monospace;
        font-size: 0.89em;
        color: #276749;
        font-weight: 700;
    }
    .results-table .lift-negative {
        font-family: 'JetBrains Mono', 'Consolas', monospace;
        font-size: 0.89em;
        color: #C0392B;
        font-weight: 700;
    }
    .results-table .lift-neutral {
        font-family: 'JetBrains Mono', 'Consolas', monospace;
        font-size: 0.89em;
        color: #7A7268;
    }
    .results-table .summary-text {
        font-size: 0.87em;
        color: rgba(26,43,34,0.72);
        line-height: 1.5;
        max-width: 260px;
    }

    /* ─────────────────────────────────────────────
       METRIC ROW  (Tab 2 — compact KPI display)
    ───────────────────────────────────────────── */
    .metric-row {
        display: flex;
        gap: 0;
        margin: 1.4rem 0 0.6rem 0;
        border: 1px solid rgba(26,43,34,0.10);
        border-radius: 10px;
        overflow: visible;
        background: #FFFFFF;
        box-shadow: 0 1px 4px rgba(26,43,34,0.05);
    }
    .metric-cell {
        flex: 1;
        padding: 1.1rem 1.4rem;
        border-right: 1px solid rgba(26,43,34,0.07);
    }
    .metric-cell:last-child { border-right: none; }
    .metric-label {
        font-size: 0.69em;
        color: rgba(26,43,34,0.42);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.35rem;
        font-weight: 700;
        font-family: 'Lato', sans-serif;
    }
    /* CSS tooltip — works inside Streamlit iframes where title= does not */
    .has-tooltip {
        position: relative;
        cursor: help;
        display: inline-block;
    }
    .has-tooltip::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: calc(100% + 6px);
        left: 50%;
        transform: translateX(-50%);
        background: #1C2B22;
        color: #E8F0E4;
        font-size: 0.78rem;
        font-weight: 400;
        text-transform: none;
        letter-spacing: 0;
        line-height: 1.5;
        padding: 8px 12px;
        border-radius: 6px;
        border: 1px solid rgba(255,255,255,0.12);
        white-space: normal;
        width: 280px;
        box-shadow: 0 6px 24px rgba(26,43,34,0.35);
        visibility: hidden;
        pointer-events: none;
        transition: visibility 0.15s ease;
        z-index: 99999;
        font-family: 'Lato', sans-serif;
    }
    .has-tooltip:hover::after {
        visibility: visible;
    }
    .metric-val {
        font-size: 1.5em;
        font-weight: 700;
        font-family: 'JetBrains Mono', 'Consolas', monospace;
        line-height: 1.1;
        color: #1A2B22;
    }
    .metric-val-green { color: #276749; }
    .metric-val-red   { color: #C0392B; }
    .metric-val-gray  { color: #7A7268; }
    .metric-sub {
        font-size: 0.71em;
        color: rgba(26,43,34,0.40);
        margin-top: 0.25rem;
        font-weight: 500;
        font-family: 'Lato', sans-serif;
    }

    /* ─────────────────────────────────────────────
       STAT DETAIL PILLS  (Tab 2 technical section)
    ───────────────────────────────────────────── */
    .stat-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        background: rgba(26,43,34,0.05);
        font-family: 'JetBrains Mono', 'Consolas', monospace;
        font-size: 0.82em;
        margin: 3px 4px 3px 0;
        border: 1px solid rgba(26,43,34,0.10);
        color: #1A2B22;
    }
    .stat-pill-label {
        color: rgba(26,43,34,0.42);
        font-size: 0.78em;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-right: 5px;
    }

    /* ─────────────────────────────────────────────
       PROMPT TECHNIQUE CARDS  (Tab 4)
    ───────────────────────────────────────────── */
    .technique-card {
        border: 1px solid rgba(26,43,34,0.09);
        border-radius: 8px;
        padding: 1rem 1.3rem;
        margin-bottom: 0.75rem;
        background: #FFFFFF;
        box-shadow: 0 1px 3px rgba(26,43,34,0.04);
        transition: border-color 0.2s, box-shadow 0.2s;
        font-family: 'Lato', sans-serif;
    }
    .technique-card:hover {
        border-color: rgba(39,103,73,0.28);
        box-shadow: 0 3px 10px rgba(39,103,73,0.07);
    }
    .technique-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 22px;
        height: 22px;
        border-radius: 50%;
        background: rgba(39,103,73,0.12);
        color: #276749;
        font-size: 0.75em;
        font-weight: 700;
        margin-right: 8px;
        flex-shrink: 0;
        vertical-align: middle;
        font-family: 'Lato', sans-serif;
    }
    .technique-name {
        font-weight: 700;
        font-size: 0.95em;
        display: inline;
        color: #1A2B22;
    }
    .technique-location {
        font-size: 0.75em;
        color: rgba(26,43,34,0.40);
        font-style: italic;
        margin-top: 0.4rem;
    }
    .technique-desc {
        font-size: 0.87em;
        line-height: 1.55;
        color: rgba(26,43,34,0.76);
        margin-top: 0.5rem;
    }

    /* ─────────────────────────────────────────────
       SECTION HEADERS  (labeled dividers)
    ───────────────────────────────────────────── */
    .section-header {
        font-size: 0.69em;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: rgba(26,43,34,0.38);
        margin: 2rem 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(26,43,34,0.10);
        font-family: 'Lato', sans-serif;
    }

    /* ─────────────────────────────────────────────
       MISC SPACING
    ───────────────────────────────────────────── */
    .section-spacer { margin-top: 1.5rem; }
    .section-spacer-sm { margin-top: 0.75rem; }

    /* ─────────────────────────────────────────────
       SIDEBAR BRANDING
    ───────────────────────────────────────────── */
    .sidebar-brand {
        font-size: 1.05rem;
        font-weight: 800;
        color: #7ABFA0;
        letter-spacing: -0.3px;
        font-family: 'Playfair Display', Georgia, serif;
    }
    .sidebar-brand-accent {
        display: block;
        width: 28px;
        height: 2px;
        background: linear-gradient(90deg, #4A9E7A, #7ABFA0);
        border-radius: 2px;
        margin-top: 7px;
        margin-bottom: 4px;
    }

    /* ─────────────────────────────────────────────
       CONFIDENCE METER BAR  (power / p-value visual)
    ───────────────────────────────────────────── */
    .conf-meter-wrap {
        margin-top: 0.4rem;
    }
    .conf-meter-bar {
        height: 4px;
        border-radius: 3px;
        background: rgba(26,43,34,0.10);
        overflow: hidden;
        margin: 4px 0 2px 0;
    }
    .conf-meter-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.4s ease;
    }
    .conf-meter-label {
        font-size: 0.70em;
        color: rgba(26,43,34,0.38);
        letter-spacing: 0.3px;
        font-family: 'Lato', sans-serif;
    }

    /* ─────────────────────────────────────────────
       PAGE TITLE AREA
    ───────────────────────────────────────────── */
    .page-title-block {
        margin-bottom: 0.25rem;
    }
    .page-kicker {
        font-size: 0.69em;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: rgba(26,43,34,0.36);
        font-weight: 700;
        margin-bottom: 0.2rem;
        font-family: 'Lato', sans-serif;
    }

    /* ─────────────────────────────────────────────
       SEGMENT TABLE — significance dot styles
    ───────────────────────────────────────────── */
    .results-table .sig-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
        flex-shrink: 0;
    }
    .results-table .sig-dot-green { background: #276749; }
    .results-table .sig-dot-red   { background: #C0392B; }
    .results-table .sig-dot-gray  { background: rgba(122,114,104,0.35); }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────

def verdict_badge(result) -> str:
    """Return a styled HTML badge for the test verdict."""
    if result.is_significant and result.direction == "positive":
        return '<span class="badge-green">&#10004; Winner</span>'
    elif result.is_significant and result.direction == "negative":
        return '<span class="badge-red">&#10008; Underperformed</span>'
    elif result.verdict == "borderline":
        return '<span class="badge-yellow">&#9679; Borderline</span>'
    else:
        return '<span class="badge-gray">&#8212; No Clear Difference</span>'


def verdict_badge_short(result) -> str:
    """Shorter badge for tables with limited space."""
    if result.is_significant and result.direction == "positive":
        return '<span class="badge-green">&#10004; Yes</span>'
    elif result.is_significant and result.direction == "negative":
        return '<span class="badge-red">&#10008; Yes (worse)</span>'
    elif result.verdict == "borderline":
        return '<span class="badge-yellow">&#9679; Borderline</span>'
    else:
        return '<span class="badge-gray">&#8212; No</span>'


def verdict_text_for_expander(result) -> str:
    """Plain text verdict label for st.expander titles (no HTML)."""
    if result.is_significant and result.direction == "positive":
        return "Winner"
    elif result.is_significant and result.direction == "negative":
        return "Underperformed"
    elif result.verdict == "borderline":
        return "Borderline"
    else:
        return "No Clear Difference"


def lift_class(result) -> str:
    """CSS class for the lift column in tables."""
    if result.is_significant and result.direction == "positive":
        return "lift-positive"
    elif result.is_significant and result.direction == "negative":
        return "lift-negative"
    return "lift-neutral"


def row_class(result) -> str:
    """CSS class for a full table row based on result outcome."""
    if result.is_significant and result.direction == "positive":
        return "row-winner"
    elif result.is_significant and result.direction == "negative":
        return "row-loser"
    return ""


def sig_dot(result) -> str:
    """Small colored dot indicating significance — for use in table rows."""
    if result.is_significant and result.direction == "positive":
        return '<span class="sig-dot sig-dot-green"></span>'
    elif result.is_significant and result.direction == "negative":
        return '<span class="sig-dot sig-dot-red"></span>'
    return '<span class="sig-dot sig-dot-gray"></span>'


def confidence_meter_html(power: float, label: str = "Statistical power") -> str:
    """Render a slim progress-bar style confidence meter."""
    pct = int(power * 100)
    if pct >= 80:
        fill_color = "#276749"
    elif pct >= 50:
        fill_color = "#B7791F"
    else:
        fill_color = "#C0392B"
    return (
        f'<div class="conf-meter-wrap">'
        f'<div class="conf-meter-label">{label}: {pct}%</div>'
        f'<div class="conf-meter-bar">'
        f'<div class="conf-meter-fill" style="width:{pct}%;background:{fill_color};"></div>'
        f'</div>'
        f'</div>'
    )


def plain_english_summary(key: str, result) -> str:
    """One-line plain-English interpretation for the results table."""
    lift_pct = result.relative_lift_pct
    direction = "more" if result.absolute_lift > 0 else "fewer"
    direction_rev = "lower" if result.absolute_lift < 0 else "higher"

    summaries = {
        "conversion_rate": (
            f"Personalized banner converted {abs(lift_pct):.1f}% {direction} visitors into buyers."
            if result.is_significant else
            "No meaningful difference in purchase conversion."
        ),
        "banner_ctr": (
            f"The personalized banner got {abs(lift_pct):.1f}% {direction} clicks."
            if result.is_significant else
            "Click-through rates were comparable across both banners."
        ),
        "bounce_rate": (
            f"{abs(lift_pct):.1f}% {direction_rev} bounce rate "
            f"({'visitors stayed longer' if result.absolute_lift < 0 else 'more visitors left immediately'})."
            if result.is_significant else
            "Bounce rates were comparable across both banners."
        ),
        "aov": (
            f"Buyers spent ${abs(result.absolute_lift):.2f} "
            f"{'more' if result.absolute_lift > 0 else 'less'} per order on average."
            if result.is_significant else
            "Average order values were comparable across both groups."
        ),
        "rpv": (
            f"Each visitor generated ${abs(result.absolute_lift):.2f} "
            f"{'more' if result.absolute_lift > 0 else 'less'} in revenue."
            if result.is_significant else
            "Revenue per visitor was comparable across both groups."
        ),
    }
    return summaries.get(key, "")


def _escape_dollars(text: str) -> str:
    """Replace literal $ with HTML entity so Streamlit's LaTeX parser ignores it."""
    return text.replace("$", "&#36;")


def plain_english_detail(key: str, result) -> str:
    """Multi-sentence HTML interpretation for the detailed results tab.
    Returns HTML (using <strong> instead of markdown **) to avoid Streamlit's
    LaTeX parser mangling dollar signs in currency values."""
    val_a = _escape_dollars(format_value(result.value_control, key))
    val_b = _escape_dollars(format_value(result.value_variant, key))
    lift = _escape_dollars(format_lift(result))

    base = {
        "conversion_rate": (
            f"<strong>{val_a}</strong> of visitors who saw the static banner made a purchase, "
            f"compared to <strong>{val_b}</strong> for the personalized banner — "
            f"a shift of <strong>{lift}</strong>."
        ),
        "banner_ctr": (
            f"<strong>{val_a}</strong> of visitors clicked the static banner, versus <strong>{val_b}</strong> "
            f"for the personalized version — a shift of <strong>{lift}</strong>. "
            "Higher engagement suggests the personalized creative resonated more."
        ),
        "bounce_rate": (
            f"<strong>{val_a}</strong> of visitors bounced from the static banner page, "
            f"compared to <strong>{val_b}</strong> for the personalized banner — "
            f"a shift of <strong>{lift}</strong>. Lower bounce means more visitors stayed to browse."
        ),
        "aov": (
            f"Buyers who saw the static banner spent an average of "
            f"<strong>{val_a}</strong> per order. The personalized banner group spent <strong>{val_b}</strong> — "
            f"a difference of <strong>{lift}</strong>."
        ),
        "rpv": (
            f"Across all visitors (including non-buyers), each static banner visitor "
            f"generated <strong>{val_a}</strong> in revenue versus <strong>{val_b}</strong> for the personalized group — "
            f"a difference of <strong>{lift}</strong>. This metric combines conversion rate and order value "
            "into a single revenue measure."
        ),
    }

    if result.is_significant:
        confidence_stmt = (
            f" We're <strong>confident this is real</strong> — less than a "
            f"{result.p_value:.1%} probability it happened by chance."
        )
    elif result.verdict == "borderline":
        confidence_stmt = (
            f" This result is <strong>borderline</strong> (p={result.p_value:.4f}). "
            "There are hints of a real effect, but we can't rule out random variation."
        )
    else:
        confidence_stmt = (
            " This difference is <strong>not statistically significant</strong> — it could easily "
            "be normal random variation between the two groups."
        )

    return base.get(key, "") + confidence_stmt


def generate_bottom_line(topline_results: dict) -> tuple[str, str]:
    """Generate the bottom-line recommendation and its callout class.
    Returns (html_content, callout_class)."""
    conv = topline_results.get("conversion_rate")
    rpv = topline_results.get("rpv")

    has_conversion_win = conv and conv.is_significant and conv.direction == "positive"
    has_rpv_win = rpv and rpv.is_significant and rpv.direction == "positive"

    if has_conversion_win and has_rpv_win:
        return (
            "<strong>Ship with targeted rollout.</strong> The personalized banner lifts both "
            "conversion and revenue per visitor overall. However, segment analysis reveals it "
            "hurts first-time mobile shoppers. Recommend launching for returning visitors and "
            "desktop traffic while optimizing the mobile fallback experience.",
            "callout-yellow"
        )
    elif has_conversion_win:
        return (
            "<strong>Promising, needs investigation.</strong> The personalized banner increases "
            "conversion rate, but the revenue picture is less clear. Review segment breakdowns "
            "before committing to a full rollout.",
            "callout-yellow"
        )
    elif has_rpv_win:
        return (
            "<strong>Cautious optimism.</strong> Revenue per visitor is up, suggesting the "
            "personalized banner drives higher-value purchases. Dig into segments to confirm "
            "the pattern holds across key audiences.",
            "callout-yellow"
        )
    else:
        return (
            "<strong>Do not ship.</strong> The personalized banner did not produce statistically "
            "significant improvements in key business metrics. Consider extending the test duration "
            "or refining the personalization algorithm before retesting.",
            "callout-red"
        )


def generate_segment_takeaway(report, selected_metric_name: str) -> str:
    """Generate a key takeaway for the current segment view. Returns HTML."""
    if not report or not report.slices:
        return ""

    high_anomalies = [a for a in report.anomalies if a.severity == "high"]
    if high_anomalies:
        a = high_anomalies[0]
        return (
            f"<strong>Key finding:</strong> {a.segment_value.title()} shows the opposite "
            f"pattern from the overall result. Investigate before rolling out broadly."
        )

    sig_slices = [s for s in report.slices if s.test_result.is_significant]
    if sig_slices:
        best = max(sig_slices, key=lambda s: abs(s.test_result.relative_lift_pct))
        r = best.test_result
        direction = "improved" if r.direction == "positive" else "declined"
        return (
            f"<strong>Key finding:</strong> {best.segment_value.title()} visitors showed the "
            f"strongest response — {selected_metric_name.lower()} {direction} by "
            f"<strong>{abs(r.relative_lift_pct):.1f}%</strong> with the personalized banner."
        )

    return (
        "<strong>Key finding:</strong> No individual segment shows a statistically "
        "significant difference for this metric."
    )


def business_anomaly_description(anomaly, report=None) -> str:
    """Rewrite anomaly descriptions in business language."""
    seg = anomaly.segment_value.title()

    # Look up visitor count for this segment if report is available
    seg_count_str = ""
    if report and report.slices:
        for s in report.slices:
            if s.segment_value == anomaly.segment_value:
                seg_count_str = f" ({s.n_control + s.n_variant:,} visitors)"
                break

    if anomaly.anomaly_type == "direction_reversal":
        if anomaly.segment_direction == "negative":
            msg = f"**Alert:** {seg} visitors{seg_count_str} performed *worse* with the personalized banner. "
            if "mobile" in anomaly.segment_value.lower() and "new" in anomaly.segment_value.lower():
                msg += (
                    "The personalization engine has no browsing history for first-time "
                    "visitors, and the dynamic banner may load slower on mobile connections."
                )
            else:
                msg += "Personalization may not work for this audience."
        else:
            msg = (
                f"**Unexpected win:** {seg} visitors showed improvement with the personalized banner, "
                f"even though the overall trend goes the other direction."
            )
        return msg

    elif anomaly.anomaly_type == "outsized_effect":
        return (
            f"**High-impact segment:** {seg} visitors{seg_count_str} are driving a disproportionate share of "
            f"the overall result. This group may be the primary beneficiary of personalization."
        )

    elif anomaly.anomaly_type == "significance_loss":
        return (
            f"**Inconsistent pattern:** {seg} visitors{seg_count_str} don't show "
            f"the same effect as the overall result, despite being a sizable audience. "
            f"The personalized banner may not resonate equally with this group."
        )

    return anomaly.description


def _base_chart_layout(height: int = 400, show_legend: bool = True) -> dict:
    """Shared Plotly layout defaults — light botanical theme."""
    return dict(
        height=height,
        plot_bgcolor="rgba(246,242,234,0.5)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color=COLORS["text"],
            family="'Lato', 'Helvetica Neue', sans-serif",
            size=12,
        ),
        showlegend=show_legend,
        margin=dict(l=16, r=16, t=48, b=16),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(size=11.5),
        ),
        hoverlabel=dict(
            bgcolor="rgba(28,43,34,0.93)",
            bordercolor="rgba(255,255,255,0.14)",
            font=dict(size=12, color="rgba(235,245,235,0.95)"),
        ),
    )


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────

st.sidebar.markdown(
    '<div class="sidebar-brand">Evergreen Home &amp; Outdoor</div>'
    '<div class="sidebar-brand-accent"></div>',
    unsafe_allow_html=True,
)
st.sidebar.caption("Spring Sale 2025  ·  Banner Personalization Test")

st.sidebar.divider()

st.sidebar.markdown("**About This Test**")
st.sidebar.markdown(
    "Evergreen's Spring Sale drives ~18% of Q1 revenue. This experiment tested "
    "whether personalizing the homepage banner — showing products from each "
    "visitor's browsing history — would outperform the standard promotional banner."
)

st.sidebar.divider()

with st.sidebar.expander("How to Read This Report"):
    st.markdown(
        "**Conversion Rate** — Share of visitors who made a purchase. Higher is better.\n\n"
        "**Lift** — How much the personalized banner moved the metric relative to the "
        "static banner. Positive means improvement.\n\n"
        "**Confident?** — Whether we can trust the difference is real, not random noise. "
        "Green = yes. Gray = no.\n\n"
        "**Revenue per Visitor** — Total revenue divided by total visitors. Combines "
        "conversion rate and order size into a single number.\n\n"
        "**Segments** — Visitor subgroups (device, traffic source, visitor type, etc.). "
        "A test can win overall but underperform in specific segments."
    )

st.sidebar.divider()

st.sidebar.markdown("**Analysis Settings**")
confidence = st.sidebar.slider(
    "Confidence Level",
    min_value=0.90,
    max_value=0.99,
    value=0.95,
    step=0.01,
    format="%.2f",
    help="How certain we need to be before calling a result real. 95% is the industry standard (recommended). Changing this recalculates all results.",
)
alpha = round(1 - confidence, 4)
st.sidebar.caption(f"Significance threshold: {alpha:.0%} false positive rate")


# ──────────────────────────────────────────────────────────────
# DATA LOADING (cached)
# ──────────────────────────────────────────────────────────────

CSV_PATH = _PROJECT_DIR / "ab_test_homepage_banner.csv"


@st.cache_data
def load_data():
    raw = pd.read_csv(CSV_PATH)
    return prepare_data(raw, exclude_contaminated=True)


@st.cache_data
def load_raw_summary():
    raw = pd.read_csv(CSV_PATH)
    df_clean = prepare_data(raw, exclude_contaminated=True)
    return len(raw), len(df_clean)


@st.cache_data
def compute_topline(confidence_level: float, alpha_val: float):
    df = load_data()
    return run_all_tests(df, confidence=confidence_level, alpha=alpha_val)


@st.cache_data
def compute_segments(confidence_level: float, alpha_val: float):
    df = load_data()
    topline = run_all_tests(df, confidence=confidence_level, alpha=alpha_val)
    return full_scan_all_metrics(
        df, confidence=confidence_level, alpha=alpha_val,
        include_compound=True, topline_results=topline,
    )


df = load_data()
raw_count, clean_count = load_raw_summary()
summary = get_data_summary(df)
topline_results = compute_topline(confidence, alpha)
segment_results = compute_segments(confidence, alpha)


# ──────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────

# Kicker line + title establishes context before the tabs load
st.markdown(
    '<div class="page-kicker">A/B Test Report  ·  Evergreen Home &amp; Outdoor  ·  Spring Sale 2025</div>',
    unsafe_allow_html=True,
)
st.title("Homepage Banner Personalization Test")
st.markdown(
    "Does replacing the static promotional banner with one personalized to each visitor's "
    "browsing history increase sales? This report answers that question with full statistical "
    "evidence, segment-level breakdowns, and an AI-generated executive briefing."
)

# ──────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary",
    "Detailed Results",
    "Who Does It Work For?",
    "AI Summary",
])


# ══════════════════════════════════════════════════════════════
# TAB 1: EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════

with tab1:

    # --- Bottom Line Recommendation ---
    bottom_line_html, callout_class = generate_bottom_line(topline_results)
    st.markdown(
        f'<div class="{callout_class}">'
        f'<div class="callout-title">Recommendation</div>'
        f'{bottom_line_html}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    # --- The Experiment ---
    st.markdown('<div class="section-header">The Experiment</div>', unsafe_allow_html=True)

    col_control, col_variant = st.columns(2, gap="medium")
    with col_control:
        st.markdown(
            '<div class="experiment-card">'
            '<div class="card-label">Control Group  ·  50% of visitors</div>'
            '<h4><span class="variant-badge variant-badge-a">A</span>Static Banner</h4>'
            '<p><strong>"Spring Sale: Up to 40% Off"</strong></p>'
            '<p>A hand-designed hero banner featuring best-selling patio furniture. '
            'Identical for every visitor regardless of browsing history.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_variant:
        st.markdown(
            '<div class="experiment-card">'
            '<div class="card-label">Test Group  ·  50% of visitors</div>'
            '<h4><span class="variant-badge variant-badge-b">B</span>Personalized Banner</h4>'
            '<p><strong>Dynamic products based on browsing history</strong></p>'
            '<p>Shows sale items from categories each visitor previously browsed. '
            'First-time visitors with no browsing history see the static banner as a fallback.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    # --- Test Metadata KPIs ---
    st.markdown('<div class="section-header">Test Details</div>', unsafe_allow_html=True)

    excluded = raw_count - clean_count
    col1, col2, col3, col4, col5 = st.columns(5)

    # Format dates for readability
    _ds = summary.get("date_start", "")
    _de = summary.get("date_end", "")
    try:
        from datetime import datetime as _dt
        _ds_fmt = _dt.strptime(_ds, "%Y-%m-%d").strftime("%b %d, %Y")
        _de_fmt = _dt.strptime(_de, "%Y-%m-%d").strftime("%b %d, %Y")
    except (ValueError, TypeError):
        _ds_fmt, _de_fmt = _ds or "—", _de or "—"

    with col1:
        st.metric(
            "Visitors Analyzed",
            f"{summary['total_visitors']:,}",
            help="Total visitors included after removing those with data quality issues.",
        )
    with col2:
        st.metric(
            "Static Banner (A)",
            f"{summary['control_count']:,}",
            help="Visitors randomly assigned to the original static banner.",
        )
    with col3:
        st.metric(
            "Personalized Banner (B)",
            f"{summary['variant_count']:,}",
            help="Visitors randomly assigned to the new personalized banner.",
        )
    with col4:
        st.metric(
            "Test Duration",
            f"{summary.get('test_duration_days', '—')} days",
            help=f"From {_ds_fmt} to {_de_fmt}.",
        )
    with col5:
        st.metric(
            "Excluded (contaminated)",
            f"{excluded:,}",
            help=f"{excluded:,} visitors saw both banners and were removed to keep groups cleanly separated.",
        )

    st.divider()

    # --- Results Summary Table ---
    st.markdown('<div class="section-header">How Did the Personalized Banner Perform?</div>', unsafe_allow_html=True)

    # Build a richer table with row-level color coding and a colored lift column
    table_rows = ""
    for key, result in topline_results.items():
        badge = verdict_badge_short(result)
        summary_text = plain_english_summary(key, result)
        lift_str = _escape_dollars(format_lift(result))
        l_class = lift_class(result)
        r_class = row_class(result)
        dot = sig_dot(result)

        table_rows += (
            f'<tr class="{r_class}">'
            f'<td class="metric-name">{dot}{result.metric_name}</td>'
            f'<td class="metric-value">{_escape_dollars(format_value(result.value_control, key))}</td>'
            f'<td class="metric-value">{_escape_dollars(format_value(result.value_variant, key))}</td>'
            f'<td class="{l_class}">{lift_str}</td>'
            f'<td>{badge}</td>'
            f'<td class="summary-text">{summary_text}</td>'
            f'</tr>'
        )

    st.markdown(
        f'<table class="results-table">'
        f'<thead><tr>'
        f'<th>Metric</th>'
        f'<th>Static (A)</th>'
        f'<th>Personalized (B)</th>'
        f'<th>Change</th>'
        f'<th>Statistically Confident?</th>'
        f'<th>Plain-English Takeaway</th>'
        f'</tr></thead>'
        f'<tbody>{table_rows}</tbody>'
        f'</table>',
        unsafe_allow_html=True,
    )

    # --- Narrative Takeaway ---
    st.markdown('<div class="section-spacer-sm"></div>', unsafe_allow_html=True)

    sig_positive = [r for r in topline_results.values() if r.is_significant and r.direction == "positive"]
    sig_negative = [r for r in topline_results.values() if r.is_significant and r.direction == "negative"]

    narrative_parts = []
    if sig_positive:
        names = ", ".join(r.metric_name.lower() for r in sig_positive)
        narrative_parts.append(f"The personalized banner delivered statistically significant gains in **{names}**.")
    if sig_negative:
        names = ", ".join(r.metric_name.lower() for r in sig_negative)
        narrative_parts.append(f"However, it showed a significant *decline* in **{names}**.")
    if not sig_positive and not sig_negative:
        narrative_parts.append("No metrics showed a statistically significant difference between the two banners.")

    narrative_parts.append(
        "These gains aren't uniform across all audiences — some segments "
        "respond differently. Explore the **\"Who Does It Work For?\"** tab "
        "to see who benefits most."
    )

    st.markdown(" ".join(narrative_parts))


# ══════════════════════════════════════════════════════════════
# TAB 2: DETAILED RESULTS
# ══════════════════════════════════════════════════════════════

with tab2:
    st.markdown(
        "Full statistical evidence for each metric. "
        "Expand any metric to see the breakdown. "
        "The first significant result opens automatically."
    )

    st.markdown('<div class="section-spacer-sm"></div>', unsafe_allow_html=True)

    first_sig = True
    for key, result in topline_results.items():
        expand = first_sig and result.is_significant
        if expand:
            first_sig = False

        verdict_label = verdict_text_for_expander(result)
        verdict_icon = (
            "✓" if (result.is_significant and result.direction == "positive") else
            "✗" if (result.is_significant and result.direction == "negative") else
            "~" if result.verdict == "borderline" else
            "–"
        )

        with st.expander(
            f"{verdict_icon}  {result.metric_name}   ·   {verdict_label}",
            expanded=expand,
        ):
            # --- Narrative interpretation ---
            st.markdown(plain_english_detail(key, result), unsafe_allow_html=True)

            if not result.is_significant and result.verdict != "borderline":
                st.caption("No action needed — the difference is within normal random variation.")

            # --- Compact KPI row with sub-labels ---
            val_a_fmt = _escape_dollars(format_value(result.value_control, key))
            val_b_fmt = _escape_dollars(format_value(result.value_variant, key))
            lift_fmt = _escape_dollars(format_lift(result))
            if result.is_significant and result.direction == "positive":
                lift_color_cls = "metric-val-green"
            elif result.is_significant and result.direction == "negative":
                lift_color_cls = "metric-val-red"
            else:
                lift_color_cls = "metric-val-gray"

            p_display = f"p = {result.p_value:.4f}"
            power_display = f"{result.observed_power:.0%}"

            st.markdown(
                f'<div class="metric-row">'
                # Control
                f'<div class="metric-cell">'
                f'<div class="metric-label has-tooltip" data-tooltip="The baseline result — how the current static banner performed. This is what the personalized banner is being measured against.">Static Banner (A) ⓘ</div>'
                f'<div class="metric-val">{val_a_fmt}</div>'
                f'<div class="metric-sub">n = {result.n_control:,}</div>'
                f'</div>'
                # Variant
                f'<div class="metric-cell">'
                f'<div class="metric-label has-tooltip" data-tooltip="The result for the new personalized banner. A higher number than Static Banner (A) means the new experience is outperforming the current one.">Personalized Banner (B) ⓘ</div>'
                f'<div class="metric-val">{val_b_fmt}</div>'
                f'<div class="metric-sub">n = {result.n_variant:,}</div>'
                f'</div>'
                # Lift
                f'<div class="metric-cell">'
                f'<div class="metric-label has-tooltip" data-tooltip="How much better (or worse) the personalized banner performed relative to the static one, expressed as a percentage change. A positive number means improvement; negative means the static banner won.">Change (B vs. A) ⓘ</div>'
                f'<div class="metric-val {lift_color_cls}">{lift_fmt}</div>'
                f'<div class="metric-sub">{p_display}</div>'
                f'</div>'
                # Power meter
                f'<div class="metric-cell">'
                f'<div class="metric-label has-tooltip" data-tooltip="Think of this as the test\'s sensitivity. An 80% power means: if the personalized banner truly performs better, this test had an 80% chance of catching it. Below 80%, the test may have missed a real improvement — so a \'no difference\' result should be treated with caution.">Statistical Power ⓘ</div>'
                f'<div class="metric-val" style="font-size:1.1em">{power_display}</div>'
                f'{confidence_meter_html(result.observed_power, "")}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── Bar chart: A vs. B comparison ───────────────────────
            #
            # Design decisions:
            #   • Each variant gets its own named x-category so Plotly
            #     spreads them naturally across the full plot width instead
            #     of squeezing two bars together at a single "" category.
            #   • textposition="inside" with white text keeps value labels
            #     inside the bars, completely avoiding any collision with
            #     the lift annotation or the chart ceiling.
            #   • For very short bars (small proportions like CTR) the text
            #     would be invisible inside, so we use textposition="auto"
            #     which Plotly resolves per-bar.
            #   • The lift annotation is anchored to paper coordinates at a
            #     fixed y=0.91 (top of the plot area), so it never moves
            #     relative to bar height and can never collide with bar text.
            #   • The CI is shown as a clean st.caption below the chart
            #     rather than a floating shape, which was rendering at an
            #     arbitrary x position unrelated to the bars.
            #   • y_max = top_val * 1.22 leaves enough headroom for the
            #     "auto" outside labels on short bars without wasting space.

            fmt_key = METRIC_REGISTRY.get(key, {}).get("format", "number")
            val_a_raw = format_value(result.value_control, key)
            val_b_raw = format_value(result.value_variant, key)

            # Determine whether bar values are "small" (< 0.3 in data
            # units), which means inside labels would be too cramped.
            top_val = max(result.value_control, result.value_variant)
            use_inside = top_val > 0.12   # proportions >12% have room inside

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name="Static Banner (A)",
                x=["Static Banner (A)"],
                y=[result.value_control],
                marker_color=COLORS["control"],
                marker_opacity=0.88,
                text=[val_a_raw],
                textposition="inside" if use_inside else "outside",
                textfont=dict(
                    color="white" if use_inside else COLORS["text"],
                    size=14,
                    family="'SF Mono', 'Consolas', monospace",
                ),
                insidetextanchor="middle",
                hovertemplate=(
                    "<b>Static Banner (A)</b><br>"
                    f"Value: {val_a_raw}<br>"
                    f"n = {result.n_control:,}<extra></extra>"
                ),
            ))
            fig_bar.add_trace(go.Bar(
                name="Personalized Banner (B)",
                x=["Personalized Banner (B)"],
                y=[result.value_variant],
                marker_color=COLORS["variant"],
                marker_opacity=0.88,
                text=[val_b_raw],
                textposition="inside" if use_inside else "outside",
                textfont=dict(
                    color="white" if use_inside else COLORS["text"],
                    size=14,
                    family="'SF Mono', 'Consolas', monospace",
                ),
                insidetextanchor="middle",
                hovertemplate=(
                    "<b>Personalized Banner (B)</b><br>"
                    f"Value: {val_b_raw}<br>"
                    f"n = {result.n_variant:,}<extra></extra>"
                ),
            ))

            # Lift annotation — pinned to paper-space top-center so it
            # is always readable regardless of bar scale.
            lift_label = f"{result.relative_lift_pct:+.1f}%"
            abs_lift_fmt = _escape_dollars(format_lift(result))
            if result.is_significant and result.direction == "positive":
                lift_annotation_color = COLORS["positive"]
                lift_bg = "rgba(46,204,138,0.12)"
            elif result.is_significant and result.direction == "negative":
                lift_annotation_color = COLORS["negative"]
                lift_bg = "rgba(224,90,90,0.12)"
            else:
                lift_annotation_color = COLORS["neutral"]
                lift_bg = "rgba(138,149,163,0.10)"

            fig_bar.add_annotation(
                x=0.5,
                y=1.0,
                xref="paper",
                yref="paper",
                xanchor="center",
                yanchor="top",
                text=f"<b>Change: {lift_label}</b>",
                showarrow=False,
                font=dict(size=15, color=lift_annotation_color),
                bgcolor=lift_bg,
                bordercolor=lift_annotation_color,
                borderwidth=1,
                borderpad=6,
                opacity=0.95,
            )

            # y-axis tick format so the scale is readable at a glance
            if fmt_key == "percent":
                ytickfmt = ".1%"
            elif fmt_key == "currency":
                ytickfmt = "$,.2f"
            else:
                ytickfmt = ""

            # Headroom: "outside" labels need more room than "inside".
            y_max = top_val * (1.22 if use_inside else 1.55)

            fig_bar.update_layout(
                barmode="group",
                height=340,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=56, r=24, t=56, b=48),
                showlegend=False,   # x-axis labels already name the groups
                font=dict(
                    color=COLORS["text"],
                    family="'Inter', 'Segoe UI', 'Helvetica Neue', sans-serif",
                    size=12,
                ),
                xaxis=dict(
                    tickfont=dict(size=13, color=COLORS["text"]),
                    linecolor="rgba(255,255,255,0.1)",
                    gridcolor="rgba(0,0,0,0)",
                ),
                yaxis=dict(
                    range=[0, y_max],
                    tickformat=ytickfmt,
                    tickfont=dict(size=11, color=COLORS["text_dim"]),
                    gridcolor=COLORS["grid"],
                    zeroline=False,
                    title=dict(
                        text=result.metric_name,
                        font=dict(size=11, color=COLORS["text_dim"]),
                    ),
                ),
                hoverlabel=dict(
                    bgcolor="rgba(20,20,30,0.92)",
                    bordercolor="rgba(255,255,255,0.15)",
                    font=dict(size=12, color="rgba(255,255,255,0.9)"),
                ),
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

            # CI displayed as a readable caption — cleaner than a floating
            # shape that renders at an arbitrary chart position.
            if fmt_key == "percent":
                ci_lo_str = f"{result.ci_lower:+.3%}"
                ci_hi_str = f"{result.ci_upper:+.3%}"
            elif fmt_key == "currency":
                ci_lo_str = f"${result.ci_lower:+.2f}"
                ci_hi_str = f"${result.ci_upper:+.2f}"
            else:
                ci_lo_str = f"{result.ci_lower:+.4f}"
                ci_hi_str = f"{result.ci_upper:+.4f}"
            st.caption(
                f"{confidence:.0%} confidence interval on the lift: "
                f"[{ci_lo_str},  {ci_hi_str}]"
                + ("  — excludes zero, consistent with significance." if result.is_significant else
                   "  — spans zero, consistent with no significant difference.")
            )

            # --- Technical details (nested expander) ---
            with st.expander("Technical Details"):
                # Render as styled pills rather than plain markdown lines
                fmt_meta = METRIC_REGISTRY.get(key, {}).get("format", "number")
                if fmt_meta == "percent":
                    ci_lo_str = f"{result.ci_lower:+.4%}"
                    ci_hi_str = f"{result.ci_upper:+.4%}"
                elif fmt_meta == "currency":
                    ci_lo_str = f"${result.ci_lower:+.4f}"
                    ci_hi_str = f"${result.ci_upper:+.4f}"
                else:
                    ci_lo_str = f"{result.ci_lower:+.6f}"
                    ci_hi_str = f"{result.ci_upper:+.6f}"

                pills_html = (
                    f'<span class="stat-pill has-tooltip" data-tooltip="The statistical method used to compare the two groups. The choice depends on the type of metric — proportions (like conversion rate) use one method, averages (like revenue) use another."><span class="stat-pill-label">Test</span>{result.test_used}</span>'
                    f'<span class="stat-pill has-tooltip" data-tooltip="The probability of seeing results this large purely by chance, if there were actually no real difference. Under 5% (0.05) is the standard threshold for calling a result statistically significant."><span class="stat-pill-label">p-value</span>{result.p_value:.6f}</span>'
                    f'<span class="stat-pill has-tooltip" data-tooltip="The range we\'re 95% confident the true lift falls within. If this range does not include zero, the result is statistically significant. A narrow range means a more precise estimate."><span class="stat-pill-label">95% CI</span>[{ci_lo_str}, {ci_hi_str}]</span>'
                    f'<span class="stat-pill has-tooltip" data-tooltip="A standardized measure of how large the difference is, independent of sample size. Small: under 0.2. Medium: around 0.5. Large: above 0.8. A result can be statistically significant but still have a small practical effect."><span class="stat-pill-label">Effect size</span>{result.effect_size:.4f}</span>'
                    f'<span class="stat-pill has-tooltip" data-tooltip="Think of this as the test\'s sensitivity. An 80% power means: if the personalized banner truly performs better, this test had an 80% chance of catching it. Below 80%, a \'no difference\' result should be treated with caution."><span class="stat-pill-label">Power</span>{result.observed_power:.1%}</span>'
                    f'<span class="stat-pill has-tooltip" data-tooltip="The overall conclusion of the statistical test. \'Significant winner\' means the result is unlikely to be due to chance. \'No significant difference\' means we cannot confidently declare a winner yet."><span class="stat-pill-label">Verdict</span>{result.verdict.replace("_", " ").title()}</span>'
                )
                if result.min_sample_needed:
                    pills_html += (
                        f'<span class="stat-pill has-tooltip" data-tooltip="The minimum number of visitors needed in each group to reliably detect a real difference with 80% confidence. If the actual sample is below this, the test may not have run long enough to draw firm conclusions."><span class="stat-pill-label">Min n (80% power)</span>'
                        f'{result.min_sample_needed:,} per arm</span>'
                    )
                st.markdown(pills_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 3: WHO DOES IT WORK FOR?
# ══════════════════════════════════════════════════════════════

with tab3:
    st.markdown(
        "The personalized banner doesn't affect every visitor the same way. "
        "Some audiences see a strong lift; others actually respond better to the static banner. "
        "Use the filters below to explore results by segment. "
        "**Start with Purchase Conversion Rate by Device x Visitor Type** — "
        "it reveals the most actionable patterns."
    )

    st.markdown('<div class="section-spacer-sm"></div>', unsafe_allow_html=True)

    # --- Dropdowns ---
    col_metric, col_seg = st.columns(2, gap="medium")

    metric_options = {v["display_name"]: k for k, v in METRIC_REGISTRY.items()}
    with col_metric:
        selected_metric_name = st.selectbox(
            "Metric",
            options=list(metric_options.keys()),
            help="Select the business metric to break down by segment.",
        )
    selected_metric_key = metric_options[selected_metric_name]

    all_segments = {}
    for k, v in SEGMENT_REGISTRY.items():
        all_segments[v["display_name"]] = k
    for k, v in COMPOUND_SEGMENTS.items():
        all_segments[v["display_name"]] = k

    seg_names = list(all_segments.keys())
    default_seg_idx = next(
        (i for i, name in enumerate(seg_names) if all_segments[name] == "device_x_visitor"),
        0,
    )

    with col_seg:
        selected_seg_name = st.selectbox(
            "Break Down By",
            options=seg_names,
            index=default_seg_idx,
            help="Choose how to split visitors into groups. 'Device x Visitor Type' reveals the most interesting patterns in this test.",
        )
    selected_seg_key = all_segments[selected_seg_name]

    # --- Find the matching report ---
    report = None
    if selected_metric_key in segment_results:
        for r in segment_results[selected_metric_key]:
            if r.segment_dimension == selected_seg_key:
                report = r
                break

    if report and report.slices:

        # --- Key Takeaway ---
        takeaway = generate_segment_takeaway(report, selected_metric_name)
        if takeaway:
            st.markdown(
                f'<div class="callout-blue">{takeaway}</div>',
                unsafe_allow_html=True,
            )

        # --- Anomaly callouts ---
        if report.anomalies:
            for anomaly in report.anomalies:
                biz_desc = business_anomaly_description(anomaly, report)
                if anomaly.severity == "high":
                    st.error(biz_desc)
                elif anomaly.severity == "medium":
                    st.warning(biz_desc)
                else:
                    st.info(biz_desc)

        st.markdown('<div class="section-spacer-sm"></div>', unsafe_allow_html=True)

        # --- Segment results table ---
        st.markdown(
            '<div class="section-header">Segment-Level Results</div>',
            unsafe_allow_html=True,
        )
        fmt = METRIC_REGISTRY[selected_metric_key].get("format", "number")
        seg_table_rows = ""
        for s in report.slices:
            r = s.test_result
            if fmt == "percent":
                val_a = f"{r.value_control:.2%}"
                val_b = f"{r.value_variant:.2%}"
                lift_str = f"{r.absolute_lift:+.2%}"
            elif fmt == "currency":
                val_a = f"${r.value_control:.2f}"
                val_b = f"${r.value_variant:.2f}"
                lift_str = f"${r.absolute_lift:+.2f}"
            else:
                val_a = f"{r.value_control:.4f}"
                val_b = f"{r.value_variant:.4f}"
                lift_str = f"{r.absolute_lift:+.4f}"

            badge = verdict_badge_short(r)
            n_total = s.n_control + s.n_variant
            l_class = lift_class(r)
            r_class = row_class(r)
            dot = sig_dot(r)

            seg_table_rows += (
                f'<tr class="{r_class}">'
                f'<td class="metric-name">{dot}{s.segment_value.title()}</td>'
                f'<td class="metric-value">{val_a}</td>'
                f'<td class="metric-value">{val_b}</td>'
                f'<td class="{l_class}">{lift_str}</td>'
                f'<td class="{l_class}">{r.relative_lift_pct:+.1f}%</td>'
                f'<td>{badge}</td>'
                f'<td class="metric-value" style="opacity:0.6">{n_total:,}</td>'
                f'</tr>'
            )

        st.markdown(
            f'<table class="results-table">'
            f'<thead><tr>'
            f'<th>Segment</th>'
            f'<th>Static (A)</th>'
            f'<th>Personalized (B)</th>'
            f'<th>Absolute Change</th>'
            f'<th>Relative Change</th>'
            f'<th>Statistically Confident?</th>'
            f'<th>Visitors</th>'
            f'</tr></thead>'
            f'<tbody>{seg_table_rows}</tbody>'
            f'</table>',
            unsafe_allow_html=True,
        )

        # --- Segment lift chart ---
        st.markdown(
            '<div class="section-header">Lift by Segment — Absolute Change vs. Static Banner</div>',
            unsafe_allow_html=True,
        )

        chart_data = []
        for s in report.slices:
            r = s.test_result
            chart_data.append({
                "Segment": s.segment_value.title(),
                "Absolute Lift": r.absolute_lift,
                "Direction": "Improvement" if r.absolute_lift >= 0 else "Decline",
                "Confidence": "Statistically confident" if r.is_significant else "Not statistically confident",
                "Relative Change": f"{r.relative_lift_pct:+.1f}%",
                "p-value": f"p = {r.p_value:.4f}",
                "n": f"n = {s.n_control + s.n_variant:,}",
                # Opacity encodes statistical confidence
                "Opacity": 1.0 if r.is_significant else 0.42,
            })

        chart_df = pd.DataFrame(chart_data)

        # Build bars manually so we can control per-bar opacity
        fig = go.Figure()

        for _, row in chart_df.iterrows():
            color = COLORS["positive"] if row["Absolute Lift"] >= 0 else COLORS["negative"]
            fig.add_trace(go.Bar(
                name=row["Direction"],
                x=[row["Segment"]],
                y=[row["Absolute Lift"]],
                marker_color=color,
                marker_opacity=row["Opacity"],
                text=row["Relative Change"],
                textposition="outside",
                textfont=dict(
                    color=COLORS["text"] if row["Opacity"] > 0.5 else COLORS["text_dim"],
                    size=11,
                ),
                customdata=[[row["Confidence"], row["p-value"], row["n"]]],
                hovertemplate=(
                    f"<b>{row['Segment']}</b><br>"
                    f"Lift: {row['Relative Change']}<br>"
                    f"%{{customdata[0]}}<br>"
                    f"%{{customdata[1]}}<br>"
                    f"%{{customdata[2]}}<extra></extra>"
                ),
                showlegend=False,
            ))

        # Zero reference line
        fig.add_hline(
            y=0,
            line_color=COLORS["zero"],
            line_width=1.5,
            annotation_text="No change",
            annotation_font=dict(size=10, color=COLORS["text_dim"]),
            annotation_position="top right",
        )

        # Annotation explaining opacity encoding
        fig.add_annotation(
            x=1, y=1,
            xref="paper", yref="paper",
            xanchor="right", yanchor="top",
            text="<i>Faded bars = not statistically confident</i>",
            showarrow=False,
            font=dict(size=10, color=COLORS["text_dim"]),
        )

        layout = _base_chart_layout(height=460, show_legend=False)
        layout.update(
            xaxis=dict(
                title="",
                tickfont=dict(size=11.5),
                gridcolor="rgba(0,0,0,0)",
                linecolor="rgba(255,255,255,0.1)",
            ),
            yaxis=dict(
                title=f"Change in {selected_metric_name}",
                gridcolor=COLORS["grid"],
                zeroline=False,
                tickformat=".1%" if fmt == "percent" else "$.2f" if fmt == "currency" else "",
                tickfont=dict(size=11),
            ),
            margin=dict(l=16, r=16, t=52, b=16),
        )
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    else:
        st.warning("No segment data available for this combination.")


# ══════════════════════════════════════════════════════════════
# TAB 4: AI SUMMARY
# ══════════════════════════════════════════════════════════════

with tab4:
    st.markdown(
        "A plain-English executive briefing powered by Claude (Anthropic), "
        "written from the statistical results above. Expand the sections "
        "below to see how the prompt is engineered and what data the AI received."
    )

    st.markdown('<div class="section-spacer-sm"></div>', unsafe_allow_html=True)

    @st.cache_data
    def get_summary_payload(confidence_level: float, alpha_val: float):
        data = load_data()
        return build_summary_payload(data, confidence=confidence_level, alpha=alpha_val)

    payload = get_summary_payload(confidence, alpha)

    # --- Pre-generated Executive Summary ---
    _summary_path = Path(__file__).parent / "ai_generated_summary.md"
    _summary_text = _summary_path.read_text(encoding="utf-8")

    with st.expander("Executive Summary — AI-Generated Briefing", expanded=True):
        st.markdown(_escape_dollars(_summary_text), unsafe_allow_html=True)

    # --- How the Prompt Works ---
    with st.expander("How the Prompt Works — Engineering Techniques"):
        st.markdown(
            "The prompt uses several deliberate techniques to produce "
            "consistent, high-quality analytical writing. Each one targets "
            "a specific failure mode that language models exhibit when "
            "generating business analysis."
        )
        st.markdown('<div class="section-spacer-sm"></div>', unsafe_allow_html=True)

        # Render technique cards with visual separation and numbered circles
        cards_html = ""
        for i, technique in enumerate(PROMPT_TECHNIQUES):
            cards_html += (
                f'<div class="technique-card">'
                f'<span class="technique-number">{i + 1}</span>'
                f'<span class="technique-name">{technique["name"]}</span>'
                f'<div class="technique-desc">{technique["description"]}</div>'
                f'<div class="technique-location">Applied in: {technique["location"]}</div>'
                f'</div>'
            )
        st.markdown(cards_html, unsafe_allow_html=True)

        st.markdown('<div class="section-spacer-sm"></div>', unsafe_allow_html=True)
        st.markdown(
            "**Model:** Claude Sonnet 4.6 &nbsp;·&nbsp; **Temperature:** 0.3 — "
            "low enough for analytical precision, high enough for natural prose.",
            unsafe_allow_html=True,
        )

    # --- Raw Prompt Preview ---
    with st.expander("Raw Prompt Preview — See What the AI Receives"):
        system_prompt, user_prompt = preview_prompt(payload)
        st.markdown("**System Prompt:**")
        st.code(system_prompt, language="text")
        st.markdown("**User Prompt (constructed from live data):**")
        st.code(user_prompt, language="text")

        total_chars = len(system_prompt) + len(user_prompt)
        est_tokens = total_chars // 4
        st.caption(
            f"Estimated prompt size: ~{est_tokens:,} tokens  ·  "
            f"System: {len(system_prompt):,} chars  ·  "
            f"User: {len(user_prompt):,} chars"
        )
