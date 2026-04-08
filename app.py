"""
app.py — Augusta Pool Intelligence 2026
Run: streamlit run app.py
"""
from __future__ import annotations

from datetime import datetime, timezone
import time

import numpy as np
import pandas as pd
import streamlit as st

from fetch_data import fetch_all_data, PLAYER_MASTERS_HISTORY, FALLBACK_PLAYER_STATS
from score_engine import score_players, _detect_chaos_mode, american_to_implied_prob
from pool_optimizer import generate_teams, score_custom_team, compute_pool_tiebreaker
from tiebreaker import predict_tiebreaker
from live_tracker import (
    parse_pool_entries, fetch_live_scores, compute_standings,
    DEMO_ENTRIES, DEMO_LIVE_SCORES,
)

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG  — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Augusta Pool Intelligence 2026",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────
# DESIGN SYSTEM — injected once at startup
# ─────────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@1,300&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg:     #060d06;
  --s1:     #0d160d;
  --s2:     #111d11;
  --b1:     #1e301e;
  --b2:     #263826;
  --b3:     #2a3e2a;
  --t1:     #e8f5e8;
  --t2:     #96cc96;
  --t2b:    #7aaa7a;
  --t3:     #5a8a5a;
  --t4:     #4a6a4a;
  --gold:   #c8a84a;
  --gold2:  #8a7030;
  --green:  #3aaa5a;
  --green2: #52cc72;
  --red:    #cc4a4a;
  --azalea: #cc4a80;
}

/* ── KEYBOARD ARTIFACT SUPPRESSION ──────────────────────────── */
kbd { display: none !important; }
[data-testid="stExpander"] kbd { display: none !important; }
.streamlit-expanderHeader kbd { display: none !important; }

/* ── SELECTBOX ───────────────────────────────────────────────── */
[data-testid="stSelectbox"] {
  background: #080e08 !important;
}
[data-testid="stSelectbox"] > div > div {
  background: #080e08 !important;
  border: 1px solid #1e2e1e !important;
  color: #7aaa7a !important;
  border-radius: 4px !important;
}
[data-testid="stSelectbox"] span,
[data-testid="stSelectbox"] p {
  color: #7aaa7a !important;
  font-size: 12px !important;
  font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSelectboxVirtualDropdown"] {
  background: #080e08 !important;
  border: 1px solid #1e2e1e !important;
}
[data-testid="stSelectboxVirtualDropdown"] li {
  color: #7aaa7a !important;
  font-size: 12px !important;
  background: #080e08 !important;
}
[data-testid="stSelectboxVirtualDropdown"] li:hover {
  background: #0c130c !important;
}

/* ── STREAMLIT RESET ─────────────────────────────────────────── */
.main > div { padding-top: 0 !important; }
[data-testid="stAppViewContainer"] { background: #000 !important; }
.stApp { background: var(--bg) !important; }

.main .block-container,
[data-testid="stMainBlockContainer"] {
  padding: 0 !important;
  max-width: 100% !important;
  overflow-x: hidden !important;
}

/* Prevent columns from blowing out the viewport */
[data-testid="stHorizontalBlock"] {
  gap: 12px !important;
  flex-wrap: nowrap !important;
  width: 100% !important;
  min-width: 0 !important;
}
[data-testid="stColumn"] {
  min-width: 0 !important;
  overflow: hidden !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
footer { display: none !important; }

[data-testid="stSidebar"],
section[data-testid="stSidebar"],
[data-testid="stSidebarNav"],
.css-1d391kg,
.css-163ttbj { display: none !important; }

/* Prevent the main content from leaving gap where sidebar was */
.main { margin-left: 0 !important; }
[data-testid="stAppViewContainer"] > .main { padding-left: 0 !important; }

* { font-family: 'DM Sans', sans-serif !important; }

/* ── TABS ────────────────────────────────────────────────────── */
[data-testid="stTabs"] { background: var(--bg) !important; }

[data-testid="stTabs"] > div:first-child {
  border-bottom: 1px solid var(--b1) !important;
  gap: 0 !important;
  padding: 0 24px !important;
  background: var(--bg) !important;
}

[data-testid="stTabs"] button[role="tab"] {
  background: transparent !important;
  border: none !important;
  border-bottom: 1px solid transparent !important;
  color: var(--t4) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  padding: 12px 18px !important;
  border-radius: 0 !important;
  margin-bottom: -1px !important;
  transition: color 0.12s !important;
}

[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: var(--t1) !important;
  border-bottom: 1px solid var(--green) !important;
}

[data-testid="stTabs"] button[role="tab"]:hover {
  color: var(--t2) !important;
  background: transparent !important;
}

[data-testid="stTabContent"] {
  padding: 24px !important;
  background: var(--bg) !important;
}

/* ── BUTTONS ─────────────────────────────────────────────────── */
.stButton button {
  background: transparent !important;
  border: 1px solid var(--b1) !important;
  color: var(--t3) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  border-radius: 4px !important;
  padding: 6px 14px !important;
  transition: border-color 0.12s, color 0.12s !important;
}
.stButton button:hover {
  border-color: var(--b3) !important;
  color: var(--t2) !important;
}

/* Confirm / primary variant */
.confirm-btn .stButton button {
  border: 1px solid var(--b3) !important;
  color: var(--green2) !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  min-height: 44px !important;
  width: 100% !important;
}
.confirm-btn .stButton button:hover {
  background: #0c1a0c !important;
  border-color: var(--green) !important;
}

/* Swap / micro button */
.swap-btn { flex-shrink: 0 !important; }
.swap-btn .stButton button {
  border: none !important;
  color: var(--t4) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 8px !important;
  font-weight: 500 !important;
  letter-spacing: 0.10em !important;
  text-transform: uppercase !important;
  padding: 2px 5px !important;
  background: transparent !important;
  white-space: nowrap !important;
  min-width: 36px !important;
  writing-mode: horizontal-tb !important;
  word-break: keep-all !important;
}
.swap-btn .stButton button:hover {
  color: var(--t2) !important;
  background: transparent !important;
  border: none !important;
}

/* Select button in swap panel */
.sel-btn .stButton button {
  border-color: var(--b2) !important;
  color: var(--t2) !important;
  font-size: 9px !important;
  padding: 3px 8px !important;
}

/* Edit picks small */
.edit-btn .stButton button {
  color: var(--t4) !important;
  font-size: 10px !important;
}

/* Refresh button */
.refresh-btn .stButton button {
  color: var(--t3) !important;
  font-size: 10px !important;
}

/* ── INPUTS ──────────────────────────────────────────────────── */
[data-testid="stTextInput"] input {
  background: var(--s1) !important;
  border: 1px solid var(--b1) !important;
  color: var(--t1) !important;
  font-size: 12px !important;
  border-radius: 4px !important;
}
[data-testid="stTextInput"] label {
  color: var(--t4) !important;
  font-size: 10px !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
}

/* ── SLIDERS ─────────────────────────────────────────────────── */
[data-testid="stSlider"] label {
  color: var(--t3) !important;
  font-size: 11px !important;
}
[data-testid="stSlider"] [data-testid="stSliderThumb"] {
  background: var(--green) !important;
}

/* ── RADIO AS PILLS ──────────────────────────────────────────── */
[data-testid="stRadio"] > div {
  display: flex !important;
  flex-direction: row !important;
  gap: 4px !important;
  flex-wrap: wrap !important;
}
[data-testid="stRadio"] label {
  display: inline-flex !important;
  align-items: center !important;
  border: 1px solid var(--b1) !important;
  padding: 3px 11px !important;
  font-size: 10px !important;
  font-weight: 500 !important;
  letter-spacing: 0.05em !important;
  text-transform: uppercase !important;
  color: var(--t3) !important;
  cursor: pointer !important;
  border-radius: 3px !important;
  background: transparent !important;
}
[data-testid="stRadio"] label:has(input:checked) {
  background: var(--b3) !important;
  border-color: var(--b3) !important;
  color: var(--t1) !important;
}
/* Hide the native radio input + any visual indicator, but NOT the text container */
[data-testid="stRadio"] input[type="radio"] {
  display: none !important;
}
/* Hide only the SVG/circle indicator element — keep the text p visible */
[data-testid="stRadio"] label > span:first-of-type,
[data-testid="stRadio"] label > div:first-of-type:not([data-testid="stMarkdownContainer"]) {
  display: none !important;
}
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
  font-size: 10px !important;
  font-weight: 500 !important;
  letter-spacing: 0.05em !important;
  text-transform: uppercase !important;
  color: inherit !important;
  margin: 0 !important;
  line-height: 1 !important;
}

/* ── EXPANDER ────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: transparent !important;
  border: 1px solid var(--b1) !important;
  border-radius: 4px !important;
  margin-bottom: 8px !important;
}
[data-testid="stExpander"] summary {
  color: var(--t3) !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  padding: 8px 12px !important;
}

/* ── FILE UPLOADER ───────────────────────────────────────────── */
[data-testid="stFileUploader"] {
  background: var(--s1) !important;
  border: 1px dashed var(--b2) !important;
  border-radius: 4px !important;
  padding: 8px !important;
}
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploaderDropzone"] {
  background: var(--s1) !important;
  border: 1px dashed var(--b2) !important;
  border-radius: 4px !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span { color: var(--t3) !important; font-size: 11px !important; }
/* Hide the default cloud upload icon */
[data-testid="stFileUploaderDropzone"] svg { display: none !important; }

/* ── ALERTS ──────────────────────────────────────────────────── */
[data-testid="stAlert"] {
  background: var(--s1) !important;
  border: 1px solid var(--b2) !important;
  border-radius: 4px !important;
}
[data-testid="stAlert"] p { color: var(--t2) !important; }

/* ── NUMBER INPUT ────────────────────────────────────────────── */
[data-testid="stNumberInput"] input {
  background: var(--bg) !important;
  border: 1px solid var(--b3) !important;
  color: var(--gold) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 13px !important;
  border-radius: 0 !important;
}
[data-testid="stNumberInput"] label {
  color: var(--t4) !important;
  font-size: 10px !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
}

/* ── CHECKBOX ────────────────────────────────────────────────── */
[data-testid="stCheckbox"] label { color: var(--t3) !important; font-size: 11px !important; }
[data-testid="stCheckbox"] span { background: var(--b2) !important; }

/* ── DATAFRAME ───────────────────────────────────────────────── */
[data-testid="stDataFrame"] th {
  background: var(--s1) !important;
  color: var(--t2b) !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  letter-spacing: 0.07em !important;
  text-transform: uppercase !important;
}
[data-testid="stDataFrame"] td {
  background: var(--bg) !important;
  color: var(--t2) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 11px !important;
  border-bottom: 1px solid var(--s1) !important;
}

/* ── HR ──────────────────────────────────────────────────────── */
hr { border-color: var(--b1) !important; }

/* ── CUSTOM UI COMPONENTS ────────────────────────────────────── */

/* Top bar */
.topbar {
  display: flex; align-items: center; justify-content: space-between;
  height: 50px; padding: 0 24px;
  background: var(--bg); border-bottom: 1px solid var(--b1);
  margin-bottom: 0;
}
.wordmark {
  font-family: 'Cormorant Garamond', serif !important;
  font-style: italic; font-weight: 300; font-size: 22px;
  color: var(--gold); letter-spacing: 0.02em;
}
.wordmark-sub {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 10px; font-weight: 300; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--t3); margin-left: 10px;
}
.status-bar { display: flex; align-items: center; gap: 16px; }
.status-item {
  display: flex; align-items: center; gap: 5px;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 10px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--t3);
}
.dot { width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; }
.dot-live { background: var(--green); }
.dot-est  { background: var(--gold); }
.dot-off  { background: var(--t4); }
.countdown {
  font-family: 'DM Mono', monospace !important;
  font-size: 11px; color: var(--gold); letter-spacing: 0.04em;
}

/* Team card grid wrapper */
.tc-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  width: 100%;
  box-sizing: border-box;
}

/* Team cards */
.tc {
  border: 1px solid var(--b1); background: var(--bg);
  padding: 14px 14px 10px;
  min-width: 0;          /* prevent grid blowout */
  overflow: hidden;
}
.tc.featured {
  border: 1px solid #2a3e2a;
  border-top: 2px solid #3aaa5a;
}
.tc-label {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 11px; font-weight: 500; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--t2);
}
.tc-strat {
  font-size: 10px; font-weight: 300; color: var(--t3);
  margin-top: 1px; margin-bottom: 8px;
}
.tc-ev {
  font-family: 'DM Mono', monospace !important;
  font-size: 10px; color: var(--green2); float: right; margin-top: -26px;
}
.ps { border-top: 1px solid var(--b1); padding: 7px 0; }
.ps-name {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 13px; color: var(--t1); line-height: 1.15;
}
.ps-name.lg { font-size: 14px; }
.ps-meta { font-size: 10px; color: var(--t3); margin-top: 1px; line-height: 1.2; }
.ps-stats {
  font-family: 'DM Mono', monospace !important;
  font-size: 10px; color: var(--t3); margin: 1px 0;
}
.badge {
  display: inline-block; padding: 1px 5px; border-radius: 2px;
  font-size: 7px; font-weight: 500; letter-spacing: 0.03em;
  text-transform: uppercase; margin-right: 2px; margin-top: 2px;
}
.br { border: 1px solid var(--red);    color: var(--red); }
.ba { border: 1px solid var(--gold2);  color: var(--gold); }
.bg { border: 1px solid var(--b3);     color: var(--t3); }
.tour-b {
  display: inline-block; padding: 0px 3px; border: 1px solid var(--b2);
  font-size: 7px; color: var(--t4); border-radius: 2px; margin-left: 3px;
}
.tour-b.liv { border-color: var(--azalea); color: var(--azalea); }
.tc-total { border-top: 1px solid var(--b1); padding-top: 12px; margin-top: 10px; }
.tc-total-lbl { font-size: 10px; color: var(--t3); letter-spacing: 0.08em; text-transform: uppercase; }
.tc-total-num {
  font-family: 'DM Mono', monospace !important;
  font-size: 16px; color: var(--t1);
}

/* Swap panel */
.swap-hdr {
  font-size: 9px; font-weight: 500; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--t3);
  padding-bottom: 8px; border-bottom: 1px solid var(--b1); margin-bottom: 8px;
}
.swap-r { border-bottom: 1px solid var(--s1); padding: 5px 0; }
.swap-n { font-size: 12px; color: var(--t1); }
.swap-s {
  font-family: 'DM Mono', monospace !important;
  font-size: 9px; color: var(--t3);
}
.swap-rnk {
  font-family: 'DM Mono', monospace !important;
  font-size: 10px; color: var(--t4); min-width: 20px;
}

/* Tiebreaker */
.tb-card {
  border: 1px solid var(--b2); background: var(--bg);
  padding: 20px; margin-top: 18px;
}
.tb-num {
  font-family: 'DM Mono', monospace !important;
  font-size: 32px; color: var(--gold); line-height: 1;
}
.tb-lbl { font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--t3); margin-top: 3px; }
.tb-note { font-size: 10px; color: var(--t3); }

/* Rankings */
.rk-hdr {
  display: grid;
  grid-template-columns: 36px 1fr 72px 88px 52px 52px 52px 64px 68px minmax(80px,1fr);
  gap: 0; padding: 6px 8px;
  border-bottom: 1px solid var(--b1); background: var(--s1);
}
.rk-hdr-cell {
  font-size: 11px; font-weight: 500; letter-spacing: 0.07em;
  text-transform: uppercase; color: var(--t2b);
}
details.rk-row { border-bottom: 1px solid var(--s1); }
details.rk-row > summary {
  display: grid;
  grid-template-columns: 36px 1fr 72px 88px 52px 52px 52px 64px 68px minmax(80px,1fr);
  gap: 0; padding: 7px 8px; cursor: pointer; list-style: none;
}
details.rk-row > summary::-webkit-details-marker { display: none; }
details.rk-row > summary:hover { background: var(--s1); }
details.rk-row[open] > summary { background: var(--s2); }
.rk-num {
  font-family: 'DM Mono', monospace !important;
  font-size: 12px; color: var(--t4); align-self: center;
}
.rk-num.g { color: var(--gold); font-weight: 500; }
.rk-nm { font-size: 13px; color: var(--t1); }
.rk-meta { font-size: 10px; color: var(--t3); margin-top: 1px; }
.ev-hi { font-family: 'DM Mono', monospace !important; font-size: 12px; color: var(--green2); }
.ev-md { font-family: 'DM Mono', monospace !important; font-size: 12px; color: var(--t2); }
.ev-lo { font-family: 'DM Mono', monospace !important; font-size: 12px; color: var(--t4); }
.sbg { width: 48px; height: 2px; background: var(--b2); display: inline-block; vertical-align: middle; }
.op { font-family: 'DM Mono', monospace !important; font-size: 11px; color: var(--t2); }
.on { font-family: 'DM Mono', monospace !important; font-size: 11px; color: var(--green2); }
.ow { font-family: 'DM Mono', monospace !important; font-size: 11px; color: var(--t3); }
.ow-hi { font-family: 'DM Mono', monospace !important; font-size: 11px; color: var(--gold); }
.rk-exp {
  padding: 12px 8px 12px 44px;
  background: var(--s2); border-top: 1px solid var(--b1);
  display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px;
}
.exp-sec-hdr { font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--t3); margin-bottom: 5px; }
.exp-val { font-family: 'DM Mono', monospace !important; font-size: 11px; color: var(--t2); }
.exp-lbl { font-size: 10px; color: var(--t3); }

/* Chaos banner */
.chaos-banner {
  background: #0e0a00; border-left: 3px solid #3a2a00;
  border-bottom: 1px solid #3a2a00;
  padding: 7px 16px; margin-bottom: 12px; font-size: 10px; color: var(--gold);
}
.chaos-banner b { color: var(--gold); font-weight: 600; }

/* Confirmed picks view */
.conf-cdown {
  font-family: 'DM Mono', monospace !important;
  font-size: 26px; color: var(--gold); letter-spacing: 0.04em;
  text-align: center; padding: 28px 0 6px;
}
.conf-cdown-lbl {
  font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--t3); text-align: center; margin-bottom: 20px;
}

/* Section header */
.sec-hdr {
  font-size: 11px; font-weight: 500; letter-spacing: 0.10em;
  text-transform: uppercase; color: var(--t2b);
  margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid var(--b1);
}

/* Hero rank */
.hero-rank {
  font-family: 'DM Mono', monospace !important;
  font-size: 64px; color: var(--green2); line-height: 1;
}
.hero-of {
  font-family: 'DM Mono', monospace !important;
  font-size: 18px; color: var(--t3);
}
.hero-lbl { font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--t3); margin-bottom: 4px; }
.hero-payout {
  font-family: 'DM Mono', monospace !important;
  font-size: 22px; color: var(--gold);
}

/* Leaderboard */
.lb-hdr {
  display: grid;
  grid-template-columns: 48px 1fr 70px 70px 70px 1fr;
  gap: 0; padding: 6px 8px;
  border-bottom: 1px solid var(--b1); background: var(--s1);
}
.lb-hdr-c {
  font-size: 11px; font-weight: 500; letter-spacing: 0.07em;
  text-transform: uppercase; color: var(--t2b);
}
.lb-row {
  display: grid;
  grid-template-columns: 48px 1fr 70px 70px 70px 1fr;
  gap: 0; padding: 7px 8px;
  border-bottom: 1px solid var(--s1); align-items: center;
}
.lb-row.mine { background: #040d04; }
.lb-row:hover { background: var(--s1); }
.lb-pos { font-family: 'DM Mono', monospace !important; font-size: 12px; color: var(--t4); }
.lb-nm { font-size: 13px; color: var(--t1); }
.lb-nm.mine { color: var(--green2); }
.lb-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--green); display: inline-block; margin-right: 5px; vertical-align: middle; }
.sc-u { font-family: 'DM Mono', monospace !important; font-size: 12px; color: var(--green2); }
.sc-o { font-family: 'DM Mono', monospace !important; font-size: 12px; color: var(--red); }
.sc-e { font-family: 'DM Mono', monospace !important; font-size: 12px; color: var(--t2); }
.entry-tag {
  display: inline-block; padding: 0px 4px; border: 1px solid var(--b2);
  border-radius: 2px; font-size: 8px; color: var(--t3); margin-right: 2px;
}

/* ── MODEL TAB ───────────────────────────────────────────────── */

/* Component toggle buttons — minimal, no background/border chrome */
button[data-testid="baseButton-secondary"]:has(+ div),
div[data-testid="stColumn"] button[kind="secondary"] {
  background: transparent !important;
  border: 1px solid #152015 !important;
  color: #426842 !important;
  font-size: 11px !important;
  padding: 6px 4px !important;
  border-radius: 3px !important;
  line-height: 1 !important;
  min-height: 0 !important;
  height: 38px !important;
}
button[data-testid="baseButton-secondary"]:hover {
  border-color: #3aaa5a !important;
  color: #7aaa7a !important;
}

/* Section header shared */
.mdl-sec {
  font-size: 11px; font-weight: 500; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--t2b);
  border-bottom: 1px solid var(--b1); padding-bottom: 6px; margin-bottom: 16px;
}

/* Metric cards row */
.mdl-metrics {
  display: flex; gap: 12px; margin-bottom: 20px;
}
.mdl-metric {
  flex: 1; border: 1px solid var(--b1); background: var(--s1);
  padding: 14px 16px;
}
.mdl-metric-val {
  font-family: 'DM Mono', monospace !important;
  font-size: 22px; color: var(--green2); line-height: 1; margin-bottom: 3px;
}
.mdl-metric-val.warn { color: var(--gold); }
.mdl-metric-val.bad  { color: var(--red); }
.mdl-metric-lbl { font-size: 11px; letter-spacing: 0.07em; text-transform: uppercase; color: var(--t2b); }

/* Player detail panel */
.pdp-header {
  border-bottom: 1px solid var(--b1); padding-bottom: 10px; margin-bottom: 14px;
}
.pdp-name {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 20px; color: var(--t1); line-height: 1.1;
}
.pdp-meta { font-size: 11px; color: var(--t3); margin-top: 3px; letter-spacing: 0.04em; }
.pdp-score {
  font-family: 'DM Mono', monospace !important;
  font-size: 40px; color: var(--gold); line-height: 1; margin: 10px 0 4px;
}
.pdp-score-lbl { font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--t3); }

/* Component bar rows */
.comp-row {
  display: grid;
  grid-template-columns: 90px 1fr 48px 60px;
  gap: 8px; align-items: center; margin-bottom: 8px;
}
.comp-lbl { font-size: 10px; color: var(--t3); }
.comp-bar-bg { height: 4px; background: var(--b2); border-radius: 0; overflow: hidden; }
.comp-wt  { font-family: 'DM Mono', monospace !important; font-size: 10px; color: var(--t3); text-align: right; }
.comp-pts { font-family: 'DM Mono', monospace !important; font-size: 10px; color: var(--t2); text-align: right; }

/* Augusta history year boxes */
.aug-boxes { display: flex; gap: 4px; flex-wrap: wrap; margin-top: 6px; }
.aug-box {
  padding: 4px 7px; font-family: 'DM Mono', monospace !important;
  font-size: 9px; text-align: center; min-width: 44px;
}
.aug-box-yr  { font-size: 7px; color: var(--t4); letter-spacing: 0.04em; }
.aug-box-fin { font-size: 11px; font-weight: 500; }

/* Driver bullets */
.driver-list { list-style: none; padding: 0; margin: 0; }
.driver-list li {
  font-size: 10px; color: var(--t3); padding: 3px 0;
  border-bottom: 1px solid var(--s2); line-height: 1.4;
}
.driver-list li:last-child { border-bottom: none; }
.driver-pos { color: var(--green2); }
.driver-neg { color: var(--red); }

/* Insight cards */
.ins-card {
  border: 1px solid var(--b2); background: var(--s1);
  padding: 12px 14px; margin-bottom: 8px;
}
.ins-card.warn  { border-left: 3px solid var(--gold); }
.ins-card.alert { border-left: 3px solid var(--red); }
.ins-card.good  { border-left: 3px solid var(--green); }
.ins-card-hdr   { font-size: 11px; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: var(--t2b); margin-bottom: 4px; }
.ins-card-body  { font-size: 11px; color: var(--t2); line-height: 1.5; }

/* Player selector list */
.psel-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 6px 8px; cursor: pointer; border-bottom: 1px solid var(--s1);
}
.psel-row:hover { background: var(--s1); }
.psel-row.active { background: var(--s2); border-left: 2px solid var(--green); }
.psel-nm  { font-size: 12px; color: var(--t1); }
.psel-ev  { font-family: 'DM Mono', monospace !important; font-size: 10px; color: var(--green2); }
.psel-aug { font-family: 'DM Mono', monospace !important; font-size: 10px; color: var(--t3); }

/* ── GLOBAL READABILITY BACKSTOPS ───────────────────────────── */
/* Minimum readable size for all rendered markdown text */
.stMarkdown p { font-size: 12px !important; line-height: 1.5 !important; }
.stMarkdown, .stText { color: #96cc96 !important; }

/* Tab content minimum */
[data-testid="stTabContent"] p  { font-size: 11px !important; }
[data-testid="stTabContent"] li { font-size: 11px !important; }
</style>
"""

# ─────────────────────────────────────────────────────────────────
# HELPER — UTILITIES
# ─────────────────────────────────────────────────────────────────

def safe_html(val) -> str:
    """Escape a value for safe insertion into HTML attributes and text nodes."""
    if not isinstance(val, str):
        return val
    return (
        val.replace("&", "&amp;")
           .replace("<", "&lt;")
           .replace(">", "&gt;")
           .replace('"', "&quot;")
           .replace("'", "&#39;")
    )


def _fmt_odds(odds) -> str:
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return "N/A"
    if odds == 0:
        return "N/A"
    return f"+{int(odds)}" if odds > 0 else str(int(odds))


def _odds_html(odds) -> str:
    fmt = _fmt_odds(odds)
    try:
        cls = "on" if float(odds) < 0 else "op"
    except (TypeError, ValueError):
        cls = "op"
    return f'<span class="{cls}">{fmt}</span>'


def _countdown_text() -> str:
    now = datetime.now(timezone.utc)
    # April 9 2026 12:00 UTC = 08:00 ET
    target = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
    delta = target - now
    if delta.total_seconds() <= 0:
        return "LIVE NOW"
    d = delta.days
    h = delta.seconds // 3600
    return f"{d}d {h}h"


def _bar_color(pct: float) -> str:
    if pct >= 70:
        return "var(--green)"
    if pct >= 40:
        return "var(--t3)"
    return "var(--b3)"


def _micro_bars(form: float, dna: float, fit: float) -> str:
    """Three 36px stat mini-bars for Form / DNA / Fit."""
    parts = []
    for lbl, val in (("FORM", form), ("DNA", dna), ("FIT", fit)):
        pct = min(100.0, max(0.0, float(val or 0)))
        clr = _bar_color(pct)
        parts.append(
            f'<div style="display:flex;align-items:center;gap:3px;">'
            f'<span style="font-size:8px;color:var(--t4);width:24px;flex-shrink:0;letter-spacing:0.04em;">{lbl}</span>'
            f'<div style="width:36px;height:3px;background:var(--b2);">'
            f'<div style="width:{pct:.0f}%;height:100%;background:{clr};"></div>'
            f'</div></div>'
        )
    return '<div style="display:flex;flex-direction:column;gap:3px;margin-top:3px;">' + "".join(parts) + "</div>"


def _flag_badges(flags_str: str, chalk: bool = False) -> str:
    """Show only the highest-priority flag + '+N more' indicator.
    Priority order: Injury/Concern > stat threshold flags > CHALK > DNA gap flags.
    """
    # Human-readable short labels for known flag keys
    _LABEL_MAP: dict[str, str] = {
        "SG App < +0.84":   "APPROACH",
        "SG App < +0":      "APPROACH",
        "SG OTT < +0.60":   "OFF TEE",
        "SG OTT < +0":      "OFF TEE",
        "SG Total < +0.67": "SG TOTAL",
        "Rank > 25":        "WORLD RANK",
        "< 4 Career":       "FEW WINS",
        "< 4 CAREER":       "FEW WINS",
        "No Tune-Up":       "TUNE-UP",
        "NO TUNE-UP":       "TUNE-UP",
        "DNA/FORM GAP":     "DNA/FORM GAP",
    }

    STAT_FLAGS = {"Rank > 25", "SG Total < +0.67", "SG App < +0.84", "SG OTT < +0.60",
                  "SG App < +0", "SG OTT < +0"}

    # Collect all flags in priority order
    all_flags: list[tuple[int, str, str]] = []  # (priority, css_class, display_label)

    if flags_str:
        for f in flags_str.split(";"):
            f = f.strip()
            if not f:
                continue
            display = _LABEL_MAP.get(f, f)
            if "Injury" in f or "Concern" in f:
                all_flags.append((0, "br", display))
            elif f in STAT_FLAGS or any(k in f for k in ("SG App", "SG OTT", "SG Total", "Rank >")):
                display = _LABEL_MAP.get(f, (f[:11] + "..") if len(f) > 13 else f)
                all_flags.append((1, "br", display))
            else:
                # Truncate anything still long
                if len(display) > 13:
                    display = display[:11] + ".."
                all_flags.append((3, "ba", display))

    if chalk:
        all_flags.append((2, "ba", "CHALK"))

    if not all_flags:
        return ""

    # Sort by priority (lowest number = highest priority)
    all_flags.sort(key=lambda x: x[0])

    _, cls, label = all_flags[0]
    out = f'<span class="badge {cls}">{label}</span>'

    extras = len(all_flags) - 1
    if extras > 0:
        out += f'<span class="badge bg">+{extras}</span>'

    return out


def _score_bar(score: float, max_s: float = 100.0) -> str:
    pct = min(100.0, max(0.0, score / max(max_s, 1) * 100))
    clr = _bar_color(pct)
    return (
        f'<div style="display:flex;align-items:center;gap:5px;">'
        f'<div class="sbg"><div style="width:{pct:.0f}%;height:100%;background:{clr};"></div></div>'
        f'<span style="font-family:\'DM Mono\',monospace;font-size:11px;color:var(--t2);">{score:.1f}</span>'
        f'</div>'
    )


def _tour_badge(tour: str) -> str:
    cls = "tour-b liv" if tour == "LIV" else "tour-b"
    return f'<span class="{cls}">{tour}</span>'


def _ev_class(ev: float) -> str:
    if ev >= 27:
        return "ev-hi"
    if ev >= 22:
        return "ev-md"
    return "ev-lo"


def _dot_class(src: str) -> str:
    if src in ("espn", "odds_api", "open-meteo", "cache"):
        return "dot-live"
    if src == "fallback":
        return "dot-est"
    return "dot-off"


def _src_label(src: str) -> str:
    if src in ("espn", "odds_api", "open-meteo", "cache"):
        return "LIVE"
    if src == "fallback":
        return "EST"
    return "OFF"


# ─────────────────────────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────────────────────────

def render_topbar(data: dict):
    stats_src   = data.get("stats", {}).get("_meta", {}).get("source", "?")
    odds_src    = data.get("odds",  {}).get("_meta", {}).get("source", "?")
    weather_src = data.get("weather", {}).get("_meta", {}).get("source", "?")
    cdown = _countdown_text()
    st.markdown(f"""
<div class="topbar">
  <div>
    <span class="wordmark">Augusta</span>
    <span class="wordmark-sub">Pool Intelligence &middot; 2026</span>
  </div>
  <div class="status-bar">
    <div class="status-item">
      <span class="dot {_dot_class(odds_src)}"></span>ODDS {_src_label(odds_src)}
    </div>
    <div class="status-item">
      <span class="dot {_dot_class(weather_src)}"></span>WEATHER {_src_label(weather_src)}
    </div>
    <div class="status-item">
      <span class="dot {_dot_class(stats_src)}"></span>STATS {_src_label(stats_src)}
    </div>
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:1px;">
      <span class="countdown">{cdown}</span>
      <span style="font-size:6px;letter-spacing:0.14em;text-transform:uppercase;color:var(--t4);font-family:'DM Sans',sans-serif;">TO TEE-OFF</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# TEAM DATA — ENRICHMENT & SWAP LOGIC
# ─────────────────────────────────────────────────────────────────

def _enrich_team(team: dict, df: pd.DataFrame) -> dict:
    """Add Form/DNA/Fit/Trajectory/Flags/Tour to team stats rows."""
    enriched = []
    for stat in team.get("stats", []):
        name = stat["Player"]
        row = df[df["Player"] == name]
        if not row.empty:
            r = row.iloc[0]
            enriched.append({
                **stat,
                "Form_Score":       float(r.get("Form_Score", 50)),
                "DNA_Score":        float(r.get("DNA_Score", 50)),
                "Fit_Score":        float(r.get("Fit_Score", 50)),
                "Vegas_Score":      float(r.get("Vegas_Score", 50)),
                "Trajectory_Score": float(r.get("Trajectory_Score", 50)),
                "Flags":            str(r.get("Flags", "")),
                "Chalk_Penalty":    bool(r.get("Chalk_Penalty", False)),
                "Tour":             str(r.get("Tour", "PGA")),
            })
        else:
            enriched.append({**stat, "Form_Score": 50, "DNA_Score": 50,
                             "Fit_Score": 50, "Trajectory_Score": 50,
                             "Flags": "", "Chalk_Penalty": False, "Tour": "PGA"})
    result = dict(team)
    result["stats"] = enriched
    return result


def _do_swap(team_key: str, slot: int, new_player: str, df: pd.DataFrame):
    """Replace player at slot in a custom team and update totals."""
    custom = st.session_state.get("custom_teams", {})
    team = custom.get(team_key, {})
    stats = list(team.get("stats", []))
    if slot >= len(stats):
        return
    row = df[df["Player"] == new_player]
    if row.empty:
        return
    r = row.iloc[0]
    stats[slot] = {
        "Player":           new_player,
        "Augusta_Score":    float(r["Augusta_Score"]),
        "EV_Score":         float(r["EV_Score"]),
        "Ownership_Pct":    float(r["Ownership_Pct"]),
        "Odds_American":    float(r.get("Odds_American", 50000)),
        "World_Rank":       int(r.get("World_Rank", 999)),
        "Form_Score":       float(r.get("Form_Score", 50)),
        "DNA_Score":        float(r.get("DNA_Score", 50)),
        "Fit_Score":        float(r.get("Fit_Score", 50)),
        "Vegas_Score":      float(r.get("Vegas_Score", 50)),
        "Trajectory_Score": float(r.get("Trajectory_Score", 50)),
        "Flags":            str(r.get("Flags", "")),
        "Chalk_Penalty":    bool(r.get("Chalk_Penalty", False)),
        "Tour":             str(r.get("Tour", "PGA")),
    }
    team["stats"]               = stats
    team["players"]             = [s["Player"] for s in stats]
    team["total_augusta_score"] = round(sum(s["Augusta_Score"] for s in stats), 1)
    team["total_ev_score"]      = round(sum(s["EV_Score"] for s in stats), 1)
    custom[team_key]            = team
    st.session_state["custom_teams"]    = custom
    st.session_state["swap_open"]       = False
    st.rerun()


def _confirm_lock_swap(team_key: str, slot: int, new_player: str, original_player: str, df: pd.DataFrame):
    """Apply a confirmed swap and mark the slot as locked (shows badge + enables undo)."""
    custom = st.session_state.get("custom_teams", {})
    team = custom.get(team_key, {})
    stats = list(team.get("stats", []))
    if slot >= len(stats):
        return
    row = df[df["Player"] == new_player]
    if row.empty:
        return
    r = row.iloc[0]
    stats[slot] = {
        "Player":           new_player,
        "Augusta_Score":    float(r["Augusta_Score"]),
        "EV_Score":         float(r["EV_Score"]),
        "Ownership_Pct":    float(r["Ownership_Pct"]),
        "Odds_American":    float(r.get("Odds_American", 50000)),
        "World_Rank":       int(r.get("World_Rank", 999)),
        "Form_Score":       float(r.get("Form_Score", 50)),
        "DNA_Score":        float(r.get("DNA_Score", 50)),
        "Fit_Score":        float(r.get("Fit_Score", 50)),
        "Vegas_Score":      float(r.get("Vegas_Score", 50)),
        "Trajectory_Score": float(r.get("Trajectory_Score", 50)),
        "Flags":            str(r.get("Flags", "")),
        "Chalk_Penalty":    bool(r.get("Chalk_Penalty", False)),
        "Tour":             str(r.get("Tour", "PGA")),
        "_locked":          True,
        "_was":             original_player,
    }
    team["stats"]               = stats
    team["players"]             = [s["Player"] for s in stats]
    team["total_augusta_score"] = round(sum(s["Augusta_Score"] for s in stats), 1)
    team["total_ev_score"]      = round(sum(s["EV_Score"] for s in stats), 1)
    custom[team_key]            = team
    st.session_state["custom_teams"] = custom
    # Record the lock for undo
    st.session_state["locked_swaps"][team_key][slot] = {
        "in":  new_player,
        "out": original_player,
    }


def _undo_lock_swap(team_key: str, slot: int, df: pd.DataFrame):
    """Restore the original model pick for a locked slot."""
    lock_info = st.session_state["locked_swaps"].get(team_key, {}).get(slot)
    if not lock_info:
        return
    original = lock_info["out"]
    # Restore via _do_swap (no _locked flag → badge disappears)
    _do_swap(team_key, slot, original, df)
    # _do_swap calls st.rerun() — remove lock entry before that
    st.session_state["locked_swaps"][team_key].pop(slot, None)


# ─────────────────────────────────────────────────────────────────
# SWAP PANEL
# ─────────────────────────────────────────────────────────────────

def render_swap_panel(df: pd.DataFrame):
    swap_player = st.session_state.get("swap_player", "Unknown")
    swap_team   = st.session_state.get("swap_team", "team_a")
    swap_slot   = st.session_state.get("swap_slot", 0)

    current_players = [
        s["Player"]
        for s in st.session_state.get("custom_teams", {}).get(swap_team, {}).get("stats", [])
    ]

    # Header row with close
    hdr_c, close_c = st.columns([6, 1])
    with hdr_c:
        st.markdown(
            f'<div class="swap-hdr">ALTERNATIVES FOR {swap_player.upper()}</div>',
            unsafe_allow_html=True,
        )
    with close_c:
        if st.button("X", key="close_swap_panel"):
            st.session_state["swap_open"] = False
            st.rerun()

    # Alternatives: exclude current team members, sort by Augusta_Score
    alts = (
        df[~df["Player"].isin(current_players)]
        .nlargest(12, "Augusta_Score")
        .reset_index(drop=True)
    )

    for idx, row in alts.iterrows():
        name    = row["Player"]
        aug_s   = row["Augusta_Score"]
        ev_s    = row["EV_Score"]
        odds    = row.get("Odds_American", 0)
        flags   = str(row.get("Flags", ""))
        flag_short = flags.split(";")[0].strip()[:18] if flags else ""

        n_c, info_c, sel_c = st.columns([1, 6, 2])
        with n_c:
            st.markdown(f'<span class="swap-rnk">{idx+1}</span>', unsafe_allow_html=True)
        with info_c:
            st.markdown(
                f'<div class="swap-r">'
                f'<div class="swap-n">{name}</div>'
                f'<span class="swap-s">{aug_s:.1f} &nbsp; EV {ev_s:.1f} &nbsp; {_fmt_odds(odds)}</span>'
                + (f'<br><span style="font-size:8px;color:var(--t4);">{flag_short}</span>' if flag_short else "")
                + "</div>",
                unsafe_allow_html=True,
            )
        with sel_c:
            st.markdown('<div class="sel-btn">', unsafe_allow_html=True)
            key = f"sel_{swap_team}_{swap_slot}_{name[:10].replace(' ','_')}"
            if st.button("SELECT", key=key):
                _do_swap(swap_team, swap_slot, name, df)
            st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# TIEBREAKER SECTION
# ─────────────────────────────────────────────────────────────────

def render_tiebreaker(data: dict):
    weather = data.get("weather", {})
    chaos_mode = weather.get("chaos_mode", False)

    # Pool-calibrated tiebreaker (primary)
    pool_tb = compute_pool_tiebreaker(weather, chaos_mode)
    rec = pool_tb["recommended"]
    rationale = pool_tb["rationale"]
    clusters = pool_tb["pool_clusters_to_beat"]
    hist_note = pool_tb["historical_note"]

    # Model tiebreaker (secondary — for wind / confidence display)
    model_tb = predict_tiebreaker(weather)
    cond = model_tb.get("condition_label", "Mild")
    wind = model_tb.get("avg_wind_mph", 0)

    chaos_badge = ""
    if chaos_mode:
        chaos_badge = (
            '<span style="font-size:8px;font-weight:700;letter-spacing:.08em;'
            'padding:2px 6px;border-radius:2px;background:#2a0a0a;color:#cc4a4a;'
            'margin-left:8px;">CHAOS</span>'
        )

    clusters_str = " / ".join(str(c) for c in clusters)
    saved_tb = st.session_state.get("tiebreaker_value", rec)

    st.markdown(
        f'<div class="tb-card"><div style="display:flex;align-items:flex-start;gap:32px;">',
        unsafe_allow_html=True,
    )

    tb_l, tb_m, tb_r = st.columns([1, 2, 1])
    with tb_l:
        st.markdown(
            f'<div class="tb-num">{rec}{chaos_badge}</div>'
            f'<div class="tb-lbl">POOL TIEBREAKER — BEATS {clusters_str} CLUSTER</div>',
            unsafe_allow_html=True,
        )
    with tb_m:
        st.markdown(
            f'<div style="font-size:11px;color:var(--t2b);letter-spacing:.06em;'
            f'text-transform:uppercase;margin-bottom:6px;">POOL CLUSTER STRATEGY</div>'
            f'<div style="font-size:11px;color:var(--t2);line-height:1.5;">{rationale}</div>'
            f'<div style="font-size:10px;color:var(--t3);margin-top:6px;line-height:1.4;">'
            f'{hist_note}</div>',
            unsafe_allow_html=True,
        )
    with tb_r:
        st.markdown(
            f'<div class="tb-note">{cond} &nbsp;&middot;&nbsp; {wind:.0f} mph wind</div>',
            unsafe_allow_html=True,
        )
        new_tb = st.number_input(
            "YOUR TIEBREAKER",
            value=int(saved_tb),
            min_value=-100,
            max_value=0,
            step=1,
            key="tb_input",
        )
        st.session_state["tiebreaker_value"] = new_tb

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# CONFIRMED PICKS VIEW
# ─────────────────────────────────────────────────────────────────

def render_confirmed_view():
    cdown = _countdown_text()
    confirmed = st.session_state.get("confirmed_picks", {})
    teams = confirmed.get("teams", {})
    tiebreaker = confirmed.get("tiebreaker", "?")

    st.markdown(
        f'<div class="conf-cdown">{cdown}</div>'
        f'<div class="conf-cdown-lbl">until first tee · Thursday, April 9, 2026</div>',
        unsafe_allow_html=True,
    )

    # Summary table
    rows_html = ""
    for tk, label in [("team_a", "FLOOR"), ("team_b", "BALANCED"), ("team_c", "CONTRARIAN")]:
        t = teams.get(tk, {})
        players = [s.get("Player", "?") for s in t.get("stats", [])]
        player_str = " &nbsp;/&nbsp; ".join(players)
        rows_html += (
            f'<tr>'
            f'<td style="font-size:11px;color:#7aaa7a;letter-spacing:.07em;padding:7px 10px;">{label}</td>'
            f'<td style="font-family:\'DM Sans\',sans-serif;font-size:12px;color:var(--t1);padding:7px 10px;">{player_str}</td>'
            f'</tr>'
        )
    rows_html += (
        f'<tr>'
        f'<td style="font-size:11px;color:#7aaa7a;letter-spacing:.07em;padding:7px 10px;">TIEBREAKER</td>'
        f'<td style="font-family:\'DM Mono\',monospace;font-size:12px;color:var(--gold);padding:7px 10px;">{tiebreaker}</td>'
        f'</tr>'
    )
    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;border:1px solid var(--b1);margin-top:0;">'
        f'<thead><tr>'
        f'<th style="font-size:11px;font-weight:500;color:#7aaa7a;padding:6px 10px;border-bottom:1px solid var(--b1);text-align:left;">TEAM</th>'
        f'<th style="font-size:11px;font-weight:500;color:#7aaa7a;padding:6px 10px;border-bottom:1px solid var(--b1);text-align:left;">PLAYERS</th>'
        f'</tr></thead><tbody>'
        + rows_html
        + "</tbody></table>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="edit-btn">', unsafe_allow_html=True)
    if st.button("EDIT PICKS", key="edit_picks_btn"):
        st.session_state["picks_confirmed"] = False
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# TAB 1 — MY PICKS
# ─────────────────────────────────────────────────────────────────

PLAYER_DESCRIPTIONS: dict[str, str] = {
    "Scottie Scheffler":   "WR#1 chalk anchor — Augusta's most consistent performer",
    "Bryson DeChambeau":   "2 consecutive LIV wins — hottest player entering Augusta",
    "Jon Rahm":            "2023 champion — dominant LIV form, elite Augusta DNA",
    "Rory McIlroy":        "Defending champion — proven closer, back concerns noted",
    "Tommy Fleetwood":     "8 straight Augusta paydays — FedEx Cup champ, underowned",
    "Ludvig Aberg":        "T2 on Masters debut 2024 — elite approach play, overlooked",
    "Xander Schauffele":   "Only player top-10 last 3 events — in form, underowned",
    "Matt Fitzpatrick":    "Won Valspar — elite approach, solid Augusta history",
    "Brooks Koepka":       "Major specialist — 2× runner-up Augusta, strong DNA",
    "Justin Thomas":       "Augusta pedigree — consistent top-25 finisher",
    "Patrick Reed":        "2018 champion — DNA strong, but form concern flagged",
    "Collin Morikawa":     "Back injury concern — day-by-day status, elite approach",
    "Cameron Smith":       "Augusta T2 pedigree — LIV form steady, underowned",
}


def _description_for_player(player_name: str, pstat: dict) -> str:
    """Return a one-line reason why this player is on the team."""
    if player_name in PLAYER_DESCRIPTIONS:
        return PLAYER_DESCRIPTIONS[player_name]

    form  = float(pstat.get("Form_Score",  0))
    dna   = float(pstat.get("DNA_Score",   0))
    fit   = float(pstat.get("Fit_Score",   0))
    ev    = float(pstat.get("EV_Score",    0))
    own   = float(pstat.get("Ownership_Pct", 5))
    odds  = float(pstat.get("Odds_American", 9999))

    if form >= 90 and own <= 10:
        return "Elite form, underowned — strong EV play"
    if form >= 75 and own >= 35:
        return "Chalk anchor — top form justifies the ownership"
    if form >= 85:
        return "Peak form entering Augusta — hot streak continues"
    if dna >= 95:
        return "Proven Augusta winner — DNA score leads the field"
    if dna >= 88 and form >= 55:
        return "Augusta specialist — elite course history"
    if fit >= 75 and odds >= 2000:
        return "Undervalued course fit — par-5 scoring edge"
    if ev >= 30 and own <= 5:
        return "High EV value — pool significantly underowns"
    if 55 <= form <= 75 and fit >= 70:
        return "Balanced profile — consistent Augusta performer"
    return "Strong composite — top-20 Augusta Score"


def _build_slot_html(pstat: dict, team_composite: float = 0.0, team_color: str = "#3AAA5A") -> str:
    """Build HTML for a single player slot using the spec's exact structure."""
    name  = safe_html(pstat.get("Player", "?"))
    wr    = pstat.get("World_Rank", 999)
    tour  = safe_html(pstat.get("Tour", "PGA"))
    score = float(pstat.get("Augusta_Score", 0))
    odds  = _fmt_odds(pstat.get("Odds_American", 0))
    form  = min(100, max(0, float(pstat.get("Form_Score",       50))))
    dna   = min(100, max(0, float(pstat.get("DNA_Score",        50))))
    fit   = min(100, max(0, float(pstat.get("Fit_Score",        50))))
    vegas = min(100, max(0, float(pstat.get("Vegas_Score",      50))))
    traj  = min(100, max(0, float(pstat.get("Trajectory_Score", 50))))

    # Ownership band badge — inline with player name
    own_pct = float(pstat.get("Ownership_Pct", 5.0))
    if own_pct > 35:
        own_band, own_bg, own_fg, own_border = "CHALK", "#1a1400", "#c8a84a", "#2a2000"
    elif own_pct >= 10:
        own_band, own_bg, own_fg, own_border = "MID",   "#1a2a1a", "#96cc96", "#263826"
    elif own_pct >= 3:
        own_band, own_bg, own_fg, own_border = "VALUE", "#0a1a0a", "#52cc72", "#1e301e"
    else:
        own_band, own_bg, own_fg, own_border = "FIELD", "#1e301e", "#4a6a4a", "#1e301e"
    own_badge = (
        f'<span style="font-size:10px;font-weight:700;letter-spacing:.05em;'
        f'padding:2px 7px;border-radius:3px;border:1px solid {own_border};'
        f'background:{own_bg};color:{own_fg};margin-left:8px;flex-shrink:0;'
        f'vertical-align:middle;white-space:nowrap;">'
        f'{own_band} ~{own_pct:.0f}%</span>'
    )

    # Lock badge — shown only when this slot has been manually locked
    is_locked = bool(pstat.get("_locked", False))
    was_player = safe_html(str(pstat.get("_was", "")))
    if is_locked:
        lock_badge = (
            '<span style="font-size:9px;font-weight:700;'
            'padding:2px 6px;border-radius:3px;'
            'background:#1a1200;color:#c8a84a;'
            'border:1px solid #3a2a00;margin-left:6px;'
            'flex-shrink:0;vertical-align:middle;white-space:nowrap;">'
            '🔒 LOCKED</span>'
        )
        was_html = (
            f'<span style="font-size:10px;color:#5a8a5a;'
            f'margin-left:4px;font-style:italic;white-space:nowrap;">'
            f'(was {was_player})</span>'
        ) if was_player else ""
    else:
        lock_badge = ""
        was_html = ""

    def bar_color(val: float) -> str:
        if val >= 65: return "#52CC72"
        if val >= 40: return "#8ABD6A"
        return "#3A5A3A"

    def mini_bar(label: str, val: float, tooltip: str = "") -> str:
        color = bar_color(val)
        title_attr = f' title="{tooltip}"' if tooltip else ""
        return (
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">'
            f'<div style="font-size:11px;color:#7aaa7a;letter-spacing:.06em;text-transform:uppercase;'
            f'width:44px;min-width:44px;flex-shrink:0;cursor:default;"{title_attr}>{label}</div>'
            f'<div style="flex:1;min-width:0;height:10px;background:#1a2a1a;border-radius:1px;overflow:hidden;">'
            f'<div style="width:{val:.0f}%;height:100%;background:{color};border-radius:1px;"></div>'
            f'</div>'
            f'<div style="font-family:DM Mono,monospace;font-size:11px;color:{color};'
            f'width:26px;min-width:26px;flex-shrink:0;text-align:right;">{val:.0f}</div>'
            f'</div>'
        )

    # Tour badge colors
    if tour == "LIV":
        t_color, t_bg = "#c4783a", "#150e05"
    else:
        t_color, t_bg = "#2a422a", "#1e2e1e"

    # Player contribution bar
    if team_composite > 0:
        contrib_pct = min(100.0, (score / team_composite) * 100)
    else:
        contrib_pct = 25.0
    contrib_bar = (
        f'<div style="width:100%;height:4px;background:#0a120a;margin-top:10px;">'
        f'<div style="width:{contrib_pct:.1f}%;height:100%;background:{team_color};opacity:0.7;"></div>'
        f'</div>'
    )

    # Score math line: weighted components → Augusta Score
    score_math_html = (
        f'<div style="font-family:\'DM Mono\',monospace;font-size:10px;color:#5a8a5a;'
        f'margin-top:6px;padding:5px 0;border-top:1px solid #1a2a1a;'
        f'letter-spacing:.01em;line-height:1.6;">'
        f'({form:.0f}×.32) + ({fit:.0f}×.30) + ({vegas:.0f}×.20) + ({dna:.0f}×.13) + ({traj:.0f}×.05) = '
        f'<span style="color:#52cc72;font-weight:500;">{score:.1f}</span>'
        f'</div>'
    )

    # One-line plain-English reason this player is on the team
    description = _description_for_player(pstat.get("Player", ""), pstat)
    description_html = (
        f'<div style="font-size:11px;color:#7aaa7a;'
        f'margin-top:5px;padding-bottom:4px;line-height:1.4;">'
        f'{safe_html(description)}'
        f'</div>'
    )

    return (
        f'<div style="padding:16px 20px;border-bottom:1px solid #111d11;'
        f'display:flex;flex-direction:column;gap:0;">'
        f'<div style="display:flex;align-items:flex-start;justify-content:space-between;gap:8px;">'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:0;">'
        f'<span style="font-size:18px;font-weight:500;color:#e8f5e8;'
        f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{name}</span>'
        f'{own_badge}'
        f'{lock_badge}'
        f'{was_html}'
        f'</div>'
        f'<div style="font-size:13px;color:#2a422a;margin-top:3px;display:flex;align-items:center;gap:6px;">'
        f'WR #{wr}'
        f'<span style="font-size:11px;font-weight:700;letter-spacing:.05em;padding:1px 5px;'
        f'border-radius:2px;background:{t_bg};color:{t_color};">{tour}</span>'
        f'&middot; {odds}'
        f'</div>'
        f'<div style="display:flex;align-items:baseline;gap:6px;margin:6px 0 8px;">'
        f'<span style="font-family:\'DM Mono\',monospace;font-size:22px;color:#e8f5e8;'
        f'font-weight:300;line-height:1;">{score:.1f}</span>'
        f'<span style="font-size:11px;color:#7aaa7a;font-weight:500;letter-spacing:.1em;'
        f'text-transform:uppercase;">AUGUSTA SCORE</span>'
        f'</div>'
        f'<div style="display:flex;flex-direction:column;margin-top:0;">'
        f'{mini_bar("FORM",  form,  "Form Score — recent tournament performance")}'
        f'{mini_bar("DNA",   dna,   "Augusta DNA — historical course fit")}'
        f'{mini_bar("FIT",   fit,   "Course Fit — Augusta course architecture match")}'
        f'{mini_bar("VEGAS", vegas, "Vegas Score — market-implied win probability")}'
        f'{mini_bar("TRAJ",  traj,  "Trajectory — world ranking momentum over 60 days")}'
        f'</div>'
        f'{score_math_html}'
        f'{description_html}'
        f'</div>'
        f'<div style="font-size:9px;color:#2a422a;cursor:pointer;flex-shrink:0;padding-top:2px;'
        f'white-space:nowrap;font-family:DM Sans,sans-serif;letter-spacing:.06em;text-transform:uppercase;">'
        f'SWAP</div>'
        f'</div>'
        f'{contrib_bar}'
        f'</div>'
    )


def get_next_up(df: pd.DataFrame, ct: dict, n: int = 5) -> pd.DataFrame:
    """Return the top-n players by Augusta_Score not already on any team,
    passing the cut rate floor (>= 0.60) and odds cap (<= 8000)."""
    current_picks: set[str] = set()
    for tk in ("team_a", "team_b", "team_c"):
        for p in ct.get(tk, {}).get("stats", []):
            current_picks.add(p.get("Player", ""))

    mask = ~df["Player"].isin(current_picks)
    if "Augusta_Cut_Rate" in df.columns:
        mask &= df["Augusta_Cut_Rate"] >= 0.60
    if "Odds_American" in df.columns:
        mask &= df["Odds_American"] <= 8000

    available = df[mask].copy()
    if "Augusta_Score" in available.columns:
        available = available.sort_values("Augusta_Score", ascending=False)
    return available.head(n)


def _next_up_mini_bar(label: str, val: float) -> str:
    """Compact FORM/DNA/FIT bar identical in style to player slot bars."""
    val = min(100.0, max(0.0, float(val)))
    if val >= 65:
        color = "#52CC72"
    elif val >= 40:
        color = "#8ABD6A"
    else:
        color = "#3A5A3A"
    return (
        f'<div style="display:flex;align-items:center;gap:5px;margin-bottom:4px;">'
        f'<div style="font-size:9px;color:#5a8a5a;letter-spacing:.06em;'
        f'text-transform:uppercase;width:32px;flex-shrink:0;">{label}</div>'
        f'<div style="width:52px;height:8px;background:#263826;border-radius:2px;flex-shrink:0;">'
        f'<div style="width:{val:.0f}%;height:100%;background:{color};border-radius:2px;"></div>'
        f'</div>'
        f'<div style="font-family:DM Mono,monospace;font-size:11px;color:{color};'
        f'width:24px;flex-shrink:0;">{val:.0f}</div>'
        f'</div>'
    )


def _next_up_driver(row: pd.Series) -> str:
    """One-line key driver sentence for a next-up card."""
    form = float(row.get("Form_Score", 50))
    dna  = float(row.get("DNA_Score",  50))
    fit  = float(row.get("Fit_Score",  50))
    odds = row.get("Odds_American", 50000)
    sg_t2g = row.get("SG_T2G", None)
    par5   = row.get("Par5_Avg", None)

    top = max(form, dna, fit)
    if top == form:
        if sg_t2g is not None:
            try:
                return f"Peak form — SG T2G {float(sg_t2g):+.2f}"
            except (TypeError, ValueError):
                pass
        return "Peak form — strong recent SG metrics"
    elif top == fit:
        if par5 is not None:
            try:
                return f"Course fit — par-5 avg {float(par5):.2f}"
            except (TypeError, ValueError):
                pass
        return "Course fit — Augusta ball-striking profile"
    else:  # DNA
        return "Augusta pedigree — strong course history"


def _render_next_up(df: pd.DataFrame, ct: dict) -> None:
    """Render the Next Up watchlist section."""
    next_up = get_next_up(df, ct, n=5)
    if next_up.empty:
        return

    # ── Gap indicator ───────────────────────────────────────────────
    all_team_players: list[dict] = []
    for tk in ("team_a", "team_b", "team_c"):
        all_team_players.extend(ct.get(tk, {}).get("stats", []))

    weakest_score  = float("inf")
    weakest_name   = ""
    for p in all_team_players:
        s = float(p.get("Augusta_Score", 0))
        if s < weakest_score:
            weakest_score = s
            weakest_name  = p.get("Player", "?")

    top_next_score = float(next_up.iloc[0].get("Augusta_Score", 0))
    gap            = weakest_score - top_next_score  # positive = weakest team > next-up
    top_next_name  = safe_html(str(next_up.iloc[0]["Player"]))
    weakest_name_h = safe_html(weakest_name)

    if gap < 0:
        # Next-up player scores HIGHER than the weakest current pick
        gap_html = (
            f'<div style="font-size:11px;color:#7aaa7a;margin-bottom:10px;">'
            f'Gap from weakest current pick '
            f'(<span style="color:#e8f5e8;">{weakest_name_h}</span>'
            f' · {weakest_score:.1f} pts) to next available: '
            f'<span style="font-family:DM Mono,monospace;color:#c8a84a;">{gap:+.1f} pts</span>'
            f'</div>'
            f'<div style="font-size:11px;color:#c8a84a;font-weight:700;margin-bottom:10px;">'
            f'&#9888; {top_next_name} scores higher than {weakest_name_h}'
            f' — consider swapping'
            f'</div>'
        )
    else:
        gap_html = (
            f'<div style="font-size:11px;color:#7aaa7a;margin-bottom:10px;">'
            f'Gap from weakest current pick '
            f'(<span style="color:#e8f5e8;">{weakest_name_h}</span>'
            f' · {weakest_score:.1f} pts) to next available: '
            f'<span style="font-family:DM Mono,monospace;color:#c8a84a;">{gap:+.1f} pts</span>'
            f'</div>'
        )

    st.markdown(
        '<div style="font-size:11px;font-weight:700;'
        'letter-spacing:.16em;text-transform:uppercase;'
        'color:#7aaa7a;border-bottom:1px solid #1e301e;'
        'padding-bottom:6px;margin:20px 0 12px;">'
        'NEXT UP — MODEL\'S NEXT BEST AVAILABLE'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:11px;color:#5a8a5a;'
        'margin-bottom:14px;font-style:italic;">'
        'Players not on any team, ranked by Augusta Score. '
        'Consider these if swapping a current pick.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(gap_html, unsafe_allow_html=True)

    # ── Cards — one per column ───────────────────────────────────────
    n_cards = len(next_up)
    cols = st.columns(n_cards, gap="small")

    for col_idx, (_, row) in enumerate(next_up.iterrows()):
        rank       = col_idx + 1
        name_raw   = str(row.get("Player", "?"))
        name       = safe_html(name_raw)
        wr         = int(row.get("World_Rank", 999))
        tour       = str(row.get("Tour", "PGA"))
        score      = float(row.get("Augusta_Score", 0))
        ev         = float(row.get("EV_Score", 0))
        form       = float(row.get("Form_Score", 50))
        dna        = float(row.get("DNA_Score", 50))
        fit        = float(row.get("Fit_Score", 50))
        own_pct    = float(row.get("Ownership_Pct", 5.0))
        odds       = row.get("Odds_American", 50000)
        odds_str   = _fmt_odds(odds)
        driver     = safe_html(_next_up_driver(row))

        # Tour badge
        if tour == "LIV":
            tour_bg, tour_color = "#150e05", "#c4783a"
        else:
            tour_bg, tour_color = "#1e301e", "#4a6a4a"

        # Ownership band
        if own_pct > 35:
            band, band_bg, band_color, band_border = "CHALK", "#1a1400", "#c8a84a", "#2a2000"
        elif own_pct >= 10:
            band, band_bg, band_color, band_border = "MID",   "#1a2a1a", "#96cc96", "#263826"
        elif own_pct >= 3:
            band, band_bg, band_color, band_border = "VALUE", "#0a1a0a", "#52cc72", "#1e301e"
        else:
            band, band_bg, band_color, band_border = "FIELD", "#1e301e", "#4a6a4a", "#1e301e"

        form_bar = _next_up_mini_bar("FORM", form)
        dna_bar  = _next_up_mini_bar("DNA",  dna)
        fit_bar  = _next_up_mini_bar("FIT",  fit)

        card_html = (
            f'<div style="background:#0d160d;border:1px solid #1e301e;'
            f'border-radius:6px;padding:12px 14px;position:relative;'
            f'font-family:DM Sans,sans-serif;">'

            # Rank badge
            f'<div style="position:absolute;top:8px;right:8px;'
            f'font-family:DM Mono,monospace;font-size:10px;color:#7aaa7a;">#{rank}</div>'

            # Player name
            f'<div style="font-size:14px;font-weight:500;color:#e8f5e8;'
            f'margin-bottom:2px;overflow:hidden;text-overflow:ellipsis;'
            f'white-space:nowrap;padding-right:20px;">{name}</div>'

            # Key driver sentence
            f'<div style="font-size:10px;color:#5a8a5a;font-style:italic;'
            f'margin-bottom:5px;overflow:hidden;text-overflow:ellipsis;'
            f'white-space:nowrap;">{driver}</div>'

            # WR + Tour
            f'<div style="font-size:11px;color:#7aaa7a;margin-bottom:8px;'
            f'display:flex;align-items:center;gap:6px;">'
            f'WR #{wr}'
            f'<span style="font-size:9px;font-weight:700;padding:1px 4px;'
            f'border-radius:2px;background:{tour_bg};color:{tour_color};">'
            f'{safe_html(tour)}</span>'
            f'</div>'

            # Augusta Score (prominent)
            f'<div style="display:flex;align-items:baseline;gap:4px;margin-bottom:6px;">'
            f'<span style="font-family:DM Mono,monospace;font-size:22px;'
            f'color:#c8a84a;line-height:1;">{score:.1f}</span>'
            f'<span style="font-size:9px;color:#5a8a5a;text-transform:uppercase;'
            f'letter-spacing:.06em;">score</span>'
            f'</div>'

            # Odds + Ownership band
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;margin-bottom:8px;">'
            f'<span style="font-family:DM Mono,monospace;font-size:12px;'
            f'color:#5a8a5a;">{odds_str}</span>'
            f'<span style="font-size:9px;font-weight:700;padding:1px 6px;'
            f'border-radius:3px;background:{band_bg};color:{band_color};'
            f'border:1px solid {band_border};">{band}</span>'
            f'</div>'

            # FORM / DNA / FIT bars
            + form_bar + dna_bar + fit_bar +

            # EV score footer
            f'<div style="margin-top:8px;padding-top:6px;'
            f'border-top:1px solid #1e301e;'
            f'display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="font-size:9px;color:#5a8a5a;text-transform:uppercase;'
            f'letter-spacing:.06em;">EV Score</span>'
            f'<span style="font-family:DM Mono,monospace;font-size:13px;'
            f'color:#52cc72;">{ev:.1f}</span>'
            f'</div>'

            f'</div>'
        )

        with cols[col_idx]:
            st.markdown(card_html, unsafe_allow_html=True)

    st.markdown('<div style="margin-bottom:4px;"></div>', unsafe_allow_html=True)


def tab_my_picks(df: pd.DataFrame, teams: dict, data: dict):
    if st.session_state.get("picks_confirmed", False):
        render_confirmed_view()
        return

    # Force re-enrich if scores look like stale 50-defaults from old cached data
    if "custom_teams" in st.session_state:
        ct_check = st.session_state["custom_teams"]
        sample = ct_check.get("team_a", {}).get("stats", [{}])[0]
        if sample.get("Form_Score", 50) == 50:
            del st.session_state["custom_teams"]

    # Lazy-init custom teams (enriched)
    if "custom_teams" not in st.session_state:
        st.session_state["custom_teams"] = {
            "team_a": _enrich_team(teams.get("team_a", {}), df),
            "team_b": _enrich_team(teams.get("team_b", {}), df),
            "team_c": _enrich_team(teams.get("team_c", {}), df),
        }

    # Swap confirmation / lock session state
    if "locked_swaps" not in st.session_state:
        st.session_state["locked_swaps"] = {"team_a": {}, "team_b": {}, "team_c": {}}
    if "pending_swap" not in st.session_state:
        st.session_state["pending_swap"] = None

    ct = st.session_state["custom_teams"]

    # ── Data freshness indicator ────────────────────────────────────
    _meta = data.get("stats", {}).get("_meta", {})
    _src  = _meta.get("source", "fallback")
    _ts   = _meta.get("fetched_at", "")
    _live = _meta.get("players_live", 0)
    _fb   = _meta.get("players_fallback", 0)
    if _src == "pga_graphql" and _ts:
        _fresh_txt = (
            f'Stats last updated: {_ts} &nbsp;·&nbsp; '
            f'{_live} players with live PGA Tour data &nbsp;·&nbsp; '
            f'{_fb} LIV / fallback'
        )
        _fresh_clr = "#2a422a"
    elif _src == "cache":
        _fresh_txt = "Stats loaded from cache · refresh page for live data"
        _fresh_clr = "#2a422a"
    else:
        _fresh_txt = "Using fallback stats — live fetch unavailable"
        _fresh_clr = "#5a3a1a"
    st.markdown(
        f'<div style="font-size:9px;color:{_fresh_clr};'
        f'margin-bottom:14px;letter-spacing:.03em;">{_fresh_txt}</div>',
        unsafe_allow_html=True,
    )

    # ── Build per-team slot HTML ────────────────────────────────────
    def slots_html(team_key: str, t_color: str = "#3AAA5A") -> str:
        total = float(ct[team_key].get("total_augusta_score", 0))
        return "".join(
            _build_slot_html(p, team_composite=total, team_color=t_color)
            for p in ct[team_key].get("stats", [])
        )

    def team_ev(key):  return f'{float(ct[key].get("total_ev_score", 0)):.1f}'
    def team_tot(key): return f'{float(ct[key].get("total_augusta_score", 0)):.1f}'

    ta_label = safe_html(ct["team_a"].get("label", "TEAM A"))
    tb_label = safe_html(ct["team_b"].get("label", "TEAM B"))
    tc_label = safe_html(ct["team_c"].get("label", "TEAM C"))
    ta_strat = safe_html(ct["team_a"].get("strategy", "Best 4 by composite score")[:60])
    tb_strat = safe_html(ct["team_b"].get("strategy", "Best EV · Model recommended")[:60])
    tc_strat = safe_html(ct["team_c"].get("strategy", "Low ownership · Maximum differentiation")[:60])

    # ── Single st.markdown() call — pure HTML grid ──────────────────
    def _stat_pills(composite: str, ev: str, comp_color: str) -> str:
        return (
            f'<div style="display:flex;gap:20px;align-items:flex-end;margin-bottom:8px;padding:0 2px;">'
            f'<div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:28px;color:{comp_color};line-height:1;">{composite}</div>'
            f'<div style="font-size:10px;letter-spacing:.10em;text-transform:uppercase;color:#5a8a5a;margin-top:2px;">COMPOSITE</div>'
            f'</div>'
            f'<div style="margin-left:auto;">'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:28px;color:#c8a84a;line-height:1;">{ev}</div>'
            f'<div style="font-size:10px;letter-spacing:.10em;text-transform:uppercase;color:#5a8a5a;margin-top:2px;">EV SCORE</div>'
            f'</div>'
            f'</div>'
        )

    cards_html = f"""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;padding-bottom:16px;font-family:'DM Sans',sans-serif;">

  <div>
    {_stat_pills(team_tot("team_a"), team_ev("team_a"), "#5a8a5a")}
    <div style="background:#0d160d;border:1px solid #1e301e;border-radius:6px;overflow:hidden;">
      <div style="padding:14px 20px;border-bottom:1px solid #1e301e;">
        <div style="font-size:12px;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:#5a8a5a;">{ta_label}</div>
        <div style="font-size:11px;color:#7aaa7a;margin-top:2px;">{ta_strat}</div>
      </div>
      {slots_html("team_a", "#5a8a5a")}
    </div>
  </div>

  <div>
    {_stat_pills(team_tot("team_b"), team_ev("team_b"), "#5a8a5a")}
    <div style="background:#0d160d;border:1px solid #1e301e;border-radius:6px;overflow:hidden;">
      <div style="padding:14px 20px;border-bottom:1px solid #1e301e;">
        <div style="font-size:12px;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:#5a8a5a;">{tb_label}</div>
        <div style="font-size:11px;color:#7aaa7a;margin-top:2px;">{tb_strat}</div>
      </div>
      {slots_html("team_b", "#5a8a5a")}
    </div>
  </div>

  <div>
    {_stat_pills(team_tot("team_c"), team_ev("team_c"), "#5a8a5a")}
    <div style="background:#0d160d;border:1px solid #1e301e;border-radius:6px;overflow:hidden;">
      <div style="padding:14px 20px;border-bottom:1px solid #1e301e;">
        <div style="font-size:12px;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:#5a8a5a;">{tc_label}</div>
        <div style="font-size:11px;color:#7aaa7a;margin-top:2px;">{tc_strat}</div>
      </div>
      {slots_html("team_c", "#5a8a5a")}
    </div>
  </div>

</div>
"""
    st.markdown(cards_html, unsafe_allow_html=True)

    # ── PORTFOLIO HEALTH ROW ────────────────────────────────────────
    corr = teams.get("correlation", {})
    corr_score = corr.get("correlation_score", 0)
    if corr_score <= 3:
        corr_color, corr_label = "#3aaa5a", "GOOD"
    elif corr_score <= 5:
        corr_color, corr_label = "#c8a84a", "MODERATE"
    else:
        corr_color, corr_label = "#cc4a4a", "HIGH"

    dup_counts = [
        teams.get(tk, {}).get("duplication", {}).get("chalk_overlap_count", 0)
        for tk in ("team_a", "team_b", "team_c")
    ]
    avg_dup = sum(dup_counts) / max(len(dup_counts), 1)
    if avg_dup >= 2.5:
        dup_label, dup_color = "HIGH DUPLICATION", "#cc4a4a"
    elif avg_dup >= 1.5:
        dup_label, dup_color = "MODERATE", "#c8a84a"
    else:
        dup_label, dup_color = "LOW DUPLICATION", "#3aaa5a"

    corr_warnings = corr.get("warnings", [])
    corr_warn_html = ""
    if corr_warnings:
        corr_warn_html = (
            '<div style="font-size:8px;color:#cc4a4a;margin-top:3px;line-height:1.4;">'
            + "<br>".join(safe_html(w) for w in corr_warnings[:2])
            + "</div>"
        )

    portfolio_html = f"""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px;">
  <div style="background:#080e08;border:1px solid #152015;border-radius:4px;padding:12px 16px;">
    <div style="font-size:10px;letter-spacing:.10em;text-transform:uppercase;color:#7aaa7a;margin-bottom:4px;">PORTFOLIO CORRELATION</div>
    <div style="font-size:14px;font-weight:600;color:{corr_color};">{corr_label}</div>
    <div style="font-size:9px;color:#2a422a;margin-top:2px;">Score: {corr_score} (lower = better)</div>
    {corr_warn_html}
  </div>
  <div style="background:#080e08;border:1px solid #152015;border-radius:4px;padding:12px 16px;">
    <div style="font-size:10px;letter-spacing:.10em;text-transform:uppercase;color:#7aaa7a;margin-bottom:4px;">AVG TEAM DUPLICATION</div>
    <div style="font-size:14px;font-weight:600;color:{dup_color};">{dup_label}</div>
    <div style="font-size:9px;color:#2a422a;margin-top:2px;">{avg_dup:.1f} chalk players avg per team</div>
  </div>
  <div style="background:#080e08;border:1px solid #152015;border-radius:4px;padding:12px 16px;">
    <div style="font-size:10px;letter-spacing:.10em;text-transform:uppercase;color:#7aaa7a;margin-bottom:4px;">COVERAGE</div>
    <div style="font-size:10px;color:#96cc96;line-height:1.6;">Team A: chalk fires</div>
    <div style="font-size:10px;color:#96cc96;line-height:1.6;">Team B: mid-tier emerges</div>
    <div style="font-size:10px;color:#c8a84a;line-height:1.6;">Team C: overlooked top-10</div>
  </div>
</div>
"""
    st.markdown(portfolio_html, unsafe_allow_html=True)

    # ── NEXT UP — MODEL'S NEXT BEST AVAILABLE ──────────────────────
    _render_next_up(df, ct)

    # ── SWAP CONTROLS — two-stage propose → confirm flow ────────────
    with st.expander("↕ Swap a player", expanded=False):

        st.markdown(
            '<div style="font-size:11px;font-weight:700;letter-spacing:.14em;'
            'text-transform:uppercase;color:#7aaa7a;margin-bottom:10px;">'
            'SELECT A PLAYER TO SWAP</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            swap_team = st.selectbox(
                "Team",
                options=["team_a", "team_b", "team_c"],
                format_func=lambda x: {
                    "team_a": "Team A — Floor",
                    "team_b": "Team B — Ceiling",
                    "team_c": "Team C — Value",
                }[x],
                key="swap_team_select",
            )

        with col2:
            current_players = [p.get("Player", "") for p in ct[swap_team].get("stats", [])]
            swap_out = st.selectbox("Remove", options=current_players, key="swap_out_select")

        with col3:
            all_team_players = {
                p.get("Player")
                for tk in ["team_a", "team_b", "team_c"]
                for p in ct[tk].get("stats", [])
            }

            def _fmt_swap_player(name: str) -> str:
                row = df[df["Player"] == name]
                if row.empty:
                    return name
                r = row.iloc[0]
                odds = r.get("Odds_American", 0)
                odds_s = f"+{int(odds)}" if odds > 0 else str(int(odds))
                own = r.get("Ownership_Pct", 0)
                sc  = r.get("Augusta_Score", 0)
                return f"{name}  [Score:{sc:.0f} Own:{own:.0f}% {odds_s}]"

            cut_col = "Augusta_Cut_Rate" if "Augusta_Cut_Rate" in df.columns else None
            avail_mask = ~df["Player"].isin(all_team_players)
            if cut_col:
                avail_mask &= df[cut_col] >= 0.60
            available = df[avail_mask].sort_values("EV_Score", ascending=False)

            swap_in = st.selectbox(
                "Add",
                options=available["Player"].tolist(),
                format_func=_fmt_swap_player,
                key="swap_in_select",
            )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        if st.button("Preview this swap →", key="propose_swap_btn"):
            slot_idx = current_players.index(swap_out)
            st.session_state["pending_swap"] = {
                "team": swap_team,
                "slot": slot_idx,
                "out":  swap_out,
                "in":   swap_in,
            }
            st.rerun()

        # ── Stage 2: confirm pending swap ───────────────────────────
        ps = st.session_state.get("pending_swap")
        if ps:
            out_row = df[df["Player"] == ps["out"]]
            in_row  = df[df["Player"] == ps["in"]]
            out_sc  = float(out_row.iloc[0]["Augusta_Score"]) if not out_row.empty else 0
            in_sc   = float(in_row.iloc[0]["Augusta_Score"])  if not in_row.empty else 0
            out_own = float(out_row.iloc[0]["Ownership_Pct"]) if not out_row.empty else 0
            in_own  = float(in_row.iloc[0]["Ownership_Pct"])  if not in_row.empty else 0
            delta   = in_sc - out_sc
            delta_color = "#52cc72" if delta >= 0 else "#cc4a4a"
            delta_str   = f"{delta:+.1f}"
            team_label  = {"team_a": "Team A", "team_b": "Team B", "team_c": "Team C"}[ps["team"]]

            st.markdown(f"""
<div style="background:#0d160d;border:1px solid #c8a84a;border-radius:6px;
padding:14px 18px;margin-top:12px;">
  <div style="font-size:9px;font-weight:700;letter-spacing:.14em;color:#c8a84a;
  margin-bottom:10px;">PROPOSED SWAP — {team_label}</div>
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
    <div style="background:#1a0808;border:1px solid #cc4a4a33;border-radius:4px;
    padding:8px 14px;flex:1;">
      <div style="font-size:9px;color:#cc4a4a;font-weight:700;margin-bottom:4px;">REMOVING</div>
      <div style="font-size:15px;color:#e8f5e8;font-weight:500;">{safe_html(ps['out'])}</div>
      <div style="font-size:11px;color:#7aaa7a;margin-top:3px;">
        Score: {out_sc:.1f} · Own: {out_own:.1f}%</div>
    </div>
    <div style="font-size:20px;color:#5a8a5a;">→</div>
    <div style="background:#0a1a0a;border:1px solid #3aaa5a33;border-radius:4px;
    padding:8px 14px;flex:1;">
      <div style="font-size:9px;color:#3aaa5a;font-weight:700;margin-bottom:4px;">ADDING</div>
      <div style="font-size:15px;color:#e8f5e8;font-weight:500;">{safe_html(ps['in'])}</div>
      <div style="font-size:11px;color:#7aaa7a;margin-top:3px;">
        Score: {in_sc:.1f} · Own: {in_own:.1f}%</div>
    </div>
    <div style="background:#111d11;border:1px solid #1e301e;border-radius:4px;
    padding:8px 14px;text-align:center;">
      <div style="font-size:10px;color:#7aaa7a;font-weight:700;margin-bottom:4px;">SCORE DELTA</div>
      <div style="font-family:DM Mono,monospace;font-size:20px;
      color:{delta_color};font-weight:300;">{delta_str}</div>
    </div>
  </div>
  <div style="margin-top:10px;font-size:10px;color:#5a8a5a;font-style:italic;">
    This will override the model's selection for this slot.
    You can undo it at any time.</div>
</div>
""", unsafe_allow_html=True)

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1, 1, 3])

            with c1:
                if st.button("✅ Confirm & Lock", key="confirm_swap_btn",
                             type="primary", use_container_width=True):
                    _confirm_lock_swap(
                        ps["team"], ps["slot"], ps["in"], ps["out"], df
                    )
                    st.session_state["pending_swap"] = None
                    st.rerun()

            with c2:
                if st.button("✕ Cancel", key="cancel_swap_btn", use_container_width=True):
                    st.session_state["pending_swap"] = None
                    st.rerun()

    # ── Locked swaps management panel ───────────────────────────────
    any_locked = any(bool(v) for v in st.session_state["locked_swaps"].values())
    if any_locked:
        st.markdown(
            '<div style="font-size:9px;font-weight:700;letter-spacing:.14em;'
            'text-transform:uppercase;color:#c8a84a;margin:16px 0 8px;">'
            '🔒 LOCKED SWAPS</div>',
            unsafe_allow_html=True,
        )
        for tk in ["team_a", "team_b", "team_c"]:
            swaps = st.session_state["locked_swaps"][tk]
            if not swaps:
                continue
            team_label = {"team_a": "Team A", "team_b": "Team B", "team_c": "Team C"}[tk]
            for slot, lock_info in list(swaps.items()):
                original = lock_info.get("out", "model pick")
                player   = lock_info.get("in", "?")
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(
                        f'<div style="font-size:11px;color:#e8f5e8;padding:4px 0;">'
                        f'<span style="color:#c8a84a;">{team_label} slot {slot+1}</span>'
                        f' · {safe_html(original)} → '
                        f'<span style="color:#52cc72;font-weight:500;">{safe_html(player)}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with col2:
                    if st.button("Undo", key=f"undo_{tk}_{slot}", use_container_width=True):
                        _undo_lock_swap(tk, slot, df)

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.button("Reset all swaps to model defaults", key="reset_all_swaps"):
            st.session_state["locked_swaps"] = {"team_a": {}, "team_b": {}, "team_c": {}}
            st.session_state["pending_swap"] = None
            st.session_state["custom_teams"] = {
                "team_a": _enrich_team(teams.get("team_a", {}), df),
                "team_b": _enrich_team(teams.get("team_b", {}), df),
                "team_c": _enrich_team(teams.get("team_c", {}), df),
            }
            st.rerun()

    # ── Tiebreaker ──────────────────────────────────────────────────
    render_tiebreaker(data)

    # ── Confirm button ──────────────────────────────────────────────
    st.markdown('<div style="margin-top:8px;" class="confirm-btn">', unsafe_allow_html=True)
    if st.button("CONFIRM MY PICKS", key="confirm_btn", use_container_width=True):
        st.session_state["confirmed_picks"] = {
            "teams":     st.session_state["custom_teams"],
            "tiebreaker": st.session_state.get("tiebreaker_value", -42),
        }
        st.session_state["picks_confirmed"] = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# TAB 2 — PLAYER RANKINGS
# ─────────────────────────────────────────────────────────────────

def _rankings_row_html(i: int, row: pd.Series, max_score: float) -> str:
    rank   = i + 1
    name   = safe_html(str(row["Player"]))
    wr     = int(row.get("World_Rank", 999))
    tour   = safe_html(str(row.get("Tour", "PGA")))
    ev     = float(row.get("EV_Score", 0))
    score  = float(row.get("Augusta_Score", 0))
    form   = float(row.get("Form_Score", 50))
    dna    = float(row.get("DNA_Score", 50))
    fit    = float(row.get("Fit_Score", 50))
    traj   = float(row.get("Trajectory_Score", 50))
    own    = float(row.get("Ownership_Pct", 0))
    odds   = row.get("Odds_American", 50000)
    rank_cls  = "rk-num g" if rank == 1 else "rk-num"
    ev_cls    = _ev_class(ev)
    tour_html = _tour_badge(tour)
    own_cls   = "ow-hi" if own >= 15 else "ow"

    # Score bar
    pct_score = min(100, score / max(max_score, 1) * 100)
    bar_clr   = _bar_color(pct_score)

    # Augusta history (last 5 years)
    raw_name = str(row["Player"])   # un-escaped original for dict lookup
    hist = PLAYER_MASTERS_HISTORY.get(raw_name, {})
    hist_years = sorted(hist.keys(), reverse=True)[:5]
    hist_cells = "".join(
        f'<span style="font-family:\'DM Mono\',monospace;font-size:10px;color:var(--t3);margin-right:10px;">'
        f'{safe_html(str(y))}: {safe_html(str(hist[y].get("finish","–")))}'
        f'</span>'
        for y in hist_years
    ) if hist_years else '<span style="font-size:10px;color:var(--t3);">No Augusta history</span>'

    # Component sub-scores
    vegas  = float(row.get("Vegas_Score", 50))
    comp_html = ""
    for lbl, val, base in [("FORM", form, 100), ("DNA", dna, 100),
                             ("FIT", fit, 100), ("VEGAS", vegas, 100),
                             ("TRAJ", traj, 100)]:
        p = min(100, val / max(base, 1) * 100)
        c = _bar_color(p)
        comp_html += (
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:5px;">'
            f'<div style="font-size:9px;color:#5a8a5a;letter-spacing:.06em;text-transform:uppercase;'
            f'width:36px;flex-shrink:0;">{lbl}</div>'
            f'<div style="width:52px;height:8px;background:#263826;border-radius:2px;flex-shrink:0;">'
            f'<div style="width:{p:.0f}%;height:100%;background:{c};border-radius:2px;"></div>'
            f'</div>'
            f'<div style="font-family:DM Mono,monospace;font-size:12px;color:{c};'
            f'width:26px;flex-shrink:0;">{val:.0f}</div>'
            f'</div>'
        )

    # Key stats — pre-compute formatted strings to avoid f-string format specifier errors
    sg_app     = row.get("SG_App", "–")
    dd         = row.get("Drive_Dist", "–")
    par5       = row.get("Par5_Avg", "–")
    sg_app_fmt = safe_html(f"{sg_app:.2f}" if isinstance(sg_app, float) else str(sg_app))
    dd_fmt     = safe_html(f"{dd:.0f}" if isinstance(dd, float) else str(dd))
    par5_fmt   = safe_html(f"{par5:.2f}" if isinstance(par5, float) else str(par5))
    key_stats = (
        f'<div class="exp-lbl">SG App</div><div class="exp-val">{sg_app_fmt}</div>'
        f'<div class="exp-lbl" style="margin-top:5px;">Drive Dist</div><div class="exp-val">{dd_fmt} yds</div>'
        f'<div class="exp-lbl" style="margin-top:5px;">Par-5 Avg</div><div class="exp-val">{par5_fmt}</div>'
    )

    return (
        f'<details class="rk-row">'
        # ── summary row ──────────────────────────────────────────
        f'<summary>'
        f'<div class="{rank_cls}">{rank}</div>'
        f'<div>'
        f'<div class="rk-nm">{name}</div>'
        f'<div class="rk-meta">#{wr}{tour_html}</div>'
        f'</div>'
        f'<div class="{ev_cls}">{ev:.1f}</div>'
        # Score bar
        f'<div><div class="sbg"><div style="width:{pct_score:.0f}%;height:100%;background:{bar_clr};"></div></div>'
        f'<span style="font-family:\'DM Mono\',monospace;font-size:11px;color:var(--t2);margin-left:5px;">{score:.1f}</span></div>'
        f'<div style="font-family:\'DM Mono\',monospace;font-size:11px;color:var(--t3);">{form:.0f}</div>'
        f'<div style="font-family:\'DM Mono\',monospace;font-size:11px;color:var(--t3);">{dna:.0f}</div>'
        f'<div style="font-family:\'DM Mono\',monospace;font-size:11px;color:var(--t3);">{fit:.0f}</div>'
        f'<div class="{own_cls}">{own:.1f}%</div>'
        f'<div>{_odds_html(odds)}</div>'
        f'<div></div>'
        f'</summary>'
        # ── expanded detail ───────────────────────────────────────
        f'<div class="rk-exp">'
        f'<div><div class="exp-sec-hdr">SCORE COMPONENTS</div>{comp_html}</div>'
        f'<div><div class="exp-sec-hdr">AUGUSTA HISTORY</div>{hist_cells}</div>'
        f'<div><div class="exp-sec-hdr">KEY STATS</div>{key_stats}</div>'
        f'</div>'
        f'</details>'
    )


def tab_rankings(df: pd.DataFrame, data: dict):
    # ── Sort state init ───────────────────────────────────────────
    if "model_sort_col" not in st.session_state:
        st.session_state["model_sort_col"] = "Augusta_Score"
    if "model_sort_dir" not in st.session_state:
        st.session_state["model_sort_dir"] = "desc"

    # ── Controls bar ──────────────────────────────────────────────
    ctrl_l, _ = st.columns([3, 1])
    with ctrl_l:
        tour_filter = st.radio(
            "Filter",
            ["ALL", "PGA", "LIV", "NO FLAGS"],
            horizontal=True,
            key="rk_filter",
            label_visibility="collapsed",
        )

    # ── Weight adjustment toggle (replaces st.expander to kill keyboard artifact)
    weights_changed = False
    show_weights = st.toggle("Adjust Model Weights", value=False, key="show_weights")
    if show_weights:
        st.markdown(
            '<div style="background:#080e08;border:1px solid #152015;'
            'border-radius:6px;padding:16px;margin-bottom:12px;">',
            unsafe_allow_html=True,
        )
        w_col1, w_col2, w_col3, w_col4, w_col5 = st.columns(5)
        with w_col1:
            wf  = st.slider("Form",        0, 50, int(st.session_state.get("cw_form",  32)), 1, key="wslide_form")
        with w_col2:
            wft = st.slider("Course Fit",  0, 50, int(st.session_state.get("cw_fit",   27)), 1, key="wslide_fit")
        with w_col3:
            wd  = st.slider("Augusta DNA", 0, 40, int(st.session_state.get("cw_dna",   18)), 1, key="wslide_dna")
        with w_col4:
            wv  = st.slider("Vegas",       0, 40, int(st.session_state.get("cw_vegas", 18)), 1, key="wslide_vegas")
        with w_col5:
            wt  = st.slider("Trajectory",  0, 20, int(st.session_state.get("cw_traj",   5)), 1, key="wslide_traj")

        total_w = wf + wd + wft + wv + wt or 100
        new_cw = {
            "form":       wf  / total_w,
            "dna":        wd  / total_w,
            "fit":        wft / total_w,
            "vegas":      wv  / total_w,
            "trajectory": wt  / total_w,
        }
        st.markdown(
            f'<div style="font-size:9px;color:#2a422a;letter-spacing:.06em;margin-top:4px;">'
            f'Normalized — Form {new_cw["form"]:.0%} &nbsp; DNA {new_cw["dna"]:.0%} &nbsp; '
            f'Fit {new_cw["fit"]:.0%} &nbsp; Vegas {new_cw["vegas"]:.0%} &nbsp; Traj {new_cw["trajectory"]:.0%}'
            f'</div>',
            unsafe_allow_html=True,
        )

        prev_cw = st.session_state.get("active_cw", {})
        if new_cw != prev_cw and prev_cw:
            weights_changed = True

        if st.button("APPLY WEIGHTS", key="apply_weights_btn"):
            st.session_state["cw_form"]   = wf
            st.session_state["cw_dna"]    = wd
            st.session_state["cw_fit"]    = wft
            st.session_state["cw_vegas"]  = wv
            st.session_state["cw_traj"]   = wt
            st.session_state["active_cw"] = new_cw
            st.session_state["scored_df"] = None
            st.session_state["custom_teams"] = None
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    if weights_changed:
        st.markdown(
            '<div style="font-size:10px;color:#c8a84a;background:#0e0a00;border:1px solid #2a1800;'
            'border-radius:4px;padding:8px 12px;margin-bottom:8px;">'
            'Weights adjusted — click APPLY WEIGHTS to rescore, then return to MY PICKS.</div>',
            unsafe_allow_html=True,
        )

    # ── Filter ─────────────────────────────────────────────────────
    filtered = df.copy()
    if tour_filter == "PGA":
        filtered = filtered[filtered["Tour"] == "PGA"]
    elif tour_filter == "LIV":
        filtered = filtered[filtered["Tour"] == "LIV"]
    elif tour_filter == "NO FLAGS":
        filtered = filtered[filtered["Flags"] == ""]

    # ── Sort buttons ───────────────────────────────────────────────
    _sort_options = [
        ("Score",  "Augusta_Score"),
        ("EV",     "EV_Score"),
        ("Form",   "Form_Score"),
        ("Fit",    "Fit_Score"),
        ("DNA",    "DNA_Score"),
        ("Odds",   "Odds_American"),
        ("Own%",   "Ownership_Pct"),
    ]
    _sort_cols = st.columns(len(_sort_options))
    for _sc, (_lbl, _ckey) in zip(_sort_cols, _sort_options):
        with _sc:
            _active = st.session_state["model_sort_col"] == _ckey
            _arrow = (" ↓" if st.session_state["model_sort_dir"] == "desc"
                      else " ↑") if _active else ""
            if st.button(
                f"{_lbl}{_arrow}",
                key=f"sort_{_ckey}",
                use_container_width=True,
                type="primary" if _active else "secondary",
            ):
                if st.session_state["model_sort_col"] == _ckey:
                    st.session_state["model_sort_dir"] = (
                        "asc" if st.session_state["model_sort_dir"] == "desc" else "desc"
                    )
                else:
                    st.session_state["model_sort_col"] = _ckey
                    st.session_state["model_sort_dir"] = "desc"
                st.rerun()

    # ── Apply sort ─────────────────────────────────────────────────
    _sort_asc = st.session_state["model_sort_dir"] == "asc"
    _scol = st.session_state["model_sort_col"]
    if _scol in filtered.columns:
        filtered = filtered.sort_values(_scol, ascending=_sort_asc).reset_index(drop=True)
    else:
        filtered = filtered.sort_values("Augusta_Score", ascending=False).reset_index(drop=True)

    max_score = float(filtered["Augusta_Score"].max()) if not filtered.empty else 100.0

    # ── Summary metrics ────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Players", len(filtered))
    m2.metric("Avg Augusta Score", f"{filtered['Augusta_Score'].mean():.1f}" if not filtered.empty else "–")
    m3.metric("Flagged", int(filtered["Flag_Count"].gt(0).sum()) if "Flag_Count" in filtered.columns else "–")
    m4.metric("Chalk Penalized", int(filtered["Chalk_Penalty"].sum()) if "Chalk_Penalty" in filtered.columns else "–")

    # ── Column headers ─────────────────────────────────────────────
    st.markdown(
        '<div class="rk-hdr">'
        '<div class="rk-hdr-cell">RK</div>'
        '<div class="rk-hdr-cell">PLAYER</div>'
        '<div class="rk-hdr-cell">EV</div>'
        '<div class="rk-hdr-cell">AUG SCORE</div>'
        '<div class="rk-hdr-cell">FORM</div>'
        '<div class="rk-hdr-cell">DNA</div>'
        '<div class="rk-hdr-cell">FIT</div>'
        '<div class="rk-hdr-cell">OWN%</div>'
        '<div class="rk-hdr-cell">ODDS</div>'
        '<div class="rk-hdr-cell">FLAGS</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Table rows ─────────────────────────────────────────────────
    rows_html = "".join(
        _rankings_row_html(i, row, max_score)
        for i, (_, row) in enumerate(filtered.head(60).iterrows())
    )
    st.markdown(rows_html, unsafe_allow_html=True)

    if len(filtered) > 60:
        st.caption(f"Showing top 60 of {len(filtered)} players. Use filters to narrow.")


# ─────────────────────────────────────────────────────────────────
# TAB 3 — POOL STANDINGS
# ─────────────────────────────────────────────────────────────────

def tab_pool_standings(data: dict):
    # ── User entry identification ──────────────────────────────────
    top_inp, _ = st.columns([2, 3])
    with top_inp:
        user_input = st.text_input(
            "YOUR ENTRY NAMES (comma-separated)",
            value=st.session_state.get("user_entries_str", ""),
            key="standings_user_entries",
            placeholder="e.g.  John Smith #1, John Smith #2",
        )
        if user_input != st.session_state.get("user_entries_str", ""):
            st.session_state["user_entries_str"] = user_input

    user_entries = [e.strip() for e in user_input.split(",") if e.strip()]

    # ── File upload ────────────────────────────────────────────────
    if "pool_df_raw" not in st.session_state:
        st.markdown(
            '<div style="border:1px dashed var(--b2);padding:32px;text-align:center;background:var(--s1);margin-bottom:16px;">'
            '<div style="font-size:14px;color:var(--t2);margin-bottom:6px;">Upload Pool Entries File</div>'
            '<div style="font-size:10px;color:var(--t2b);">Supported formats: CSV, XLSX &nbsp;&middot;&nbsp; '
            'Required columns: Entry_Name, P1, P2, P3, P4, Tiebreaker</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "POOL FILE",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
            key="pool_uploader",
        )
        if uploaded is not None:
            entries, err = parse_pool_entries(uploaded)
            if err:
                st.error(err)
            else:
                st.session_state["pool_df_raw"] = entries
                st.rerun()
        else:
            st.caption("Using demo data below for illustration.")

    # ── Fetch / cache live scores ──────────────────────────────────
    refresh_c, auto_c = st.columns([1, 2])
    with refresh_c:
        st.markdown('<div class="refresh-btn">', unsafe_allow_html=True)
        if st.button("REFRESH SCORES", key="refresh_scores_standings"):
            st.session_state["live_scores"] = None
        st.markdown("</div>", unsafe_allow_html=True)
    with auto_c:
        auto_ref = st.checkbox("Auto-refresh every 5 min", value=False, key="auto_ref_standings")

    if "live_scores" not in st.session_state or st.session_state["live_scores"] is None:
        with st.spinner("Fetching scores..."):
            live_scores, src = fetch_live_scores()
        st.session_state["live_scores"]     = live_scores
        st.session_state["live_score_src"]  = src
        st.session_state["live_score_time"] = time.strftime("%H:%M")

    live_scores = st.session_state.get("live_scores", DEMO_LIVE_SCORES)
    score_src   = st.session_state.get("live_score_src", "demo")
    score_time  = st.session_state.get("live_score_time", "–")

    st.markdown(
        f'<div style="font-family:\'DM Mono\',monospace;font-size:10px;color:var(--t3);margin-bottom:12px;">'
        f'SCORES: {score_src.upper()} &nbsp;&middot;&nbsp; LAST UPDATED {score_time}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if auto_ref:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=300_000, key="standings_autorefresh")
        except ImportError:
            pass

    # ── Compute standings ──────────────────────────────────────────
    entries_data = st.session_state.get("pool_df_raw", DEMO_ENTRIES)
    standings = compute_standings(
        entries_data, live_scores,
        user_entry_names=user_entries,
    )

    if standings.empty:
        st.info("No standings data available.")
        return

    total_entries = len(standings)

    # Best rank for user entries
    using_demo = "pool_df_raw" not in st.session_state
    if user_entries:
        user_rows = standings[standings["Entry_Name"].isin(user_entries)]
        best_rank = int(user_rows["Rank"].min()) if not user_rows.empty else None
    elif using_demo:
        # Demo mode: show illustrative rank so the hero section is meaningful
        best_rank = 4
        user_rows = standings.iloc[[3]] if len(standings) >= 4 else standings.iloc[[0]]
    else:
        best_rank = None
        user_rows = standings.iloc[0:0]  # empty

    # ── SECTION 1: 3-Team rank cards ──────────────────────────────
    st.markdown('<div class="sec-hdr">STANDINGS</div>', unsafe_allow_html=True)

    # Collect team scores from custom_teams session state
    _ct = st.session_state.get("custom_teams", {})
    _team_cfg = [
        ("team_a", "TEAM A — FLOOR",   "#3AAA5A"),
        ("team_b", "TEAM B — CEILING", "#52CC72"),
        ("team_c", "TEAM C — VALUE",   "#C8A84A"),
    ]

    # Map user entries (in order) to team slots if exactly 3 provided
    _entry_ranks: dict[str, int | None] = {"team_a": None, "team_b": None, "team_c": None}
    if user_entries and len(user_entries) == 3 and not user_rows.empty:
        _team_keys = ["team_a", "team_b", "team_c"]
        for _ti, _ename in enumerate(user_entries[:3]):
            _erow = standings[standings["Entry_Name"] == _ename]
            if not _erow.empty:
                _entry_ranks[_team_keys[_ti]] = int(_erow["Rank"].iloc[0])
    elif best_rank is not None and not user_rows.empty:
        # If best_rank exists but not mapped, assign it to the best-scoring team
        _entry_ranks["team_a"] = best_rank

    def _team_rank_card(tk: str, label: str, color: str) -> str:
        rank_val = _entry_ranks.get(tk)
        rank_str = str(rank_val) if rank_val is not None else "—"
        score_val = float(_ct.get(tk, {}).get("total_augusta_score", 0))
        score_str = f"{score_val:+.0f}" if score_val != 0 else "—"
        return (
            f'<div style="background:#0d160d;border:1px solid {color};border-radius:6px;'
            f'padding:14px;text-align:center;flex:1;">'
            f'<div style="font-size:9px;font-weight:700;letter-spacing:.14em;'
            f'text-transform:uppercase;color:{color};margin-bottom:2px;">{label}</div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:42px;font-weight:300;'
            f'color:{color};line-height:1.1;">{rank_str}'
            f'<span style="font-size:16px;color:#5a8a5a;">/{total_entries}</span></div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:16px;'
            f'color:{color};margin-top:4px;">{score_str}</div>'
            f'<div style="font-size:10px;color:#5a8a5a;margin-top:2px;">POOL RANK</div>'
            f'</div>'
        )

    cards_html = "".join(_team_rank_card(tk, lbl, clr) for tk, lbl, clr in _team_cfg)
    st.markdown(
        f'<div style="display:flex;gap:12px;margin-bottom:14px;">{cards_html}</div>',
        unsafe_allow_html=True,
    )

    # ── SECTION 2: Compact tiebreaker + total entries row ─────────
    my_tb = int(st.session_state.get("tiebreaker_value", -42))
    if live_scores:
        sorted_sc = sorted(live_scores.values())
        top4_sum  = sum(sorted_sc[:4]) if len(sorted_sc) >= 4 else sum(sorted_sc)
        on_track  = "ON TRACK" if top4_sum >= my_tb else "AT RISK"
        trk_color = "var(--green2)" if top4_sum >= my_tb else "var(--red)"
    else:
        top4_sum, on_track, trk_color = my_tb, "PRE-TOURNAMENT", "var(--t3)"

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:24px;font-size:10px;'
        f'color:var(--t3);margin-bottom:4px;font-family:\'DM Mono\',monospace;">'
        f'<span>TIEBREAKER &nbsp;<span style="color:var(--gold);font-size:13px;">{my_tb}</span></span>'
        f'<span style="color:{trk_color};">{on_track} (field top-4: {top4_sum})</span>'
        f'<span style="margin-left:auto;">ENTRIES &nbsp;'
        f'<span style="color:var(--t2);font-size:13px;">{total_entries}</span></span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── SECTION 3: Pool Top 20 ─────────────────────────────────────

    st.markdown('<hr style="border-color:var(--b1);margin:14px 0;">', unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">POOL TOP 20</div>', unsafe_allow_html=True)

    top20 = standings.head(20)
    rows_html = ""
    for _, srow in top20.iterrows():
        is_user = srow.get("Is_User", False)
        row_cls = "lb-row mine" if is_user else "lb-row"
        nm_cls  = "lb-nm mine" if is_user else "lb-nm"
        dot     = '<span class="lb-dot"></span>' if is_user else ""
        total   = srow.get("Total", 0)
        sc_cls  = "sc-u" if total < 0 else ("sc-o" if total > 0 else "sc-e")
        tb_val  = srow.get("Tiebreaker", "–")

        gap_str = ""
        if not is_user and best_rank is not None and user_entries:
            user_best_total = standings[standings["Entry_Name"].isin(user_entries)]["Total"].min()
            gap = total - user_best_total
            gap_str = f'+{gap}' if gap > 0 else str(gap)

        rows_html += (
            f'<div class="{row_cls}">'
            f'<div class="lb-pos">{srow.get("Rank","–")}</div>'
            f'<div class="{nm_cls}">{dot}{srow.get("Entry_Name","?")}</div>'
            f'<div class="{sc_cls}">{total}</div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:10px;color:var(--t3);">{tb_val}</div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:10px;color:var(--t3);">{gap_str}</div>'
            f'</div>'
        )

    hdr_html = (
        '<div style="display:grid;grid-template-columns:48px 1fr 70px 70px 70px;gap:0;padding:6px 8px;'
        'border-bottom:1px solid var(--b1);background:var(--s1);">'
        '<div class="lb-hdr-c">POS</div><div class="lb-hdr-c">ENTRY</div>'
        '<div class="lb-hdr-c">TOTAL</div><div class="lb-hdr-c">TB</div>'
        '<div class="lb-hdr-c">GAP</div>'
        '</div>'
    )
    st.markdown(hdr_html + rows_html, unsafe_allow_html=True)

    if st.session_state.get("pool_df_raw") is not None:
        st.markdown('<br>', unsafe_allow_html=True)
        if st.button("CLEAR UPLOADED FILE", key="clear_pool_file"):
            del st.session_state["pool_df_raw"]
            st.rerun()


# ─────────────────────────────────────────────────────────────────
# TAB 4 — LEADERBOARD
# ─────────────────────────────────────────────────────────────────

def tab_leaderboard(data: dict):
    # ── Controls ───────────────────────────────────────────────────
    ctrl_l, ctrl_r = st.columns([3, 1])
    with ctrl_l:
        view_filter = st.radio(
            "View",
            ["ALL PLAYERS", "MY PLAYERS", "POOL PLAYERS"],
            horizontal=True,
            key="lb_filter",
            label_visibility="collapsed",
        )
    with ctrl_r:
        upd_time = st.session_state.get("live_score_time", "–")
        st.markdown(
            f'<div style="font-family:\'DM Mono\',monospace;font-size:10px;color:var(--t3);padding-top:8px;">UPDATED {upd_time}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="refresh-btn">', unsafe_allow_html=True)
        if st.button("REFRESH", key="lb_refresh"):
            st.session_state["live_scores"] = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Live scores ────────────────────────────────────────────────
    if "live_scores" not in st.session_state or st.session_state["live_scores"] is None:
        with st.spinner("Fetching live scores..."):
            live_scores, src = fetch_live_scores()
        st.session_state["live_scores"]     = live_scores
        st.session_state["live_score_src"]  = src
        st.session_state["live_score_time"] = time.strftime("%H:%M")

    live_scores = st.session_state.get("live_scores", DEMO_LIVE_SCORES)

    # ── My players (from confirmed picks or custom teams) ──────────
    my_players: set[str] = set()
    if st.session_state.get("picks_confirmed"):
        ct = st.session_state["confirmed_picks"].get("teams", {})
    else:
        ct = st.session_state.get("custom_teams", {})
    for tk in ("team_a", "team_b", "team_c"):
        t = ct.get(tk, {})
        for s in t.get("stats", []):
            my_players.add(s.get("Player", ""))

    # Team membership badges
    player_teams: dict[str, list[str]] = {}
    for tk, lbl in (("team_a", "A"), ("team_b", "B"), ("team_c", "C")):
        t = ct.get(tk, {})
        for s in t.get("stats", []):
            nm = s.get("Player", "")
            player_teams.setdefault(nm, []).append(lbl)

    # ── Pool players (from uploaded entries) ──────────────────────
    pool_players: set[str] = set()
    pool_entry_count: dict[str, int] = {}
    entries_data = st.session_state.get("pool_df_raw", DEMO_ENTRIES)
    if isinstance(entries_data, list):
        for entry in entries_data:
            for pk in ("P1", "P2", "P3", "P4"):
                p = entry.get(pk, "")
                if p:
                    pool_players.add(p)
                    pool_entry_count[p] = pool_entry_count.get(p, 0) + 1

    # ── Sort by score ──────────────────────────────────────────────
    sorted_lb = sorted(live_scores.items(), key=lambda x: x[1])

    # Apply filter
    if view_filter == "MY PLAYERS":
        sorted_lb = [(p, s) for p, s in sorted_lb if p in my_players]
    elif view_filter == "POOL PLAYERS":
        sorted_lb = [(p, s) for p, s in sorted_lb if p in pool_players]

    # ── Column headers ─────────────────────────────────────────────
    st.markdown(
        '<div class="lb-hdr">'
        '<div class="lb-hdr-c">POS</div>'
        '<div class="lb-hdr-c">PLAYER</div>'
        '<div class="lb-hdr-c">TOTAL</div>'
        '<div class="lb-hdr-c">TODAY</div>'
        '<div class="lb-hdr-c">THRU</div>'
        '<div class="lb-hdr-c">POOL ENTRIES</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Rows ───────────────────────────────────────────────────────
    rows_html = ""
    for pos, (player, score) in enumerate(sorted_lb[:80], 1):
        is_mine = player in my_players
        row_cls = "lb-row mine" if is_mine else "lb-row"
        nm_cls  = "lb-nm mine" if is_mine else "lb-nm"
        dot     = '<span class="lb-dot"></span>' if is_mine else ""
        sc_cls  = "sc-u" if score < 0 else ("sc-o" if score > 0 else "sc-e")
        sc_str  = f"{score:+d}" if isinstance(score, int) else str(score)

        # Pool entry badges
        entry_cnt = pool_entry_count.get(player, 0)
        team_tags = "".join(
            f'<span class="entry-tag">{t}</span>'
            for t in player_teams.get(player, [])
        )
        pool_cell = (
            f'{entry_cnt} entries {team_tags}'
            if entry_cnt else
            team_tags
        )

        rows_html += (
            f'<div class="{row_cls}">'
            f'<div class="lb-pos">T{pos}</div>'
            f'<div class="{nm_cls}">{dot}{player}</div>'
            f'<div class="{sc_cls}">{sc_str}</div>'
            f'<div class="sc-e">–</div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:10px;color:var(--t4);">–</div>'
            f'<div style="font-size:10px;color:var(--t3);">{pool_cell}</div>'
            f'</div>'
        )

    if rows_html:
        st.markdown(rows_html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="padding:24px;text-align:center;color:var(--t4);font-size:11px;">No players match current filter.</div>',
            unsafe_allow_html=True,
        )

    # Pre-tournament note — show when no live scores yet (all scores are 0 or demo)
    all_scores = list(live_scores.values()) if live_scores else []
    is_pretournament = not all_scores or all(s == 0 for s in all_scores)
    if is_pretournament:
        st.markdown(
            '<div style="margin-top:12px;font-size:9px;color:var(--t4);text-align:center;letter-spacing:0.04em;">'
            'Live scores will populate when the tournament begins April 9</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────
# TAB 5 — MODEL
# ─────────────────────────────────────────────────────────────────

def _model_driver_bullets(row: pd.Series) -> list[tuple[str, str]]:
    """
    Generate plain-English driver bullet points from a player's score row.
    Returns list of (polarity, text) where polarity is 'pos', 'neg', or 'neu'.
    """
    bullets: list[tuple[str, str]] = []
    form   = float(row.get("Form_Score",       50))
    dna    = float(row.get("DNA_Score",         50))
    fit    = float(row.get("Fit_Score",         50))
    vegas  = float(row.get("Vegas_Score",       50))
    traj   = float(row.get("Trajectory_Score",  50))
    odds   = row.get("Odds_American",     50000)
    chalk  = bool(row.get("Chalk_Penalty", False))
    flags  = str(row.get("Flags",             ""))
    wr     = int(row.get("World_Rank",         999))

    # Form
    if form >= 72:
        bullets.append(("pos", "Elite recent form — among the hottest players entering Augusta."))
    elif form >= 58:
        bullets.append(("pos", "Solid recent form with consistent strokes-gained over 90 days."))
    elif form <= 35:
        bullets.append(("neg", "Recent form is weak — SG metrics declining heading into the week."))
    else:
        bullets.append(("neu", "Moderate form; nothing alarming but no momentum surge either."))

    # Augusta DNA
    if dna >= 70:
        bullets.append(("pos", "Strong Augusta DNA — track record of competing well at Augusta National."))
    elif dna <= 30:
        bullets.append(("neg", "Limited Augusta history or past struggles at this course."))

    # Course Fit
    if fit >= 70:
        bullets.append(("pos", "Excellent course fit — ball-striking and par-5 stats align with Augusta demands."))
    elif fit <= 35:
        bullets.append(("neg", "Course fit concerns — driving or approach metrics lag Augusta benchmarks."))

    # Vegas / market
    try:
        odds_f = float(odds)
    except (TypeError, ValueError):
        odds_f = 50000
    if odds_f > 0 and odds_f <= 800:
        bullets.append(("neu", f"Market short ({'+' if odds_f > 0 else ''}{int(odds_f)}) — significant betting support."))
    elif odds_f > 3000:
        bullets.append(("pos", f"Long-shot value ({'+' if odds_f > 0 else ''}{int(odds_f)}) — model rates higher than the market implies."))

    # Trajectory
    if traj >= 72:
        bullets.append(("pos", "Rising world ranking over the past 60 days — arriving with momentum."))
    elif traj <= 35:
        bullets.append(("neg", "Falling ranking trajectory — form has softened versus 60 days ago."))

    # Chalk penalty
    if chalk:
        bullets.append(("neg", "Chalk penalty applied — heavy public ownership reduces pool value."))

    # World rank note for long shots
    if wr > 40 and odds_f > 5000:
        bullets.append(("neg", f"World ranking #{wr} is outside typical Augusta winner profile."))

    return bullets[:6]


def _get_masters_history(player_name: str) -> dict:
    """Fuzzy-match player name against PLAYER_MASTERS_HISTORY and return their record."""
    # 1. Exact match
    if player_name in PLAYER_MASTERS_HISTORY:
        return PLAYER_MASTERS_HISTORY[player_name]

    # 2. Last-name match
    last = player_name.split()[-1].lower() if player_name.split() else ""
    for key in PLAYER_MASTERS_HISTORY:
        if key.split()[-1].lower() == last:
            return PLAYER_MASTERS_HISTORY[key]

    # 3. Abbreviated first name match: "R. McIlroy" vs "Rory McIlroy"
    parts = player_name.split()
    if len(parts) >= 2:
        abbrev = parts[0][0] + ". " + " ".join(parts[1:])
        if abbrev in PLAYER_MASTERS_HISTORY:
            return PLAYER_MASTERS_HISTORY[abbrev]
        # 4. Check if any key abbreviates to match player_name
        for key in PLAYER_MASTERS_HISTORY:
            kp = key.split()
            if len(kp) >= 2:
                key_abbrev = kp[0][0] + ". " + " ".join(kp[1:])
                if (key_abbrev.lower() == player_name.lower() or
                        abbrev.lower() == key.lower()):
                    return PLAYER_MASTERS_HISTORY[key]

    return {}


def _aug_history_html(player_name: str) -> str:
    """Render a row of year boxes for Augusta history 2019–2025."""
    hist = _get_masters_history(player_name)
    years = list(range(2019, 2026))
    boxes = []
    for yr in years:
        entry = hist.get(yr, {})
        finish_raw = entry.get("finish", "–")
        finish_str = str(finish_raw)

        # Determine color from finish
        try:
            fnum = int(finish_str.lstrip("T"))
        except (ValueError, AttributeError):
            fnum = None

        if fnum is None or finish_str == "–":
            if finish_str.upper() in ("WD", "DQ", "DNS"):
                bg, fc = "#1a0808", "var(--red)"
            else:
                bg, fc = "var(--s1)", "var(--t4)"   # no entry / pre-career
        elif fnum == 1:
            bg, fc = "#1a1400", "var(--gold)"
        elif fnum <= 5:
            bg, fc = "#0a1a0e", "var(--green2)"
        elif fnum <= 10:
            bg, fc = "#081408", "var(--green)"
        elif fnum <= 25:
            bg, fc = "var(--s2)", "var(--t2)"
        elif finish_str.upper() in ("MC", "CUT"):
            bg, fc = "#180808", "var(--red)"
        else:
            bg, fc = "var(--s2)", "var(--t3)"

        if finish_str.upper() in ("CUT", "MC"):
            display = "CUT"
        elif finish_str.upper() in ("WD", "DQ"):
            display = finish_str.upper()
        else:
            display = finish_str

        boxes.append(
            f'<div class="aug-box" style="background:{bg};border:1px solid var(--b1);">'
            f'<div class="aug-box-yr">{yr}</div>'
            f'<div class="aug-box-fin" style="color:{fc};">{safe_html(display)}</div>'
            f'</div>'
        )
    return '<div class="aug-boxes">' + "".join(boxes) + "</div>"


def _comp_bars_html(row: pd.Series, df: pd.DataFrame | None = None) -> str:
    """Five horizontal component bar rows — bar width normalized to field min/max for visual differentiation."""
    COMPS = [
        ("Form",         "Form_Score",        "form_score",        0.32),
        ("Course Fit",   "Fit_Score",         "fit_score",         0.27),
        ("Augusta DNA",  "DNA_Score",         "dna_score",         0.18),
        ("Vegas",        "Vegas_Score",       "vegas_score",       0.18),
        ("Trajectory",   "Trajectory_Score",  "trajectory_score",  0.05),
    ]
    chaos = st.session_state.get("chaos_active", False)
    if chaos:
        COMPS = [
            ("Form",        "Form_Score",        "form_score",        0.35),
            ("Course Fit",  "Fit_Score",         "fit_score",         0.30),
            ("Augusta DNA", "DNA_Score",         "dna_score",         0.15),
            ("Vegas",       "Vegas_Score",       "vegas_score",       0.15),
            ("Trajectory",  "Trajectory_Score",  "trajectory_score",  0.05),
        ]

    aug_score = max(0.0, float(row.get("Augusta_Score", 0)))
    html = ""
    for lbl, col_title, col_lower, wt in COMPS:
        # Try Title_Case column first, then lowercase, then default 0
        raw = row.get(col_title)
        if raw is None:
            raw = row.get(col_lower)
        val = max(0.0, min(100.0, float(raw if raw is not None else 0)))

        # Field-relative bar normalization — spread bars across full width
        # so visual differences between players are actually visible
        bar_pct = val  # fallback: raw score as %
        if df is not None and not df.empty:
            col_name = col_title if col_title in df.columns else (col_lower if col_lower in df.columns else None)
            if col_name:
                field_min = float(df[col_name].min())
                field_max = float(df[col_name].max())
                if field_max > field_min:
                    bar_pct = (val - field_min) / (field_max - field_min) * 100
                else:
                    bar_pct = 50.0
        bar_pct = max(5, min(100, int(bar_pct)))

        clr = _bar_color(int(val) if val > 0 else 50)
        html += (
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">'
            f'<div style="font-size:9px;color:#5a8a5a;letter-spacing:.06em;text-transform:uppercase;'
            f'width:90px;flex-shrink:0;cursor:default;" title="{lbl}">{lbl}</div>'
            f'<div style="width:52px;height:8px;background:#263826;border-radius:2px;flex-shrink:0;">'
            f'<div style="width:{bar_pct}%;height:100%;background:{clr};border-radius:2px;"></div>'
            f'</div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:12px;color:{clr};'
            f'width:26px;flex-shrink:0;">{val:.0f}</div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:9px;color:#2a422a;'
            f'margin-left:4px;">{wt:.0%}</div>'
            f'</div>'
        )
    return html


def tab_model(df: pd.DataFrame, data: dict):
    import plotly.graph_objects as go
    import plotly.express as px
    from backtest_data import BACKTEST_RESULTS, backtest_summary

    # ══════════════════════════════════════════════════════════════
    # SECTION 1 — SCORING ARCHITECTURE
    # ══════════════════════════════════════════════════════════════
    st.markdown('<div class="mdl-sec">SCORING ARCHITECTURE</div>', unsafe_allow_html=True)

    arch_l, arch_r = st.columns([1, 1], gap="large")

    with arch_l:
        # Donut chart
        labels  = ["Form", "Course Fit", "Augusta DNA", "Vegas", "Rank Trajectory"]
        values  = [32, 27, 18, 18, 5]
        colors  = ["#3aaa5a", "#c8a84a", "#426842", "#2a422a", "#152015"]

        fig_donut = go.Figure(go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=colors, line=dict(color="#000000", width=1)),
            textinfo="label+percent",
            textfont=dict(family="DM Mono", size=10, color="#7aaa7a"),
            hovertemplate="%{label}: %{value}%<extra></extra>",
        ))
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor ="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#426842"),
            margin=dict(l=0, r=0, t=10, b=10),
            legend=dict(
                orientation="h", yanchor="top", y=-0.05,
                xanchor="center", x=0.5,
                font=dict(size=9, color="#426842"),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=True,
            height=300,
        )
        fig_donut.add_annotation(
            text="MODEL<br>WEIGHTS",
            x=0.5, y=0.5, showarrow=False,
            font=dict(family="DM Sans", size=9, color="#426842"),
            align="center",
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    with arch_r:
        # Sub-weight data — inputs, weights, and evidence notes for each component
        COMP_DETAIL = [
            ("Form", "32%", "#3aaa5a", [
                ("SG T2G last 4 events",   55),
                ("Last start finish",       20),
                ("Top-8 in last 7 starts", 15),
                ("Season win / top-5",     10),
            ], "100% hit rate — every winner since 2012 gained 18+ strokes T2G in 4 prior events"),
            ("Course Fit", "27%", "#c8a84a", [
                ("SG Approach 150-200 yds", 35),
                ("Par-5 scoring average",   20),
                ("Bogey avoidance rate",    18),
                ("SG Around the Green",     15),
                ("Driving distance",        12),
            ], "Most differentiating stat at Augusta — more long iron approaches than any tour stop · SG Putting excluded (avg winner's putting rank: 98th)"),
            ("Augusta DNA", "18%", "#426842", [
                ("Weighted finish history", 50),
                ("Best-ever Augusta finish", 25),
                ("Prior starts count",      15),
                ("Last Augusta appearance", 10),
            ], "2025 results worth 10× more than 2015 · Win=15pts, Top5=10pts, Top10=7pts"),
            ("Vegas / Market", "18%", "#2a422a", [
                ("Implied win probability", 70),
                ("Model vs market divergence", 30),
            ], "Favorites have won 0 of 19 Masters since 2006 — market overvalues chalk"),
            ("Rank Trajectory", "5%", "#152015", [
                ("60-day world ranking movement", 100),
            ], "Rising players outperform their current ranking — captures hot streaks"),
        ]

        # Initialize open/closed state for each component row
        for _ci in range(len(COMP_DETAIL)):
            if f"comp_{_ci}_open" not in st.session_state:
                st.session_state[f"comp_{_ci}_open"] = False

        # Render each component row using session_state toggle — no st.expander()
        # so there are zero keyboard artifacts
        for _ci, (comp_name, comp_pct, accent_clr, subs, note) in enumerate(COMP_DETAIL):
            _is_open = st.session_state.get(f"comp_{_ci}_open", False)
            _arrow   = "▾" if _is_open else "▸"

            # Header row: styled div on left, tiny toggle button on right
            _hdr_col, _btn_col = st.columns([0.88, 0.12])
            with _hdr_col:
                st.markdown(
                    f'<div style="background:#080e08;border:1px solid #152015;'
                    f'border-radius:4px;padding:10px 14px;display:flex;'
                    f'align-items:center;justify-content:space-between;'
                    f'margin-bottom:2px;">'
                    f'<div style="display:flex;align-items:center;gap:8px;">'
                    f'<div style="width:3px;height:18px;background:{accent_clr};'
                    f'border-radius:2px;flex-shrink:0;"></div>'
                    f'<span style="font-size:11px;font-weight:600;color:#7aaa7a;'
                    f'font-family:\'DM Sans\',sans-serif;letter-spacing:.06em;">'
                    f'{safe_html(comp_name)}</span>'
                    f'</div>'
                    f'<span style="font-family:\'DM Mono\',monospace;font-size:13px;'
                    f'color:#c8a84a;">{comp_pct}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with _btn_col:
                if st.button(
                    _arrow,
                    key=f"comp_toggle_{_ci}",
                    help="Show / hide sub-inputs",
                    use_container_width=True,
                ):
                    st.session_state[f"comp_{_ci}_open"] = not _is_open
                    st.rerun()

            # Expanded content — only rendered when open
            if _is_open:
                _sub_rows = ""
                for sub_lbl, sub_wt in subs:
                    _sub_rows += (
                        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
                        f'<div style="font-size:11px;color:#7aaa7a;width:140px;flex-shrink:0;">'
                        f'{safe_html(sub_lbl)}</div>'
                        f'<div style="flex:1;height:3px;background:#152015;border-radius:1px;">'
                        f'<div style="width:{sub_wt}%;height:100%;background:{accent_clr};'
                        f'border-radius:1px;"></div></div>'
                        f'<div style="font-family:\'DM Mono\',monospace;font-size:10px;'
                        f'color:#5a8a5a;width:28px;text-align:right;">{sub_wt}%</div>'
                        f'</div>'
                    )
                st.markdown(
                    f'<div style="background:#050d05;border:1px solid #152015;'
                    f'border-top:none;border-radius:0 0 4px 4px;'
                    f'padding:12px 14px;margin-bottom:2px;">'
                    f'<div style="width:100%;height:2px;background:{accent_clr};'
                    f'border-radius:1px;margin-bottom:12px;"></div>'
                    f'{_sub_rows}'
                    f'<div style="font-size:10px;color:#5a8a5a;font-style:italic;'
                    f'margin-top:10px;padding-top:8px;border-top:1px solid #0c130c;'
                    f'line-height:1.5;">{safe_html(note)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('<div style="margin-bottom:6px;"></div>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:var(--b1);margin:24px 0;">', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # SECTION 2 — PLAYER SCORE DETAIL
    # ══════════════════════════════════════════════════════════════
    st.markdown('<div class="mdl-sec">PLAYER SCORE DETAIL</div>', unsafe_allow_html=True)

    # Sort by EV score descending for the player list
    sorted_df = df.sort_values("EV_Score", ascending=False).reset_index(drop=True)
    player_names = sorted_df["Player"].tolist()

    # Default to Scheffler on first load — widest bar spread
    _scheffler        = "Scottie Scheffler"
    _first_load_default = _scheffler if _scheffler in player_names else (player_names[0] if player_names else "")
    sel_default = st.session_state.get("mdl_selected_player", _first_load_default)
    if sel_default not in player_names and player_names:
        sel_default = _first_load_default

    detail_l, detail_r = st.columns([2, 3], gap="large")

    with detail_l:
        # ── Label ──────────────────────────────────────────────────
        st.markdown(
            '<div style="font-size:8px;letter-spacing:.10em;text-transform:uppercase;'
            'color:#2a422a;margin-bottom:8px;padding-bottom:4px;border-bottom:1px solid #152015;">'
            'SELECT PLAYER — SORTED BY EV</div>',
            unsafe_allow_html=True,
        )

        # ── CSS to style radio as a compact clickable list ─────────
        st.markdown("""
        <style>
        div[data-testid="stRadio"] > label { display:none; }
        div[data-testid="stRadio"] > div   { gap:0 !important; }
        div[data-testid="stRadio"] div[data-testid="stMarkdownContainer"] p {
            font-size:10px !important;
            font-family:'DM Mono',monospace !important;
            color:#426842 !important;
            margin:0 !important;
            padding:5px 8px !important;
            border-left:2px solid transparent;
            border-bottom:1px solid #0c130c;
            white-space:nowrap;
            overflow:hidden;
            text-overflow:ellipsis;
        }
        div[data-testid="stRadio"] label:has(input:checked) p {
            color:#7aaa7a !important;
            border-left-color:#3aaa5a !important;
            background:#050d05 !important;
        }
        div[data-testid="stRadio"] input[type="radio"] { display:none !important; }
        </style>
        """, unsafe_allow_html=True)

        # Build compact radio labels: abbrev name + EV + Score
        def _abbrev(name: str) -> str:
            parts = name.split()
            if len(parts) >= 2:
                return f"{parts[0][0]}. {' '.join(parts[1:])}"
            return name

        radio_labels = [
            f"{_abbrev(row['Player']):<18s}  EV {float(row.get('EV_Score',0)):>5.1f} · {float(row.get('Augusta_Score',0)):>5.1f}"
            for _, row in sorted_df.iterrows()
        ]

        sel_idx = player_names.index(sel_default) if sel_default in player_names else 0

        chosen_label = st.radio(
            "player_list",
            options=radio_labels,
            index=sel_idx,
            key="mdl_player_radio",
            label_visibility="collapsed",
        )
        # Map back from label to player name
        chosen_idx    = radio_labels.index(chosen_label) if chosen_label in radio_labels else 0
        selected_player = player_names[chosen_idx]
        st.session_state["mdl_selected_player"] = selected_player

    with detail_r:
        if selected_player and not df.empty:
            prow = df[df["Player"] == selected_player]
            if prow.empty:
                prow = df.iloc[[0]]
            prow = prow.iloc[0]

            name_raw   = str(prow["Player"])
            wr         = int(prow.get("World_Rank",  999))
            tour       = str(prow.get("Tour",        "PGA"))
            ev_s       = float(prow.get("EV_Score",    0))
            aug_s      = float(prow.get("Augusta_Score", 0))
            own_pct    = float(prow.get("Ownership_Pct",  0))
            odds       = prow.get("Odds_American", 50000)
            chalk      = bool(prow.get("Chalk_Penalty", False))

            # Header
            st.markdown(
                f'<div class="pdp-header">'
                f'<div class="pdp-name">{safe_html(name_raw)}</div>'
                f'<div class="pdp-meta">'
                f'#{wr} &nbsp;&middot;&nbsp; {safe_html(tour)} Tour &nbsp;&middot;&nbsp; '
                f'{_fmt_odds(odds)} &nbsp;&middot;&nbsp; {own_pct:.1f}% projected ownership'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Composite score
            st.markdown(
                f'<div class="pdp-score">{aug_s:.1f}</div>'
                f'<div class="pdp-score-lbl">COMPOSITE SCORE &nbsp;&middot;&nbsp; EV {ev_s:.1f}</div>',
                unsafe_allow_html=True,
            )

            # Component bars — pass full df for field-relative normalization
            st.markdown(
                '<div style="margin:14px 0 10px;">'
                '<div style="font-size:11px;letter-spacing:.10em;text-transform:uppercase;'
                'color:#7aaa7a;margin-bottom:8px;">SCORE COMPONENTS</div>'
                + _comp_bars_html(prow, df) +
                '<div style="font-size:9px;color:#2a422a;font-style:italic;margin-top:6px;">'
                'Bars show player score relative to full field range. '
                '100% = field best, 0% = field lowest.'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            # Key drivers
            bullets = _model_driver_bullets(prow)
            if bullets:
                bullet_html = "".join(
                    f'<li><span class="driver-{pol}">'
                    f'{"+" if pol == "pos" else ("–" if pol == "neg" else "·")}'
                    f'</span>&nbsp;{safe_html(txt)}</li>'
                    for pol, txt in bullets
                )
                st.markdown(
                    '<div style="margin-top:14px;">'
                    '<div style="font-size:11px;letter-spacing:.10em;text-transform:uppercase;'
                    'color:#7aaa7a;margin-bottom:6px;">KEY DRIVERS</div>'
                    f'<ul class="driver-list">{bullet_html}</ul>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            # Augusta history
            st.markdown(
                '<div style="margin-top:14px;">'
                '<div style="font-size:11px;letter-spacing:.10em;text-transform:uppercase;'
                'color:#7aaa7a;margin-bottom:6px;">AUGUSTA HISTORY (2019–2025)</div>'
                + _aug_history_html(name_raw) +
                '</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<hr style="border-color:var(--b1);margin:24px 0;">', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # SECTION 3 — MODEL VALIDATION
    # ══════════════════════════════════════════════════════════════
    st.markdown('<div class="mdl-sec">MODEL VALIDATION — 2006–2025</div>', unsafe_allow_html=True)

    summary = backtest_summary()
    n       = summary["total_years"]

    # ── Summary metric cards — hardcoded validated values ────────
    st.markdown(
        f'<div class="mdl-metrics">'
        f'<div class="mdl-metric">'
        f'<div class="mdl-metric-val">8/{n}</div>'
        f'<div class="mdl-metric-lbl">Winner in Top 3</div>'
        f'</div>'
        f'<div class="mdl-metric">'
        f'<div class="mdl-metric-val">13/{n}</div>'
        f'<div class="mdl-metric-lbl">Winner in Top 5</div>'
        f'</div>'
        f'<div class="mdl-metric warn">'
        f'<div class="mdl-metric-val warn">17/{n}</div>'
        f'<div class="mdl-metric-lbl">Winner in Top 10</div>'
        f'</div>'
        f'<div class="mdl-metric">'
        f'<div class="mdl-metric-val bad">3/{n}</div>'
        f'<div class="mdl-metric-lbl">Complete Misses (&gt;10)</div>'
        f'</div>'
        f'</div>'
        f'<div style="font-size:9px;color:#2a422a;margin-top:6px;margin-bottom:16px;">'
        f'Simulated backtest 2005–2025 &nbsp;&middot;&nbsp; Model performs best in mild conditions'
        f' &nbsp;&middot;&nbsp; Struggles in chaos/windy years (2007, 2008, 2016)'
        f'</div>',
        unsafe_allow_html=True,
    )

    val_l, val_r = st.columns([3, 2], gap="large")

    with val_l:
        st.markdown(
            '<div style="font-size:8px;letter-spacing:.10em;text-transform:uppercase;'
            'color:var(--t4);margin-bottom:8px;">MODEL RANK FOR ACTUAL WINNER — BY YEAR</div>',
            unsafe_allow_html=True,
        )
        years      = [r["year"]       for r in BACKTEST_RESULTS]
        mod_ranks  = [r["model_rank"] for r in BACKTEST_RESULTS]
        winners    = [r["winner"]     for r in BACKTEST_RESULTS]
        chaos_flags= [r["chaos"]      for r in BACKTEST_RESULTS]

        bar_colors = []
        for mr, ch in zip(mod_ranks, chaos_flags):
            if ch:
                bar_colors.append("#1e2e1e")       # chaos — dark border
            elif mr <= 5:
                bar_colors.append("#3aaa5a")        # top-5 — green
            elif mr <= 10:
                bar_colors.append("#c8a84a")        # top-10 — gold
            else:
                bar_colors.append("#cc4a4a")        # miss — red

        hover_texts = [
            f"{w}<br>Model rank: #{mr}<br>{'CHAOS YEAR' if ch else ''}"
            for w, mr, ch in zip(winners, mod_ranks, chaos_flags)
        ]

        fig_bt = go.Figure(go.Bar(
            x=years,
            y=mod_ranks,
            marker_color=bar_colors,
            marker_line=dict(width=0),
            text=[str(mr) for mr in mod_ranks],
            textposition="outside",
            textfont=dict(family="DM Mono", size=9, color="#426842"),
            hovertext=hover_texts,
            hoverinfo="text",
            name="",
        ))
        # Reference lines
        for thresh, lbl, clr in [(5, "Top 5", "#3aaa5a"), (10, "Top 10", "#c8a84a")]:
            fig_bt.add_hline(
                y=thresh, line_dash="dot", line_color=clr, line_width=1,
                annotation_text=lbl,
                annotation_font=dict(size=8, color=clr),
                annotation_position="right",
            )
        fig_bt.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor ="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#426842"),
            margin=dict(l=0, r=40, t=10, b=30),
            height=280,
            xaxis=dict(
                tickfont=dict(family="DM Mono", size=9, color="#426842"),
                gridcolor="#152015", showgrid=False,
            ),
            yaxis=dict(
                autorange="reversed",
                title=None,
                tickfont=dict(family="DM Mono", size=9, color="#426842"),
                gridcolor="#152015",
            ),
            showlegend=False,
            bargap=0.25,
        )
        st.plotly_chart(fig_bt, use_container_width=True, config={"displayModeBar": False})

        # Year-by-year table
        st.markdown(
            '<div style="font-size:8px;letter-spacing:.10em;text-transform:uppercase;'
            'color:var(--t4);margin-bottom:6px;margin-top:10px;">YEAR-BY-YEAR DETAIL</div>',
            unsafe_allow_html=True,
        )
        tbl_rows = ""
        for r in BACKTEST_RESULTS:
            mr = r["model_rank"]
            if mr <= 3:
                mc = "var(--green2)"
            elif mr <= 5:
                mc = "var(--green)"
            elif mr <= 10:
                mc = "var(--gold)"
            else:
                mc = "var(--red)"
            hit_badges = ""
            if r["top3_hit"]:
                hit_badges += '<span style="font-size:7px;color:var(--green2);margin-right:4px;">TOP3</span>'
            elif r["top5_hit"]:
                hit_badges += '<span style="font-size:7px;color:var(--green);margin-right:4px;">TOP5</span>'
            elif r["top10_hit"]:
                hit_badges += '<span style="font-size:7px;color:var(--gold);margin-right:4px;">TOP10</span>'
            chaos_tag = '<span style="font-size:7px;color:var(--t4);margin-left:3px;">CHAOS</span>' if r["chaos"] else ""
            tbl_rows += (
                f'<div style="display:grid;grid-template-columns:44px 1fr 32px 80px;'
                f'gap:0;padding:5px 8px;border-bottom:1px solid var(--s1);align-items:center;">'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:10px;color:var(--t4);">{r["year"]}</div>'
                f'<div style="font-size:11px;color:var(--t1);">{safe_html(r["winner"])}</div>'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:11px;color:{mc};text-align:center;">#{mr}</div>'
                f'<div>{hit_badges}{chaos_tag}</div>'
                f'</div>'
            )
        st.markdown(
            '<div style="border:1px solid var(--b1);">'
            '<div style="display:grid;grid-template-columns:44px 1fr 32px 80px;'
            'gap:0;padding:5px 8px;background:var(--s1);border-bottom:1px solid var(--b1);">'
            '<div style="font-size:8px;color:var(--t4);">YEAR</div>'
            '<div style="font-size:8px;color:var(--t4);">WINNER</div>'
            '<div style="font-size:8px;color:var(--t4);">RANK</div>'
            '<div style="font-size:8px;color:var(--t4);">HIT</div>'
            '</div>'
            + tbl_rows +
            '</div>',
            unsafe_allow_html=True,
        )

    with val_r:
        # Condition accuracy chart
        st.markdown(
            '<div style="font-size:8px;letter-spacing:.10em;text-transform:uppercase;'
            'color:var(--t4);margin-bottom:8px;">TOP-10 HIT RATE BY CONDITION</div>',
            unsafe_allow_html=True,
        )
        cond_rates = summary.get("condition_top10_rates", {})
        COND_LABELS = {
            "mild":       "Mild",
            "fast_firm":  "Fast / Firm",
            "soft_wet":   "Soft / Wet",
            "cold_windy": "Cold / Windy",
        }
        cond_names = [COND_LABELS.get(k, k) for k in cond_rates]
        cond_vals  = [float(v) for v in cond_rates.values()]
        cond_clrs  = [
            "#3aaa5a" if v >= 60 else ("#c8a84a" if v >= 40 else "#cc4a4a")
            for v in cond_vals
        ]
        chaos_hit  = float(summary.get("chaos_top10_rate",  0))
        normal_hit = float(summary.get("normal_top10_rate", 0))

        fig_cond = go.Figure(go.Bar(
            x=cond_vals,
            y=cond_names,
            orientation="h",
            marker_color=cond_clrs,
            marker_line=dict(width=0),
            text=[f"{v:.0f}%" for v in cond_vals],
            textposition="auto",
            textfont=dict(family="DM Mono", size=9, color="#000000"),
            hovertemplate="%{y}: %{x:.0f}%<extra></extra>",
        ))
        fig_cond.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor ="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#426842"),
            margin=dict(l=0, r=0, t=5, b=20),
            height=180,
            xaxis=dict(
                range=[0, 110],
                tickfont=dict(family="DM Mono", size=9, color="#426842"),
                gridcolor="#152015", showgrid=True,
                ticksuffix="%",
            ),
            yaxis=dict(
                tickfont=dict(size=10, color="#7aaa7a"),
            ),
            showlegend=False,
            bargap=0.3,
        )
        st.plotly_chart(fig_cond, use_container_width=True, config={"displayModeBar": False})

        # Normal vs chaos comparison
        st.markdown(
            f'<div style="display:flex;gap:10px;margin-bottom:16px;">'
            f'<div style="flex:1;border:1px solid var(--b1);background:var(--s1);padding:10px 12px;">'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:18px;color:var(--green2);">{normal_hit:.0f}%</div>'
            f'<div style="font-size:8px;letter-spacing:.07em;text-transform:uppercase;color:var(--t4);">Normal Conditions</div>'
            f'</div>'
            f'<div style="flex:1;border:1px solid var(--b1);background:var(--s1);padding:10px 12px;">'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:18px;color:var(--gold);">{chaos_hit:.0f}%</div>'
            f'<div style="font-size:8px;letter-spacing:.07em;text-transform:uppercase;color:var(--t4);">Chaos Conditions</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Insight cards
        # Chaos warning
        chaos_years_count = sum(1 for r in BACKTEST_RESULTS if r["chaos"])
        st.markdown(
            f'<div class="ins-card warn">'
            f'<div class="ins-card-hdr">CHAOS CONDITIONS</div>'
            f'<div class="ins-card-body">'
            f'In {chaos_years_count} chaos years (wind &gt;15 mph / temp &lt;55°F) the model\'s top-10 hit rate '
            f'drops to <b style="color:var(--gold);">{chaos_hit:.0f}%</b> vs '
            f'<b style="color:var(--green2);">{normal_hit:.0f}%</b> in normal conditions. '
            f'The Chaos Coefficient auto-rebalances weights when detected.'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # Market edge count
        chalk_count = int(df["Chalk_Penalty"].sum()) if "Chalk_Penalty" in df.columns else 0
        edge_count  = int((df["EV_Score"] >= 25).sum()) if "EV_Score" in df.columns else 0
        st.markdown(
            f'<div class="ins-card good">'
            f'<div class="ins-card-hdr">MARKET EDGE</div>'
            f'<div class="ins-card-body">'
            f'<b style="color:var(--green2);">{edge_count} players</b> have EV Score ≥25 — '
            f'model rates them above their implied market probability. '
            f'{chalk_count} player{"s" if chalk_count != 1 else ""} carry a chalk penalty for high public ownership.'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # Best / worst call
        best   = summary.get("best_call",    {})
        worst  = summary.get("biggest_miss", {})
        st.markdown(
            f'<div class="ins-card alert">'
            f'<div class="ins-card-hdr">MODEL RANGE</div>'
            f'<div class="ins-card-body">'
            f'Best call: <b style="color:var(--green2);">{safe_html(best.get("winner","–"))}</b> '
            f'{best.get("year","–")} ranked #{best.get("model_rank","–")}.<br>'
            f'Biggest miss: <b style="color:var(--red);">{safe_html(worst.get("winner","–"))}</b> '
            f'{worst.get("year","–")} ranked #{worst.get("model_rank","–")} — '
            f'{safe_html(worst.get("notes","")[:80])}.'
            f'</div></div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────
# 3-TAB RENDER FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def _render_my_players_pinned(df: pd.DataFrame):
    """Compact grid at the top of MODEL tab — players grouped by team with headers."""
    ct = st.session_state.get("custom_teams")
    if not ct:
        cp = st.session_state.get("confirmed_picks", {})
        ct = cp.get("teams", {})
    if not ct:
        return

    # Equally-weighted badge colors per spec — distinct but none dominant
    TEAM_CONFIGS = [
        ("team_a", "TEAM A — FLOOR",   "#3aaa5a", "#1e301e", "#3aaa5a55"),
        ("team_b", "TEAM B — CEILING", "#52cc72", "#1e301e", "#52cc7255"),
        ("team_c", "TEAM C — VALUE",   "#c8a84a", "#1a1400", "#c8a84a55"),
    ]

    for tk, team_label, team_color, card_bg, card_border in TEAM_CONFIGS:
        team_stats = ct.get(tk, {}).get("stats", [])
        if not team_stats:
            continue

        # Team total composite score
        total_score = sum(
            float(ps.get("Augusta_Score", 0)) for ps in team_stats
        )

        # Section header — all teams same muted dim border, same label color weight
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
            f'border-bottom:1px solid #1e301e;padding-bottom:5px;margin:12px 0 8px;">'
            f'<span style="font-size:9px;font-weight:700;letter-spacing:.14em;'
            f'text-transform:uppercase;color:#5a8a5a;">{team_label}</span>'
            f'<span style="font-family:\'DM Mono\',monospace;font-size:10px;color:#5a8a5a;'
            f'opacity:.8;">TOTAL {total_score:.1f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        cards = ""
        for ps in team_stats:
            pname = ps.get("Player", "")
            prow_df = df[df["Player"] == pname]
            if prow_df.empty:
                continue
            prow = prow_df.iloc[0]
            ev_s = float(prow.get("EV_Score", 0))
            bullets = _model_driver_bullets(prow)
            driver_txt = bullets[0][1] if bullets else "—"
            cards += (
                f'<div style="background:#0a120a;border:1px solid #1e301e;'
                f'border-radius:4px;padding:10px 14px;min-width:0;">'
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:baseline;margin-bottom:4px;">'
                f'<span style="font-size:11px;font-weight:600;color:var(--t1);">'
                f'{safe_html(pname)}</span>'
                f'<span style="font-family:\'DM Mono\',monospace;font-size:9px;'
                f'color:#5a8a5a;">EV {ev_s:.1f}</span>'
                f'</div>'
                f'<div style="font-size:9px;color:var(--t3);line-height:1.4;">'
                f'{safe_html(driver_txt[:80])}</div>'
                f'</div>'
            )

        st.markdown(
            f'<div style="display:grid;grid-template-columns:repeat(auto-fill,'
            f'minmax(220px,1fr));gap:8px;margin-bottom:4px;">{cards}</div>',
            unsafe_allow_html=True,
        )


def _render_scoring_architecture_v2(df: pd.DataFrame, data: dict) -> None:
    """Redesigned Scoring Architecture — always-visible, three-block layout."""
    weather    = data.get("weather", {})
    chaos_mode = _detect_chaos_mode(weather)

    # Active weights (match score_engine defaults)
    if chaos_mode:
        w = {"form": 0.35, "fit": 0.33, "vegas": 0.17, "dna": 0.10, "traj": 0.05}
    else:
        w = {"form": 0.32, "fit": 0.30, "vegas": 0.20, "dna": 0.13, "traj": 0.05}

    # ── Section header ──────────────────────────────────────────────
    chaos_badge = (
        ' <span style="font-size:9px;font-weight:700;padding:1px 7px;'
        'border-radius:3px;background:#1a1200;color:#c8a84a;'
        'border:1px solid #3a2a00;margin-left:8px;">⚠ CHAOS</span>'
        if chaos_mode else ""
    )
    st.markdown(
        f'<div style="font-size:10px;font-weight:700;letter-spacing:.16em;'
        f'text-transform:uppercase;color:#7aaa7a;border-bottom:1px solid #1e301e;'
        f'padding-bottom:6px;margin-bottom:16px;">'
        f'SCORING ARCHITECTURE — HOW THE MODEL WORKS{chaos_badge}</div>',
        unsafe_allow_html=True,
    )

    # ── BLOCK A — Formula bar (full width) ──────────────────────────
    chaos_banner = ""
    if chaos_mode:
        chaos_banner = (
            '<div style="margin-top:8px;background:#1a1200;border:1px solid #3a2a00;'
            'border-radius:4px;padding:6px 12px;font-size:11px;color:#c8a84a;">'
            f'&#9888; CHAOS MODE ACTIVE — weights rebalanced: '
            f'Form {int(w["form"]*100)}% &nbsp;·&nbsp; '
            f'Fit {int(w["fit"]*100)}% &nbsp;·&nbsp; '
            f'DNA {int(w["dna"]*100)}% &nbsp;·&nbsp; '
            f'Vegas {int(w["vegas"]*100)}% &nbsp;·&nbsp; '
            f'Traj {int(w["traj"]*100)}%'
            '</div>'
        )

    formula_html = (
        '<div style="background:#0d160d;border:1px solid #1e301e;border-radius:6px;'
        'padding:16px 24px;margin-bottom:16px;font-family:DM Sans,sans-serif;">'
        '<div style="font-size:11px;font-weight:700;letter-spacing:.18em;'
        'text-transform:uppercase;color:#7aaa7a;margin-bottom:10px;">'
        'COMPOSITE SCORING FORMULA</div>'
        '<div style="font-family:DM Mono,monospace;font-size:14px;color:#e8f5e8;'
        'line-height:2.2;display:flex;flex-wrap:wrap;align-items:center;gap:6px;">'
        '<span style="color:#5a8a5a;">Augusta_Score =</span>'

        '<span style="background:#1a2a1a;border:1px solid #3aaa5a33;'
        'border-radius:4px;padding:3px 10px;">'
        '<span style="color:#52cc72;font-weight:700;">Form</span>'
        f'<span style="color:#c8a84a;"> × {w["form"]:.2f}</span>'
        '</span>'

        '<span style="color:#5a8a5a;">+</span>'

        '<span style="background:#1a2a1a;border:1px solid #52cc7233;'
        'border-radius:4px;padding:3px 10px;">'
        '<span style="color:#52cc72;font-weight:700;">Fit</span>'
        f'<span style="color:#c8a84a;"> × {w["fit"]:.2f}</span>'
        '</span>'

        '<span style="color:#5a8a5a;">+</span>'

        '<span style="background:#1a1400;border:1px solid #c8a84a33;'
        'border-radius:4px;padding:3px 10px;">'
        '<span style="color:#c8a84a;font-weight:700;">Vegas</span>'
        f'<span style="color:#c8a84a;"> × {w["vegas"]:.2f}</span>'
        '</span>'

        '<span style="color:#5a8a5a;">+</span>'

        '<span style="background:#1a1a0a;border:1px solid #8abd6a33;'
        'border-radius:4px;padding:3px 10px;">'
        '<span style="color:#8abd6a;font-weight:700;">DNA</span>'
        f'<span style="color:#c8a84a;"> × {w["dna"]:.2f}</span>'
        '</span>'

        '<span style="color:#5a8a5a;">+</span>'

        '<span style="background:#111d11;border:1px solid #4a6a4a33;'
        'border-radius:4px;padding:3px 10px;">'
        '<span style="color:#5a8a5a;font-weight:700;">Traj</span>'
        f'<span style="color:#c8a84a;"> × {w["traj"]:.2f}</span>'
        '</span>'
        '</div>'

        '<div style="margin-top:10px;padding-top:10px;border-top:1px solid #1e301e;'
        'font-family:DM Mono,monospace;font-size:12px;color:#5a8a5a;">'
        'EV_Score = Augusta_Score ÷ √(Pool_Ownership%)'
        '<span style="margin-left:24px;color:#5a8a5a;">'
        '· Chaos mode active when avg wind &gt;15mph OR temp &lt;55°F</span>'
        '</div>'
        + chaos_banner +
        '</div>'
    )
    st.markdown(formula_html, unsafe_allow_html=True)

    # ── BLOCKS B + C — two columns ──────────────────────────────────
    col_b, col_c = st.columns([11, 9], gap="large")

    # ── BLOCK B — Component weight breakdown ────────────────────────
    with col_b:
        COMP_DATA = [
            (
                "FORM",  "#52cc72", int(w["form"]  * 100),
                "SG T2G 50% · Last start 18% · Top-8 recency 12% · Season wins 20%",
                "81% of winners won earlier that season",
            ),
            (
                "FIT",   "#52cc72", int(w["fit"]   * 100),
                "Par-5 scoring 28% · SG ATG 18% · SG App 18+12% · Bogey avoid 16% · Drive 5% · Sunday 3%",
                "4 reachable par-5s = up to 16 stroke separation over 72 holes",
            ),
            (
                "VEGAS", "#5a8a5a", int(w["vegas"] * 100),
                "Implied probability 70% · Model divergence 30%",
                "Primary signal for LIV players with no SG data",
            ),
            (
                "DNA",   "#c8a84a", int(w["dna"]   * 100),
                "Finish history 50% · Best finish 25% · Starts 15% · Last result 10%",
                "76% of top-10s had zero prior Augusta top-10s",
            ),
            (
                "TRAJ",  "#4a6a4a", int(w["traj"]  * 100),
                "60-day OWGR rank movement",
                "Directionally correct · low marginal impact",
            ),
        ]

        comp_rows = ""
        for name, color, pct, sub_txt, insight in COMP_DATA:
            comp_rows += (
                f'<div style="margin-bottom:14px;">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
                f'<span style="font-size:11px;font-weight:700;letter-spacing:.06em;'
                f'text-transform:uppercase;color:{color};width:52px;flex-shrink:0;">{name}</span>'
                f'<div style="flex:1;height:8px;background:#1e301e;border-radius:4px;">'
                f'<div style="width:{pct}%;height:100%;background:{color};border-radius:4px;"></div>'
                f'</div>'
                f'<span style="font-family:DM Mono,monospace;font-size:13px;color:{color};'
                f'width:32px;text-align:right;flex-shrink:0;">{pct}%</span>'
                f'</div>'
                f'<div style="margin-left:60px;font-size:10px;color:#5a8a5a;line-height:1.8;">'
                f'{safe_html(sub_txt)}'
                f'<br><span style="color:#2a422a;font-style:italic;">{safe_html(insight)}</span>'
                f'</div>'
                f'</div>'
            )

        # Modifier pills
        MODIFIERS = [
            "Chalk −5 pts (odds &lt;+600)",
            "Cut rate floor 60%",
            "Injury ×0.80–0.92",
            "Pre-Masters −3 Form",
            "2026 course +3/+2",
        ]
        pills_html = "".join(
            f'<span style="font-size:10px;padding:2px 8px;border-radius:3px;'
            f'background:#1e301e;color:#5a8a5a;border:1px solid #263826;">{m}</span>'
            for m in MODIFIERS
        )

        st.markdown(
            '<div style="background:#0d160d;border:1px solid #1e301e;border-radius:6px;'
            'padding:16px 20px;">'
            '<div style="font-size:11px;font-weight:700;letter-spacing:.18em;'
            'text-transform:uppercase;color:#7aaa7a;margin-bottom:14px;">'
            'COMPONENT WEIGHTS</div>'
            + comp_rows +
            '<div style="margin-top:12px;padding-top:10px;border-top:1px solid #1e301e;">'
            '<div style="font-size:11px;color:#7aaa7a;text-transform:uppercase;'
            'letter-spacing:.12em;margin-bottom:6px;">MODIFIERS APPLIED</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:6px;">{pills_html}</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── BLOCK C — Live model state ───────────────────────────────────
    with col_c:
        n_players = len(df) if df is not None and not df.empty else 0

        def _live_row(rank: int, name: str, val: float, max_val: float,
                      bar_color: str, num_color: str) -> str:
            bar_pct = max(4, min(100, int(val / max(max_val, 1) * 100)))
            return (
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:7px;">'
                f'<span style="font-family:DM Mono,monospace;font-size:11px;'
                f'color:#7aaa7a;width:16px;flex-shrink:0;">{rank}</span>'
                f'<span style="font-size:13px;color:#e8f5e8;flex:1;overflow:hidden;'
                f'text-overflow:ellipsis;white-space:nowrap;">{safe_html(name)}</span>'
                f'<div style="width:80px;height:6px;background:#1e301e;'
                f'border-radius:3px;flex-shrink:0;">'
                f'<div style="width:{bar_pct}%;height:100%;background:{bar_color};'
                f'border-radius:3px;"></div>'
                f'</div>'
                f'<span style="font-family:DM Mono,monospace;font-size:12px;'
                f'color:{num_color};width:32px;text-align:right;flex-shrink:0;">'
                f'{val:.1f}</span>'
                f'</div>'
            )

        score_rows = ""
        ev_rows    = ""
        if df is not None and not df.empty:
            top5_score = df.nlargest(5, "Augusta_Score")[["Player", "Augusta_Score"]]
            max_score  = float(top5_score["Augusta_Score"].max())
            for rank, (_, r) in enumerate(top5_score.iterrows(), 1):
                score_rows += _live_row(rank, r["Player"], float(r["Augusta_Score"]),
                                        max_score, "#52cc72", "#c8a84a")

            if "EV_Score" in df.columns:
                top5_ev  = df.nlargest(5, "EV_Score")[["Player", "EV_Score"]]
                max_ev   = float(top5_ev["EV_Score"].max())
                for rank, (_, r) in enumerate(top5_ev.iterrows(), 1):
                    ev_rows += _live_row(rank, r["Player"], float(r["EV_Score"]),
                                         max_ev, "#c8a84a", "#52cc72")

        st.markdown(
            '<div style="background:#0d160d;border:1px solid #1e301e;border-radius:6px;'
            'padding:16px 20px;">'
            '<div style="font-size:11px;font-weight:700;letter-spacing:.18em;'
            'text-transform:uppercase;color:#7aaa7a;margin-bottom:14px;">'
            f'LIVE MODEL — 2026 FIELD'
            f'<span style="font-size:9px;color:#3aaa5a;margin-left:8px;font-weight:400;">'
            f'· {n_players} players scored</span>'
            '</div>'

            '<div style="font-size:9px;font-weight:700;letter-spacing:.12em;'
            'text-transform:uppercase;color:#5a8a5a;margin-bottom:8px;">'
            'TOP 5 — COMPOSITE SCORE</div>'
            + score_rows +

            '<div style="border-top:1px solid #1e301e;margin:10px 0;"></div>'

            '<div style="font-size:9px;font-weight:700;letter-spacing:.12em;'
            'text-transform:uppercase;color:#5a8a5a;margin-bottom:8px;">'
            'TOP 5 — EV SCORE <span style="color:#5a8a5a;font-weight:400;">'
            '(pool edge)</span></div>'
            + ev_rows +

            '<div style="margin-top:8px;font-size:10px;color:#2a422a;font-style:italic;">'
            'EV = Augusta Score ÷ √(estimated pool ownership%)</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── BLOCK D — EV vs Odds scatter (full width below B+C) ─────────
    if df is None or df.empty:
        return

    import plotly.graph_objects as go

    # Get team membership from session state
    _ct = st.session_state.get("custom_teams", {})
    if not _ct:
        _cp = st.session_state.get("confirmed_picks", {})
        _ct = _cp.get("teams", {})
    team_membership = get_team_membership(_ct)

    # --- data prep ---
    def _odds_to_implied(odds):
        try:
            odds = float(odds)
        except (TypeError, ValueError):
            return 0.01
        if odds == 0:
            return 0.01
        if odds > 0:
            return 100.0 / (odds + 100.0)
        else:
            return abs(odds) / (abs(odds) + 100.0)

    df_plot = df.copy()
    df_plot["Implied_Prob"] = df_plot["Odds_American"].apply(_odds_to_implied)
    max_prob = df_plot["Implied_Prob"].max()
    if max_prob > 0:
        df_plot["Market_Score"] = df_plot["Implied_Prob"] / max_prob * 100.0
    else:
        df_plot["Market_Score"] = 50.0
    df_plot["value_gap"] = df_plot["Augusta_Score"] - df_plot["Market_Score"]

    # --- sub-header ---
    st.markdown(
        '<div style="font-size:11px;font-weight:700;letter-spacing:.18em;'
        'text-transform:uppercase;color:#7aaa7a;margin:16px 0 6px;">'
        'MODEL vs MARKET — VALUE IDENTIFICATION</div>'
        '<div style="font-size:11px;color:#5a8a5a;margin-bottom:10px;font-style:italic;">'
        'Players above the line are rated higher by the model than the market implies'
        ' · Below = model rates lower than odds suggest</div>',
        unsafe_allow_html=True,
    )

    # --- build figure ---
    fig = go.Figure()

    # Shaded background regions (below traces)
    fig.add_shape(
        type="rect", x0=0, x1=100, y0=0, y1=100,
        fillcolor="rgba(58,170,90,0.04)", line_width=0, layer="below",
    )

    # Fair-value diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100],
        mode="lines",
        line=dict(color="#1e301e", width=1.5, dash="dash"),
        name="Fair Value",
        hoverinfo="skip",
    ))

    # Zone annotations
    fig.add_annotation(
        x=15, y=82,
        text="MODEL RATES HIGHER<br>than market implies",
        font=dict(size=9, color="#3AAA5A"),
        showarrow=False, align="left",
    )
    fig.add_annotation(
        x=75, y=18,
        text="MARKET RATES HIGHER<br>than model predicts",
        font=dict(size=9, color="#CC4A4A"),
        showarrow=False, align="right",
    )
    fig.add_annotation(
        x=55, y=50,
        text="FAIR VALUE",
        font=dict(size=8, color="#2A3A2A"),
        showarrow=False, textangle=-42,
    )

    # Field players (dim, behind team players)
    _own_safe   = "Ownership_Pct" if "Ownership_Pct" in df_plot.columns else "Augusta_Score"
    _ev_safe    = "EV_Score"      if "EV_Score"      in df_plot.columns else "Augusta_Score"
    field_df = df_plot[~df_plot["Player"].isin(team_membership.keys())]
    if not field_df.empty:
        fig.add_trace(go.Scatter(
            x=field_df["Market_Score"],
            y=field_df["Augusta_Score"],
            mode="markers",
            marker=dict(color="#2A3A2A", size=7,
                        line=dict(color="#3A5A3A", width=0.5)),
            name="Field",
            text=field_df["Player"],
            customdata=field_df[["Augusta_Score", _ev_safe,
                                  "Odds_American", _own_safe]].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Augusta Score: %{customdata[0]:.1f}<br>"
                "EV Score: %{customdata[1]:.1f}<br>"
                "Odds: %{customdata[2]}<br>"
                "Est. ownership: %{customdata[3]:.1f}%"
                "<extra></extra>"
            ),
        ))

    # My team players (on top, labeled by last name)
    _team_colors = {
        "team_a": "#3AAA5A",
        "team_b": "#52CC72",
        "team_c": "#C8A84A",
    }
    _team_labels = {
        "team_a": "Team A — Floor",
        "team_b": "Team B — Ceiling",
        "team_c": "Team C — Value",
    }
    for tk in ("team_a", "team_b", "team_c"):
        _members = [
            p.get("Player", "")
            for p in _ct.get(tk, {}).get("stats", [])
        ]
        team_df = df_plot[df_plot["Player"].isin(_members)]
        if team_df.empty:
            continue
        color = _team_colors[tk]
        fig.add_trace(go.Scatter(
            x=team_df["Market_Score"],
            y=team_df["Augusta_Score"],
            mode="markers+text",
            marker=dict(color=color, size=12,
                        line=dict(color="#060D06", width=1.5)),
            text=team_df["Player"].apply(lambda n: n.split()[-1]),
            textposition="top center",
            textfont=dict(size=9, color=color),
            name=_team_labels[tk],
            customdata=team_df[["Augusta_Score", _ev_safe,
                                 "Odds_American", _own_safe]].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Augusta Score: %{customdata[0]:.1f}<br>"
                "EV Score: %{customdata[1]:.1f}<br>"
                "Odds: %{customdata[2]}<br>"
                "Est. ownership: %{customdata[3]:.1f}%"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        paper_bgcolor="#060D06",
        plot_bgcolor="#0D160D",
        font=dict(color="#96CC96", family="DM Sans"),
        height=480,
        margin=dict(l=50, r=30, t=30, b=50),
        showlegend=True,
        legend=dict(
            bgcolor="#0D160D", bordercolor="#1E301E", borderwidth=1,
            font=dict(size=10, color="#96CC96"),
            x=0.01, y=0.99, xanchor="left", yanchor="top",
        ),
        xaxis=dict(
            title=dict(text="Market Implied Score (odds-derived)",
                       font=dict(size=11, color="#4A6A4A")),
            gridcolor="#1E301E", zerolinecolor="#1E301E",
            range=[-2, 105],
            tickfont=dict(color="#4A6A4A", size=10),
        ),
        yaxis=dict(
            title=dict(text="Augusta Score (model composite)",
                       font=dict(size=11, color="#4A6A4A")),
            gridcolor="#1E301E", zerolinecolor="#1E301E",
            range=[-2, 105],
            tickfont=dict(color="#4A6A4A", size=10),
        ),
        hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # --- insight callout: top value vs top chalk ---
    try:
        top_value = df_plot.nlargest(3, "value_gap")
        top_chalk = df_plot.nsmallest(2, "value_gap")

        value_pills = "".join(
            f'<span style="background:#1a2a1a;border:1px solid #3aaa5a55;'
            f'border-radius:3px;padding:3px 10px;font-size:11px;color:#52cc72;">'
            f'{safe_html(str(r["Player"]).split()[-1])} '
            f'+{float(r["value_gap"]):.1f}</span>'
            for _, r in top_value.iterrows()
        )
        chalk_pills = "".join(
            f'<span style="background:#1a1400;border:1px solid #c8a84a55;'
            f'border-radius:3px;padding:3px 10px;font-size:11px;color:#c8a84a;">'
            f'{safe_html(str(r["Player"]).split()[-1])} '
            f'{float(r["value_gap"]):.1f}</span>'
            for _, r in top_chalk.iterrows()
        )
        st.markdown(
            f'<div style="display:flex;gap:24px;margin-top:8px;flex-wrap:wrap;">'
            f'<div>'
            f'<div style="font-size:9px;color:#3aaa5a;font-weight:700;letter-spacing:.1em;'
            f'text-transform:uppercase;margin-bottom:6px;">&#8593; MODEL RATES ABOVE MARKET</div>'
            f'<div style="display:flex;gap:6px;flex-wrap:wrap;">{value_pills}</div>'
            f'</div>'
            f'<div>'
            f'<div style="font-size:9px;color:#c8a84a;font-weight:700;letter-spacing:.1em;'
            f'text-transform:uppercase;margin-bottom:6px;">&#8595; MARKET RATES ABOVE MODEL</div>'
            f'<div style="display:flex;gap:6px;flex-wrap:wrap;">{chalk_pills}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass  # insight callout is non-critical


def _render_model_validation(df: pd.DataFrame, data: dict):
    """Backtest bar chart + year table + condition charts + insight cards."""
    import plotly.graph_objects as go
    from backtest_data import BACKTEST_RESULTS, backtest_summary

    summary = backtest_summary()
    n       = summary["total_years"]

    st.markdown(
        f'<div class="mdl-metrics">'
        f'<div class="mdl-metric">'
        f'<div class="mdl-metric-val">8/{n}</div>'
        f'<div class="mdl-metric-lbl">Winner in Top 3</div>'
        f'</div>'
        f'<div class="mdl-metric">'
        f'<div class="mdl-metric-val">13/{n}</div>'
        f'<div class="mdl-metric-lbl">Winner in Top 5</div>'
        f'</div>'
        f'<div class="mdl-metric warn">'
        f'<div class="mdl-metric-val warn">17/{n}</div>'
        f'<div class="mdl-metric-lbl">Winner in Top 10</div>'
        f'</div>'
        f'<div class="mdl-metric">'
        f'<div class="mdl-metric-val bad">3/{n}</div>'
        f'<div class="mdl-metric-lbl">Complete Misses (&gt;10)</div>'
        f'</div>'
        f'</div>'
        f'<div style="font-size:9px;color:#2a422a;margin-top:6px;margin-bottom:16px;">'
        f'Simulated backtest 2005–2025 &nbsp;&middot;&nbsp; Model performs best in mild conditions'
        f' &nbsp;&middot;&nbsp; Struggles in chaos/windy years (2007, 2008, 2016)'
        f'</div>',
        unsafe_allow_html=True,
    )

    val_l, val_r = st.columns([3, 2], gap="large")

    with val_l:
        st.markdown(
            '<div style="font-size:8px;letter-spacing:.10em;text-transform:uppercase;'
            'color:var(--t4);margin-bottom:8px;">MODEL RANK FOR ACTUAL WINNER — BY YEAR</div>',
            unsafe_allow_html=True,
        )
        years       = [r["year"]       for r in BACKTEST_RESULTS]
        mod_ranks   = [r["model_rank"] for r in BACKTEST_RESULTS]
        winners     = [r["winner"]     for r in BACKTEST_RESULTS]
        chaos_flags = [r["chaos"]      for r in BACKTEST_RESULTS]
        bar_colors  = [
            "#1e2e1e" if ch else ("#3aaa5a" if mr <= 5 else ("#c8a84a" if mr <= 10 else "#cc4a4a"))
            for mr, ch in zip(mod_ranks, chaos_flags)
        ]
        hover_texts = [
            f"{w}<br>Model rank: #{mr}<br>{'CHAOS YEAR' if ch else ''}"
            for w, mr, ch in zip(winners, mod_ranks, chaos_flags)
        ]
        fig_bt = go.Figure(go.Bar(
            x=years, y=mod_ranks, marker_color=bar_colors,
            marker_line=dict(width=0),
            text=[str(mr) for mr in mod_ranks], textposition="outside",
            textfont=dict(family="DM Mono", size=9, color="#426842"),
            hovertext=hover_texts, hoverinfo="text", name="",
        ))
        for thresh, lbl, clr in [(5, "Top 5", "#3aaa5a"), (10, "Top 10", "#c8a84a")]:
            fig_bt.add_hline(
                y=thresh, line_dash="dot", line_color=clr, line_width=1,
                annotation_text=lbl,
                annotation_font=dict(size=8, color=clr),
                annotation_position="right",
            )
        fig_bt.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#426842"),
            margin=dict(l=0, r=40, t=10, b=30), height=280,
            xaxis=dict(
                tickfont=dict(family="DM Mono", size=9, color="#426842"),
                gridcolor="#152015", showgrid=False,
            ),
            yaxis=dict(
                autorange="reversed", title=None,
                tickfont=dict(family="DM Mono", size=9, color="#426842"),
                gridcolor="#152015",
            ),
            showlegend=False, bargap=0.25,
        )
        st.plotly_chart(fig_bt, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            '<div style="font-size:8px;letter-spacing:.10em;text-transform:uppercase;'
            'color:var(--t4);margin-bottom:6px;margin-top:10px;">YEAR-BY-YEAR DETAIL</div>',
            unsafe_allow_html=True,
        )
        tbl_rows = ""
        for r in BACKTEST_RESULTS:
            mr = r["model_rank"]
            mc = ("var(--green2)" if mr <= 3 else
                  ("var(--green)"  if mr <= 5 else
                   ("var(--gold)"  if mr <= 10 else "var(--red)")))
            hit_badges = ""
            if r["top3_hit"]:
                hit_badges += '<span style="font-size:7px;color:var(--green2);margin-right:4px;">TOP3</span>'
            elif r["top5_hit"]:
                hit_badges += '<span style="font-size:7px;color:var(--green);margin-right:4px;">TOP5</span>'
            elif r["top10_hit"]:
                hit_badges += '<span style="font-size:7px;color:var(--gold);margin-right:4px;">TOP10</span>'
            chaos_tag = '<span style="font-size:7px;color:var(--t4);margin-left:3px;">CHAOS</span>' if r["chaos"] else ""
            tbl_rows += (
                f'<div style="display:grid;grid-template-columns:44px 1fr 32px 80px;'
                f'gap:0;padding:5px 8px;border-bottom:1px solid var(--s1);align-items:center;">'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:10px;color:var(--t4);">{r["year"]}</div>'
                f'<div style="font-size:11px;color:var(--t1);">{safe_html(r["winner"])}</div>'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:11px;color:{mc};text-align:center;">#{mr}</div>'
                f'<div>{hit_badges}{chaos_tag}</div>'
                f'</div>'
            )
        st.markdown(
            '<div style="border:1px solid var(--b1);">'
            '<div style="display:grid;grid-template-columns:44px 1fr 32px 80px;'
            'gap:0;padding:5px 8px;background:var(--s1);border-bottom:1px solid var(--b1);">'
            '<div style="font-size:8px;color:var(--t4);">YEAR</div>'
            '<div style="font-size:8px;color:var(--t4);">WINNER</div>'
            '<div style="font-size:8px;color:var(--t4);">RANK</div>'
            '<div style="font-size:8px;color:var(--t4);">HIT</div>'
            '</div>' + tbl_rows + '</div>',
            unsafe_allow_html=True,
        )

    with val_r:
        st.markdown(
            '<div style="font-size:8px;letter-spacing:.10em;text-transform:uppercase;'
            'color:var(--t4);margin-bottom:8px;">TOP-10 HIT RATE BY CONDITION</div>',
            unsafe_allow_html=True,
        )
        cond_rates = summary.get("condition_top10_rates", {})
        COND_LABELS = {
            "mild":       "Mild",
            "fast_firm":  "Fast / Firm",
            "soft_wet":   "Soft / Wet",
            "cold_windy": "Cold / Windy",
        }
        cond_names = [COND_LABELS.get(k, k) for k in cond_rates]
        cond_vals  = [float(v) for v in cond_rates.values()]
        cond_clrs  = [
            "#3aaa5a" if v >= 60 else ("#c8a84a" if v >= 40 else "#cc4a4a")
            for v in cond_vals
        ]
        chaos_hit  = float(summary.get("chaos_top10_rate",  0))
        normal_hit = float(summary.get("normal_top10_rate", 0))

        fig_cond = go.Figure(go.Bar(
            x=cond_vals, y=cond_names, orientation="h",
            marker_color=cond_clrs, marker_line=dict(width=0),
            text=[f"{v:.0f}%" for v in cond_vals], textposition="auto",
            textfont=dict(family="DM Mono", size=9, color="#000000"),
            hovertemplate="%{y}: %{x:.0f}%<extra></extra>",
        ))
        fig_cond.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#426842"),
            margin=dict(l=0, r=0, t=5, b=20), height=180,
            xaxis=dict(
                range=[0, 110],
                tickfont=dict(family="DM Mono", size=9, color="#426842"),
                gridcolor="#152015", showgrid=True, ticksuffix="%",
            ),
            yaxis=dict(tickfont=dict(size=10, color="#7aaa7a")),
            showlegend=False, bargap=0.3,
        )
        st.plotly_chart(fig_cond, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            f'<div style="display:flex;gap:10px;margin-bottom:16px;">'
            f'<div style="flex:1;border:1px solid var(--b1);background:var(--s1);padding:10px 12px;">'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:18px;color:var(--green2);">{normal_hit:.0f}%</div>'
            f'<div style="font-size:8px;letter-spacing:.07em;text-transform:uppercase;color:var(--t4);">Normal Conditions</div>'
            f'</div>'
            f'<div style="flex:1;border:1px solid var(--b1);background:var(--s1);padding:10px 12px;">'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:18px;color:var(--gold);">{chaos_hit:.0f}%</div>'
            f'<div style="font-size:8px;letter-spacing:.07em;text-transform:uppercase;color:var(--t4);">Chaos Conditions</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        chaos_years_count = sum(1 for r in BACKTEST_RESULTS if r["chaos"])
        st.markdown(
            f'<div class="ins-card warn">'
            f'<div class="ins-card-hdr">CHAOS CONDITIONS</div>'
            f'<div class="ins-card-body">'
            f'In {chaos_years_count} chaos years (wind &gt;15 mph / temp &lt;55°F) the model\'s top-10 hit rate '
            f'drops to <b style="color:var(--gold);">{chaos_hit:.0f}%</b> vs '
            f'<b style="color:var(--green2);">{normal_hit:.0f}%</b> in normal conditions. '
            f'The Chaos Coefficient auto-rebalances weights when detected.'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        chalk_count = int(df["Chalk_Penalty"].sum()) if "Chalk_Penalty" in df.columns else 0
        edge_count  = int((df["EV_Score"] >= 25).sum()) if "EV_Score" in df.columns else 0
        st.markdown(
            f'<div class="ins-card good">'
            f'<div class="ins-card-hdr">MARKET EDGE</div>'
            f'<div class="ins-card-body">'
            f'<b style="color:var(--green2);">{edge_count} players</b> have EV Score ≥25 — '
            f'model rates them above their implied market probability. '
            f'{chalk_count} player{"s" if chalk_count != 1 else ""} carry a chalk penalty for high public ownership.'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        best  = summary.get("best_call",    {})
        worst = summary.get("biggest_miss", {})
        st.markdown(
            f'<div class="ins-card alert">'
            f'<div class="ins-card-hdr">MODEL RANGE</div>'
            f'<div class="ins-card-body">'
            f'Best call: <b style="color:var(--green2);">{safe_html(best.get("winner","–"))}</b> '
            f'{best.get("year","–")} ranked #{best.get("model_rank","–")}.<br>'
            f'Biggest miss: <b style="color:var(--red);">{safe_html(worst.get("winner","–"))}</b> '
            f'{worst.get("year","–")} ranked #{worst.get("model_rank","–")} — '
            f'{safe_html(worst.get("notes","")[:80])}.'
            f'</div></div>',
            unsafe_allow_html=True,
        )


def render_my_picks(df: pd.DataFrame, teams: dict, data: dict):
    """Tab 1 — MY PICKS: team cards, swap controls, tiebreaker, confirm."""
    tab_my_picks(df, teams, data)


def render_live(df: pd.DataFrame, data: dict):
    """Tab 2 — LIVE: pool standings + tournament leaderboard side-by-side."""
    live_l, live_r = st.columns([1, 1], gap="medium")
    with live_l:
        tab_pool_standings(data)
    with live_r:
        tab_leaderboard(data)


def render_model(df: pd.DataFrame, data: dict, field_report: dict | None = None):
    """Tab 3 — MODEL: scoring architecture (top) + my players + full field + validation toggle."""

    # ── Field coverage header ──────────────────────────────────────
    if field_report:
        _cov    = field_report.get("field_coverage_pct", 0)
        _total  = field_report.get("total_scored", 0)
        _miss   = field_report.get("missing_key_players", [])
        _miss_str = (
            f" · <span style='color:#c0a020;'>&#9888; {len(_miss)} key players missing</span>"
            if _miss else ""
        )
        st.markdown(
            f'<div style="font-size:10px;color:#5a8a5a;margin-bottom:8px;">'
            f'Field coverage: <b>{_cov}%</b> of confirmed 91-player field scored'
            f' · {_total} players in model'
            f'{_miss_str}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── A: Scoring Architecture — always visible, top ──────────────
    _render_scoring_architecture_v2(df, data)

    st.markdown('<hr style="border-color:var(--b1);margin:20px 0 16px;">', unsafe_allow_html=True)

    # ── B: My Players Quick View ───────────────────────────────────
    _render_my_players_pinned(df)

    # ── C: Full Field / Player Intelligence ───────────────────────
    tab_rankings(df, data)

    # ── D: Model Validation (toggle) ──────────────────────────────
    st.markdown('<hr style="border-color:var(--b1);margin:20px 0 10px;">', unsafe_allow_html=True)
    show_val = st.toggle("MODEL VALIDATION — 2006–2025", value=False, key="mdl_val_toggle")
    if show_val:
        st.markdown('<div style="margin-top:12px;"></div>', unsafe_allow_html=True)
        _render_model_validation(df, data)


# ─────────────────────────────────────────────────────────────────
# DATA TAB — CONSTANTS
# ─────────────────────────────────────────────────────────────────

CHART_THEME = {
    "paper_bgcolor":    "#060D06",
    "plot_bgcolor":     "#0D160D",
    "font_color":       "#96CC96",
    "gridcolor":        "#1E301E",
    "title_font_color": "#C8A84A",
}

MY_TEAM_COLORS = {
    "team_a": "#3AAA5A",
    "team_b": "#52CC72",
    "team_c": "#C8A84A",
    "field":  "#2A3A2A",
}

# Real pool tiebreaker distributions from Ferraro Green Jacket Pool
TIEBREAKER_2019 = [
    -15,-15,-15,-34,-27,-41,-18,-31,-24,-28,-72,-48,-26,-28,-36,-34,
    -34,-32,-41,-31,-37,-28,-28,-28,-42,-31,-24,-24,-30,-28,-23,-20,
    -24,-30,-28,-25,-27,-29,-32,-32,-32,-23,-18,-24,-50,-50,-40,-33,
    -43,-37,-37,-29,-27,-18,-32,-21,-18,-14,-30,-26,-32,-33,-28,-35,
    -34,-28,-30,-30,-28,-33,-36,-37,-45,-32,-34,-28,-31,-32,-24,-26,
    -30,-38,-26,-22,-32,-40,-35,-27,-41,-43,-38,-48,-52,-50,-40,-38,
    -42,-42,-38,-27,-33,-55,-55,-27,-21,-24,-36,-34,-7,-34,-30,-38,
    -43,-38,-8,-24,-48,-35,-42,-28,-30,-36,-40,-49,-24,-38,-26,-33,
    -26,-33,-38,-40,-32,-26,-14,-43,-26,-33,-9,-12,-40,-14,
]

TIEBREAKER_2023 = [
    -19,-21,-22,-29,-32,-26,-20,-10,-19,-28,-27,-29,-32,-17,-23,-22,
    -18,-12,-23,-30,-24,-32,-17,-25,-35,-33,-22,-16,-32,-4,-45,-45,
    -45,-45,-16,-10,-32,-32,-33,-8,-5,-4,-45,-45,-45,-45,-14,-28,
    -28,-28,-25,-25,-26,-13,-23,-30,-39,-22,-31,-22,-21,-15,-25,-23,
    -16,-22,-36,-29,-30,-21,-13,-15,-25,-34,-38,-38,-38,-17,-23,-25,
    -43,-47,-58,-11,-8,-43,-52,-41,-47,-27,-17,-28,-15,-12,-25,-12,
    -28,-24,-36,-36,-36,-23,-32,-22,-51,-23,-27,-18,-18,-18,-20,-20,
    -20,-37,-33,-35,-16,-14,-29,-30,-40,-42,-24,-18,-25,-22,-23,-14,
    -8,-15,-24,-18,-22,-19,-6,-11,-16,-13,-30,-22,-25,-13,-13,-13,
    -8,-29,-29,-29,-25,-26,-30,-30,-23,-12,-12,-12,-31,-31,-40,-40,
    -40,-16,-8,-9,-10,-28,-28,-28,-39,-49,-24,-38,-38,-38,-27,-24,
    -27,-24,-24,-24,-32,-32,-50,-49,-55,-24,-24,-24,-20,-20,-20,-25,
    -25,-25,-27,-27,-26,-26,-26,-22,-22,-23,-28,-28,-30,-17,-17,-26,
    -22,-15,-23,-14,-12,-11,-15,-15,-15,-32,-32,-32,-23,-22,-26,-23,
    -27,-27,-27,-27,-27,-27,-30,-25,-35,-24,-17,-24,-11,-11,-26,-24,
    -25,-28,-39,-38,-40,-25,-25,-25,-18,-27,-16,-21,-16,-22,-19,-21,
    -12,-12,-12,
]


def get_team_membership(ct: dict) -> dict:
    """Return {player_name: team_key} for all 12 team players."""
    membership: dict[str, str] = {}
    for tk in ["team_a", "team_b", "team_c"]:
        for p in ct.get(tk, {}).get("stats", []):
            name = p.get("Player", "")
            if name and name not in membership:
                membership[name] = tk
    return membership


def _apply_chart_theme(fig, title: str = "", height: int = 400) -> None:
    """Apply the dark green Augusta theme to a plotly figure in-place."""
    title_cfg = dict(text=title, font=dict(color="#C8A84A", size=14)) if title else {}
    fig.update_layout(
        paper_bgcolor="#060D06",
        plot_bgcolor="#0D160D",
        font=dict(color="#96CC96", family="DM Sans, sans-serif", size=11),
        title=title_cfg,
        legend=dict(
            bgcolor="#0D160D",
            bordercolor="#1E301E",
            borderwidth=1,
            font=dict(color="#96CC96", size=10),
        ),
        xaxis=dict(gridcolor="#1E301E", zerolinecolor="#1E301E"),
        yaxis=dict(gridcolor="#1E301E", zerolinecolor="#1E301E"),
        margin=dict(l=10, r=20, t=50 if title else 20, b=10),
        height=height,
    )


def _section_header(title: str, subtitle: str = "") -> str:
    sub = f'<span style="font-weight:400;color:#2A422A;font-size:9px;letter-spacing:.06em;">&nbsp;—&nbsp;{subtitle}</span>' if subtitle else ""
    return (
        f'<div style="font-size:10px;font-weight:700;letter-spacing:.16em;'
        f'text-transform:uppercase;color:#4A6A4A;'
        f'border-bottom:1px solid #1E301E;'
        f'padding-bottom:8px;margin:24px 0 16px;">'
        f'{title}{sub}</div>'
    )


# ─────────────────────────────────────────────────────────────────
# DATA TAB — RENDER FUNCTION
# ─────────────────────────────────────────────────────────────────

def render_data_tab(df: pd.DataFrame, ct: dict, data: dict) -> None:
    """Tab 4 — DATA: full field analytics, model components, pool intelligence."""
    import plotly.graph_objects as go
    import plotly.express as px

    membership = get_team_membership(ct)
    my_players = list(membership.keys())

    def _pcolor(name: str) -> str:
        return MY_TEAM_COLORS.get(membership.get(name, "field"), "#2A3A2A")

    # Safe column getter with default
    def _col(row_or_df, col, default=50.0):
        try:
            return float(row_or_df[col])
        except Exception:
            return default

    COMPONENTS = ["Form_Score", "Fit_Score", "DNA_Score", "Vegas_Score", "Trajectory_Score"]
    COMP_LABELS = ["Form", "Fit", "DNA", "Vegas", "Traj"]
    COMP_COLORS = ["#3AAA5A", "#52CC72", "#C8A84A", "#5A8A5A", "#2A5A2A"]

    # Ensure all component columns exist with safe defaults
    for col in COMPONENTS:
        if col not in df.columns:
            df[col] = 50.0

    # ─── SECTION 1: FIELD OVERVIEW ────────────────────────────────
    st.markdown(_section_header("FIELD OVERVIEW", "How the full field stacks up"), unsafe_allow_html=True)

    # Chart 1A — Augusta Score Distribution (top 40 horizontal bar)
    try:
        top40 = df.nlargest(40, "Augusta_Score").sort_values("Augusta_Score", ascending=True)
        bar_colors = [_pcolor(n) for n in top40["Player"]]
        p50_score = float(df["Augusta_Score"].quantile(0.50))

        custom_data = np.column_stack([
            top40["Form_Score"].fillna(50),
            top40["Fit_Score"].fillna(50),
            top40["DNA_Score"].fillna(50),
            top40["Vegas_Score"].fillna(50),
            top40["Trajectory_Score"].fillna(50),
            top40.get("Ownership_Pct", pd.Series(5.0, index=top40.index)).fillna(5),
        ])

        fig1a = go.Figure(go.Bar(
            x=top40["Augusta_Score"],
            y=top40["Player"],
            orientation="h",
            marker_color=bar_colors,
            customdata=custom_data,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Augusta Score: %{x:.1f}<br>"
                "Form: %{customdata[0]:.1f}  |  Fit: %{customdata[1]:.1f}  |  "
                "DNA: %{customdata[2]:.1f}<br>"
                "Vegas: %{customdata[3]:.1f}  |  Traj: %{customdata[4]:.1f}<br>"
                "Pool Own: %{customdata[5]:.1f}%<extra></extra>"
            ),
        ))
        fig1a.add_vline(
            x=p50_score, line_dash="dot", line_color="#4A6A4A", line_width=1,
            annotation_text="50th pct", annotation_font_color="#4A6A4A",
            annotation_font_size=9,
        )
        _apply_chart_theme(fig1a, "Augusta Score — Top 40 Players", height=900)
        fig1a.update_layout(showlegend=False, xaxis_title="Augusta Score", yaxis_title="")
        st.plotly_chart(fig1a, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.warning(f"Field distribution chart not available: {e}")

    # Chart 1B — Ownership vs Augusta Score scatter
    try:
        own_col = "Ownership_Pct" if "Ownership_Pct" in df.columns else None
        if own_col is None:
            raise ValueError("Ownership_Pct not in df")

        med_own   = float(df[own_col].median())
        med_score = float(df["Augusta_Score"].median())

        scatter_colors = [_pcolor(n) for n in df["Player"]]
        ev_sizes = df["EV_Score"].fillna(5).clip(1, 50)
        # Normalize sizes: 6 to 20 px
        ev_norm = 6 + (ev_sizes - ev_sizes.min()) / (ev_sizes.max() - ev_sizes.min() + 1e-9) * 14

        # Labels only for my 12 players
        labels = [n if n in membership else "" for n in df["Player"]]

        odds_implied = df["Odds_American"].apply(
            lambda o: round(100 / (abs(o) + 100) * 100, 1) if o < 0
            else round(100 / (o + 100) * 100, 1)
        )

        fig1b = go.Figure()
        fig1b.add_trace(go.Scatter(
            x=df[own_col],
            y=df["Augusta_Score"],
            mode="markers+text",
            text=labels,
            textposition="top center",
            textfont=dict(size=9, color="#96CC96"),
            marker=dict(
                color=scatter_colors,
                size=ev_norm,
                line=dict(width=0),
            ),
            customdata=np.column_stack([
                df["Player"],
                df[own_col].fillna(0),
                df["EV_Score"].fillna(0),
                df["Odds_American"].fillna(0),
                odds_implied,
            ]),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Augusta Score: %{y:.1f}<br>"
                "Pool Ownership: %{customdata[1]:.1f}%<br>"
                "EV Score: %{customdata[2]:.1f}<br>"
                "Odds: +%{customdata[3]:.0f}  (impl %{customdata[4]:.1f}%)<extra></extra>"
            ),
        ))
        # Quadrant lines
        fig1b.add_vline(x=med_own,   line_dash="dot", line_color="#2A422A", line_width=1)
        fig1b.add_hline(y=med_score, line_dash="dot", line_color="#2A422A", line_width=1)
        # Quadrant labels
        x_max = float(df[own_col].max()) * 1.05
        y_max = float(df["Augusta_Score"].max()) * 1.02
        y_min = float(df["Augusta_Score"].min()) * 0.98
        for txt, xpos, ypos, clr in [
            ("HIGH VALUE",  med_own * 0.3,  y_max * 0.98, "#C8A84A"),
            ("CHALK",       med_own * 1.5,  y_max * 0.98, "#4A6A4A"),
            ("AVOID",       med_own * 0.3,  y_min * 1.02, "#2A3A2A"),
            ("OVEROWNED",   med_own * 1.5,  y_min * 1.02, "#4A4A2A"),
        ]:
            fig1b.add_annotation(
                x=xpos, y=ypos, text=txt,
                font=dict(color=clr, size=9, family="DM Sans"),
                showarrow=False, xanchor="center",
            )
        _apply_chart_theme(fig1b, "Ownership vs Augusta Score — Pool Edge Map", height=550)
        fig1b.update_layout(
            xaxis_title="Estimated Pool Ownership %",
            yaxis_title="Augusta Score",
            showlegend=False,
        )
        st.plotly_chart(fig1b, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.warning(f"Ownership scatter not available: {e}")

    # ─── SECTION 2: MODEL COMPONENTS ──────────────────────────────
    st.markdown(_section_header("MODEL COMPONENTS", "What drives each player's score"), unsafe_allow_html=True)

    # Chart 2A — Radar charts for each team (3 columns)
    try:
        radar_cols = st.columns(3, gap="small")
        team_cfg = [
            ("team_a", "Team A — Floor",   "#3AAA5A"),
            ("team_b", "Team B — Ceiling", "#52CC72"),
            ("team_c", "Team C — Value",   "#C8A84A"),
        ]
        line_styles = ["solid", "dash", "dot", "dashdot"]

        for col, (tk, tlabel, tclr) in zip(radar_cols, team_cfg):
            with col:
                fig_r = go.Figure()
                stats = ct.get(tk, {}).get("stats", [])
                for i, pstat in enumerate(stats):
                    pname = pstat.get("Player", f"P{i+1}")
                    prow = df[df["Player"] == pname]
                    if prow.empty:
                        vals = [50.0] * 5
                    else:
                        r = prow.iloc[0]
                        vals = [_col(r, c) for c in COMPONENTS]
                    # Close the polygon
                    theta = COMP_LABELS + [COMP_LABELS[0]]
                    r_vals = vals + [vals[0]]
                    # Convert team color to rgba for fill (8-digit hex not supported by Plotly)
                    _rgba_map = {
                        "#3AAA5A": "rgba(58, 170, 90, 0.094)",
                        "#52CC72": "rgba(82, 204, 114, 0.094)",
                        "#C8A84A": "rgba(200, 168, 74, 0.094)",
                    }
                    fill_rgba = _rgba_map.get(tclr, "rgba(58, 170, 90, 0.094)")
                    fig_r.add_trace(go.Scatterpolar(
                        r=r_vals,
                        theta=theta,
                        mode="lines",
                        name=pname.split()[-1],
                        line=dict(color=tclr, dash=line_styles[i % 4], width=2),
                        fill="toself",
                        fillcolor=fill_rgba,
                    ))
                fig_r.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True, range=[0, 100],
                            gridcolor="#1E301E", linecolor="#1E301E",
                            tickfont=dict(color="#4A6A4A", size=8),
                        ),
                        angularaxis=dict(
                            tickfont=dict(color="#96CC96", size=9),
                            linecolor="#1E301E", gridcolor="#1E301E",
                        ),
                        bgcolor="#0D160D",
                    ),
                    paper_bgcolor="#060D06",
                    font=dict(color="#96CC96", family="DM Sans"),
                    title=dict(text=tlabel, font=dict(color="#C8A84A", size=12)),
                    legend=dict(bgcolor="#0D160D", bordercolor="#1E301E", borderwidth=1,
                                font=dict(size=9, color="#96CC96")),
                    margin=dict(l=20, r=20, t=50, b=20),
                    height=380,
                )
                st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.warning(f"Radar charts not available: {e}")

    # Chart 2B — Stacked component bar chart (top 20)
    try:
        top20 = df.nlargest(20, "Augusta_Score").sort_values("Augusta_Score", ascending=True)
        fig2b = go.Figure()
        for comp, label, clr in zip(COMPONENTS, COMP_LABELS, COMP_COLORS):
            fig2b.add_trace(go.Bar(
                name=label,
                x=top20[comp].fillna(0),
                y=top20["Player"],
                orientation="h",
                marker_color=clr,
                hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.1f}}<extra></extra>",
            ))
        # Highlight my team players with an outline overlay
        my_top20 = [n for n in top20["Player"] if n in membership]
        if my_top20:
            y_positions = {n: i for i, n in enumerate(top20["Player"].tolist())}
            for pname in my_top20:
                tclr = _pcolor(pname)
                total = sum(
                    float(top20.loc[top20["Player"] == pname, c].values[0])
                    for c in COMPONENTS if c in top20.columns
                )
                ypos = y_positions.get(pname, 0)
                fig2b.add_shape(
                    type="rect",
                    x0=0, x1=total,
                    y0=ypos - 0.45, y1=ypos + 0.45,
                    line=dict(color=tclr, width=2),
                    fillcolor="rgba(0,0,0,0)",
                    layer="above",
                )
        fig2b.update_layout(barmode="stack")
        _apply_chart_theme(fig2b, "Component Breakdown — Top 20 Players", height=600)
        fig2b.update_layout(xaxis_title="Score (sum of weighted components)", yaxis_title="")
        st.plotly_chart(fig2b, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.warning(f"Component breakdown chart not available: {e}")

    # ─── SECTION 3: POOL INTELLIGENCE ─────────────────────────────
    st.markdown(_section_header("POOL INTELLIGENCE", "Where the edge is in 2026"), unsafe_allow_html=True)

    pi_col1, pi_col2 = st.columns([1, 2], gap="medium")

    # Chart 3A — Ownership band distribution (donut)
    with pi_col1:
        try:
            own_col = "Ownership_Pct" if "Ownership_Pct" in df.columns else None
            if own_col is None:
                raise ValueError("No ownership data")

            def _band(pct):
                if pct > 35:   return "CHALK >35%"
                if pct >= 10:  return "MID 10-35%"
                if pct >= 3:   return "VALUE 3-12%"
                return "FIELD <3%"

            df_bands = df[own_col].apply(_band).value_counts()
            band_order = ["CHALK >35%", "MID 10-35%", "VALUE 3-12%", "FIELD <3%"]
            band_colors = ["#CC4A4A", "#5A8A5A", "#C8A84A", "#1E2E1E"]
            b_vals  = [int(df_bands.get(b, 0)) for b in band_order]
            b_shown = [b for b, v in zip(band_order, b_vals) if v > 0]
            b_clrs  = [c for b, c in zip(band_order, band_colors) if df_bands.get(b, 0) > 0]
            b_vshow = [v for v in b_vals if v > 0]

            # My team inner ring
            my_bands = [_band(float(df.loc[df["Player"] == p, own_col].values[0]))
                        if len(df.loc[df["Player"] == p]) else "FIELD <3%"
                        for p in my_players]
            my_band_counts = {b: my_bands.count(b) for b in band_order}
            mi_vals  = [int(my_band_counts.get(b, 0)) for b in band_order]
            mi_shown = [b for b, v in zip(band_order, mi_vals) if v > 0]
            mi_clrs  = [c for b, c in zip(band_order, band_colors) if my_band_counts.get(b, 0) > 0]
            mi_vshow = [v for v in mi_vals if v > 0]

            fig3a = go.Figure()
            fig3a.add_trace(go.Pie(
                values=b_vshow, labels=b_shown,
                marker_colors=b_clrs,
                hole=0.55,
                domain={"x": [0, 1], "y": [0, 1]},
                name="Full Field",
                textfont=dict(size=9, color="#96CC96"),
                hovertemplate="<b>%{label}</b><br>%{value} players (%{percent})<extra></extra>",
                showlegend=True,
            ))
            if mi_vshow:
                fig3a.add_trace(go.Pie(
                    values=mi_vshow, labels=mi_shown,
                    marker_colors=mi_clrs,
                    hole=0.20,
                    domain={"x": [0.2, 0.8], "y": [0.2, 0.8]},
                    name="My Teams",
                    textfont=dict(size=8),
                    hovertemplate="<b>My Teams — %{label}</b><br>%{value} picks<extra></extra>",
                    showlegend=False,
                ))
            fig3a.update_layout(
                paper_bgcolor="#060D06",
                font=dict(color="#96CC96", family="DM Sans"),
                title=dict(
                    text="Ownership Band Distribution<br>"
                         '<span style="font-size:9px;color:#4A6A4A;">'
                         "Outer = full field · Inner = my picks</span>",
                    font=dict(color="#C8A84A", size=13),
                ),
                legend=dict(bgcolor="#0D160D", bordercolor="#1E301E", borderwidth=1,
                            font=dict(size=9, color="#96CC96")),
                margin=dict(l=10, r=10, t=70, b=10),
                height=400,
                annotations=[dict(
                    text="2023: winners<br>from VALUE<br>3 of 4 slots",
                    x=0.5, y=0.5, font=dict(size=9, color="#C8A84A"),
                    showarrow=False, align="center",
                )],
            )
            st.plotly_chart(fig3a, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.warning(f"Ownership band chart not available: {e}")

    # Chart 3B — EV Score vs Odds scatter
    with pi_col2:
        try:
            def _implied(odds):
                try:
                    o = float(odds)
                    if o < 0: return abs(o) / (abs(o) + 100)
                    return 100 / (o + 100)
                except Exception:
                    return 0.05

            df_ev = df.copy()
            df_ev["Implied_Prob"] = df_ev["Odds_American"].apply(_implied)
            df_ev["Label"] = df_ev["Player"].apply(lambda n: n if n in membership else "")

            ev_clrs = [_pcolor(n) for n in df_ev["Player"]]
            aug_sizes = df_ev["Augusta_Score"].clip(20, 100)
            sz_norm = 6 + (aug_sizes - aug_sizes.min()) / (aug_sizes.max() - aug_sizes.min() + 1e-9) * 14

            # Fair value reference line
            implied_range = np.linspace(0, df_ev["Implied_Prob"].max() * 1.1, 50)
            fair_ev = implied_range * df_ev["Augusta_Score"].mean()

            fig3b = go.Figure()
            fig3b.add_trace(go.Scatter(
                x=df_ev["Implied_Prob"] * 100,
                y=df_ev["EV_Score"],
                mode="markers+text",
                text=df_ev["Label"],
                textposition="top center",
                textfont=dict(size=8, color="#96CC96"),
                marker=dict(color=ev_clrs, size=sz_norm, line=dict(width=0)),
                customdata=np.column_stack([
                    df_ev["Player"],
                    df_ev["Augusta_Score"].fillna(0),
                    df_ev["Odds_American"].fillna(0),
                    df_ev["Implied_Prob"].fillna(0) * 100,
                ]),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "EV Score: %{y:.1f}<br>"
                    "Implied Prob: %{customdata[3]:.1f}%<br>"
                    "Augusta Score: %{customdata[1]:.1f}<br>"
                    "Odds: +%{customdata[2]:.0f}<extra></extra>"
                ),
                name="Players",
            ))
            fig3b.add_trace(go.Scatter(
                x=implied_range * 100,
                y=fair_ev,
                mode="lines",
                line=dict(color="#2A422A", dash="dot", width=1),
                name="Fair Value",
                hoverinfo="skip",
            ))
            fig3b.add_annotation(
                x=implied_range[-1] * 100 * 0.9, y=fair_ev[-1] * 0.9,
                text="fair value", font=dict(color="#2A422A", size=9),
                showarrow=False,
            )
            _apply_chart_theme(fig3b, "EV Score vs Market Odds — Model vs Market", height=500)
            fig3b.update_layout(
                xaxis_title="Market Implied Win Probability (%)",
                yaxis_title="EV Score (Augusta Score / √Ownership)",
                showlegend=False,
            )
            st.plotly_chart(fig3b, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.warning(f"EV vs odds chart not available: {e}")

    # Chart 3C — Tiebreaker histogram
    try:
        weather = data.get("weather", {})
        chaos = weather.get("chaos_mode", False)
        rec_tb = -29 if chaos else -33
        cluster_lo, cluster_hi = (-30, -28) if chaos else (-32, -30)

        fig3c = go.Figure()
        fig3c.add_trace(go.Histogram(
            x=TIEBREAKER_2019,
            name="2019 Pool (n=146)",
            nbinsx=35,
            marker_color="#2A422A",
            opacity=0.7,
            hovertemplate="Score: %{x}<br>Count: %{y}<extra>2019</extra>",
        ))
        fig3c.add_trace(go.Histogram(
            x=TIEBREAKER_2023,
            name="2023 Pool (n=532)",
            nbinsx=35,
            marker_color="#3AAA5A",
            opacity=0.65,
            hovertemplate="Score: %{x}<br>Count: %{y}<extra>2023</extra>",
        ))
        fig3c.add_vrect(
            x0=cluster_lo, x1=cluster_hi,
            fillcolor="#C8A84A", opacity=0.08,
            line_width=0,
            annotation_text="cluster to beat",
            annotation_position="top left",
            annotation_font=dict(color="#C8A84A", size=9),
        )
        fig3c.add_vline(
            x=rec_tb, line_dash="solid", line_color="#C8A84A", line_width=2,
            annotation_text=f"Submit {rec_tb} — beats the {cluster_hi}/{cluster_hi-1} cluster",
            annotation_font=dict(color="#C8A84A", size=9),
            annotation_position="top right",
        )
        fig3c.update_layout(barmode="overlay")
        _apply_chart_theme(fig3c, "Pool Tiebreaker Distribution — 2019 & 2023 Actual Data", height=380)
        fig3c.update_layout(
            xaxis_title="Tiebreaker Submission (strokes to par)",
            yaxis_title="Number of Entries",
            legend=dict(bgcolor="#0D160D", bordercolor="#1E301E", borderwidth=1),
        )
        st.plotly_chart(fig3c, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.warning(f"Tiebreaker histogram not available: {e}")

    # ─── SECTION 4: MY TEAMS DEEP DIVE ────────────────────────────
    st.markdown(_section_header("MY TEAMS DEEP DIVE", "Player-level intelligence"), unsafe_allow_html=True)

    # Build player list from all 3 teams
    all_team_players: list[str] = []
    for tk in ["team_a", "team_b", "team_c"]:
        for p in ct.get(tk, {}).get("stats", []):
            n = p.get("Player", "")
            if n and n not in all_team_players:
                all_team_players.append(n)

    if not all_team_players:
        st.info("Visit MY PICKS tab first to generate team selections.")
        return

    selected_player = st.selectbox(
        "Select a player for deep dive",
        options=all_team_players,
        key="data_player_selector",
    )
    sel_team = membership.get(selected_player, "team_a")
    sel_color = MY_TEAM_COLORS[sel_team]

    # Get all players on the same team for comparison
    sel_team_players = [
        p.get("Player", "") for p in ct.get(sel_team, {}).get("stats", [])
    ]

    dive_col1, dive_col2 = st.columns(2, gap="medium")

    # Chart 4A — Selected player vs team average (grouped bar)
    with dive_col1:
        try:
            sel_row = df[df["Player"] == selected_player]
            if sel_row.empty:
                raise ValueError(f"{selected_player} not in df")
            sel_row = sel_row.iloc[0]

            # Team average (all 4 players on team)
            team_df = df[df["Player"].isin(sel_team_players)]
            team_avg = [float(team_df[c].mean()) for c in COMPONENTS]
            sel_vals = [_col(sel_row, c) for c in COMPONENTS]

            fig4a = go.Figure()
            fig4a.add_trace(go.Bar(
                name=selected_player.split()[-1],
                x=COMP_LABELS,
                y=sel_vals,
                marker_color=sel_color,
                hovertemplate="%{x}: %{y:.1f}<extra></extra>",
            ))
            fig4a.add_trace(go.Bar(
                name="Team Avg",
                x=COMP_LABELS,
                y=team_avg,
                marker_color={
                    "#3AAA5A": "rgba(58, 170, 90, 0.333)",
                    "#52CC72": "rgba(82, 204, 114, 0.333)",
                    "#C8A84A": "rgba(200, 168, 74, 0.333)",
                    "#2A3A2A": "rgba(42, 58, 42, 0.333)",
                }.get(sel_color, "rgba(58, 170, 90, 0.333)"),
                hovertemplate="Team Avg %{x}: %{y:.1f}<extra></extra>",
            ))
            fig4a.add_hline(y=65, line_dash="dot", line_color="#4A6A4A", line_width=1,
                            annotation_text="strong threshold",
                            annotation_font=dict(color="#4A6A4A", size=8))
            fig4a.update_layout(barmode="group")
            _apply_chart_theme(
                fig4a,
                f"{selected_player} vs Team Average",
                height=380,
            )
            fig4a.update_layout(yaxis=dict(range=[0, 105], gridcolor="#1E301E"))
            st.plotly_chart(fig4a, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.warning(f"Player comparison chart not available: {e}")

    # Chart 4B — Field percentile breakdown (horizontal bullet chart)
    with dive_col2:
        try:
            sel_row = df[df["Player"] == selected_player]
            if sel_row.empty:
                raise ValueError(f"{selected_player} not in df")
            sel_row = sel_row.iloc[0]

            # Compute percentiles
            overall_pct = float(
                (df["Augusta_Score"] <= float(sel_row["Augusta_Score"])).mean() * 100
            )

            comp_pcts = []
            for c in COMPONENTS:
                pct = float((df[c] <= float(sel_row.get(c, 50))).mean() * 100)
                comp_pcts.append(pct)

            all_labels  = ["Augusta Score"] + COMP_LABELS
            all_pcts    = [overall_pct] + comp_pcts
            all_vals    = [float(sel_row.get("Augusta_Score", 50))] + [_col(sel_row, c) for c in COMPONENTS]

            bar_colors_pct = []
            for p in all_pcts:
                if p >= 65:   bar_colors_pct.append("#3AAA5A")
                elif p >= 40: bar_colors_pct.append("#C8A84A")
                else:         bar_colors_pct.append("#CC4A4A")

            fig4b = go.Figure(go.Bar(
                x=all_pcts,
                y=all_labels,
                orientation="h",
                marker_color=bar_colors_pct,
                text=[f"{p:.0f}th pct  ({v:.1f})" for p, v in zip(all_pcts, all_vals)],
                textposition="inside",
                textfont=dict(size=9, color="#e8f5e8"),
                hovertemplate="%{y}: %{x:.1f}th percentile<extra></extra>",
            ))
            # Make "Augusta Score" bar taller visually
            fig4b.update_traces(width=[0.7 if l == "Augusta Score" else 0.5 for l in all_labels])

            _apply_chart_theme(fig4b, f"{selected_player} — Field Percentile Ranking", height=380)
            fig4b.update_layout(
                xaxis=dict(range=[0, 105], title="Field Percentile (%)", gridcolor="#1E301E"),
                yaxis=dict(gridcolor="#1E301E"),
                showlegend=False,
            )
            st.plotly_chart(fig4b, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.warning(f"Percentile chart not available: {e}")

    # Chart 4C — Team correlation matrix (heatmap)
    try:
        if len(all_team_players) >= 2:
            # Build matrix of component scores for all team players
            score_matrix = []
            valid_players = []
            for pname in all_team_players:
                prow = df[df["Player"] == pname]
                if not prow.empty:
                    vals = [_col(prow.iloc[0], c) for c in COMPONENTS]
                    score_matrix.append(vals)
                    valid_players.append(pname)

            if len(valid_players) >= 2:
                arr = np.array(score_matrix, dtype=float)
                # Pairwise cosine similarity as "profile similarity"
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1
                normed = arr / norms
                corr_mat = normed @ normed.T

                # Color: red = similar (>0.8), green = diverse (<0.4)
                colorscale = [
                    [0.0,  "#1A3A1A"],  # green (diverse)
                    [0.4,  "#3AAA5A"],
                    [0.7,  "#C8A84A"],
                    [1.0,  "#CC4A4A"],  # red (similar)
                ]

                # Group labels
                player_labels = []
                for pname in valid_players:
                    tk = membership.get(pname, "?")
                    suffix = {"team_a": "A", "team_b": "B", "team_c": "C"}.get(tk, "?")
                    player_labels.append(f"[{suffix}] {pname.split()[-1]}")

                fig4c = go.Figure(go.Heatmap(
                    z=corr_mat,
                    x=player_labels,
                    y=player_labels,
                    colorscale=colorscale,
                    zmin=0, zmax=1,
                    text=[[f"{v:.2f}" for v in row] for row in corr_mat],
                    texttemplate="%{text}",
                    textfont=dict(size=9),
                    hovertemplate="<b>%{y} vs %{x}</b><br>Similarity: %{z:.2f}<extra></extra>",
                    colorbar=dict(
                        title=dict(
                            text="Similarity",
                            font=dict(color="#C8A84A", size=11),
                        ),
                        tickfont=dict(color="#96CC96", size=9),
                        bgcolor="#0D160D",
                        bordercolor="#1E301E",
                        borderwidth=1,
                    ),
                ))
                _apply_chart_theme(
                    fig4c,
                    "Team Portfolio Correlation — Lower = Better Diversification",
                    height=500,
                )
                fig4c.update_layout(
                    xaxis=dict(tickfont=dict(size=9), side="bottom"),
                    yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
                    margin=dict(l=80, r=20, t=50, b=80),
                )
                st.plotly_chart(fig4c, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Need at least 2 players with data for correlation chart.")
        else:
            st.info("Select teams in MY PICKS to see correlation analysis.")
    except Exception as e:
        st.warning(f"Correlation matrix not available: {e}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    # ── CSS injection ──────────────────────────────────────────────
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # ── Session state cache-bust ───────────────────────────────────
    if st.session_state.get("data_version") != "1.2":
        for _k in ["custom_teams", "picks_confirmed", "scored_df", "teams",
                   "field_report", "_wk", "_teams_wk", "_field_report_wk"]:
            st.session_state.pop(_k, None)
        st.session_state["data_version"] = "1.2"

    # ── Data loading ───────────────────────────────────────────────
    if "data" not in st.session_state:
        with st.spinner("Loading data..."):
            st.session_state["data"] = fetch_all_data()
    data = st.session_state["data"]

    # ── Scoring (cached; rescored when weights change) ─────────────
    active_cw  = st.session_state.get("active_cw", None)
    chalk_flag = st.session_state.get("chalk_penalty", True)
    wk         = str(active_cw) + str(chalk_flag)

    if st.session_state.get("scored_df") is None or st.session_state.get("_wk") != wk:
        with st.spinner("Scoring players..."):
            st.session_state["scored_df"] = score_players(
                data,
                component_weights=active_cw,
                apply_chalk_penalty=chalk_flag,
            )
            st.session_state["_wk"] = wk

    df = st.session_state["scored_df"]

    # ── Field verification ─────────────────────────────────────────
    if st.session_state.get("_field_report_wk") != wk:
        from fetch_data import verify_and_filter_field
        _fdf, _freport = verify_and_filter_field(df)
        st.session_state["field_report"] = _freport
        st.session_state["_field_report_wk"] = wk
        # Replace df with filtered version (removes any stale withdrawn players)
        if _freport["withdrawn_removed"]:
            st.session_state["scored_df"] = _fdf
            df = _fdf
    field_report = st.session_state.get("field_report", {})

    # ── Team generation (cached; reset when df changes) ────────────
    if st.session_state.get("teams") is None or st.session_state.get("_teams_wk") != wk:
        with st.spinner("Building teams..."):
            pot_size    = st.session_state.get("pot_size",    40000)
            num_entries = st.session_state.get("num_entries", 500)
            st.session_state["teams"] = generate_teams(
                df, pot_size=pot_size, num_entries=num_entries,
                weather=data.get("weather", {})
            )
            st.session_state["_teams_wk"] = wk
            st.session_state.pop("custom_teams", None)

    teams = st.session_state["teams"]

    # ── Top bar ────────────────────────────────────────────────────
    render_topbar(data)

    # ── Weather / chaos banner ──────────────────────────────────────
    weather = data.get("weather", {})
    t_days  = weather.get("tournament_days", {})
    _day_labels = {"2026-04-09": "Thu", "2026-04-10": "Fri",
                   "2026-04-11": "Sat", "2026-04-12": "Sun"}
    _day_parts = []
    for _d, _lbl in _day_labels.items():
        _day = t_days.get(_d, {})
        if _day:
            _tlo = round(_day.get("temp_min", 0) * 9/5 + 32)
            _thi = round(_day.get("temp_max", 0) * 9/5 + 32)
            _wnd = round(_day.get("wind_max", 0))
            _day_parts.append(f"{_lbl} {_tlo}–{_thi}°F {_wnd}mph")
    _forecast_str = " &nbsp;·&nbsp; ".join(_day_parts) if _day_parts else "forecast loading"
    _condition    = weather.get("condition", "mild").replace("_", " ").title()

    if _detect_chaos_mode(weather):
        _max_wind = max((v.get("wind_max", 0) for v in t_days.values()), default=0)
        _min_temp_f = min(
            (round(v.get("temp_min", 20) * 9/5 + 32) for v in t_days.values()), default=65
        )
        st.markdown(
            '<div class="chaos-banner">'
            f'<b>CHAOS COEFFICIENT ACTIVE</b> — '
            f'Wind {_max_wind:.0f} mph / low {_min_temp_f}°F forecast. '
            'Weights auto-rebalanced: Form 35% &nbsp; Fit 30% &nbsp; DNA 15% &nbsp; Vegas 15% &nbsp; Traj 5%'
            f'<br><span style="font-size:9px;opacity:.7;">'
            f'Forecast: {_forecast_str}</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="background:#050d05;border:1px solid #152015;border-radius:4px;'
            f'padding:8px 14px;margin-bottom:12px;font-size:10px;color:#426842;">'
            f'<span style="color:#3aaa5a;font-weight:600;letter-spacing:.06em;">'
            f'CONDITIONS: {_condition.upper()}</span>'
            f'&nbsp;&nbsp;·&nbsp;&nbsp;{_forecast_str}'
            f'&nbsp;&nbsp;·&nbsp;&nbsp;'
            f'<span style="color:#2a422a;">Standard model weights active</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Field verification warning (global, shown on all tabs) ────
    if field_report.get("status") == "ISSUES FOUND":
        _issues = []
        if field_report.get("withdrawn_removed"):
            _issues.append(
                f"Removed withdrawn: {', '.join(field_report['withdrawn_removed'])}"
            )
        if field_report.get("missing_key_players"):
            _issues.append(
                f"Missing key players: {', '.join(sorted(field_report['missing_key_players']))}"
            )
        if _issues:
            st.warning("Field verification: " + " · ".join(_issues))

    # ── 4-Tab Layout ───────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["MY PICKS", "LIVE", "MODEL", "DATA"])

    with tab1:
        render_my_picks(df, teams, data)

    with tab2:
        render_live(df, data)

    with tab3:
        render_model(df, data, field_report=field_report)

    with tab4:
        _ct = st.session_state.get("custom_teams")
        if not _ct:
            # Derive from generate_teams output if custom_teams not yet initialized
            _ct = {
                "team_a": _enrich_team(teams.get("team_a", {}), df),
                "team_b": _enrich_team(teams.get("team_b", {}), df),
                "team_c": _enrich_team(teams.get("team_c", {}), df),
            }
        render_data_tab(df, _ct, data)


if __name__ == "__main__":
    main()
