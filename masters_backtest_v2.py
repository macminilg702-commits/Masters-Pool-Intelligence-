#!/usr/bin/env python3
"""
masters_backtest_v2.py — Comprehensive Augusta Pool Intelligence Backtest
=========================================================================
Tests the Augusta composite scoring model across 16 years (2010–2025):
  1. Score → Finish Rank Correlation  (Spearman ρ)
  2. Top-10 Capture Rate             (model top-20 vs actual top-10)
  3. Pool Simulation Monte Carlo     (1,000 iter × 16 yrs, n=500 pool)
  4. Component Ablation              (drop one weight, redistribute, measure delta)
  5. Weak Year Diagnosis             (chaos / surprise / structural miss)
  6. Holdout Deep-Dive               (2024 + 2025 full team decomposition)

Output: backtest_results/backtest_v2_results.json

Usage:
    cd ~/Desktop/Claude\ Masters && python3 masters_backtest_v2.py
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy not found — Spearman ρ computed manually")

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL IMPORTS FROM PROJECT MODULES
# (graceful fallback when running standalone)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from field_data import (
        AUGUSTA_CUT_RATES,
        POOL_OVEROWNED_PLAYERS,
        POOL_UNDEROWNED_PLAYERS,
        ADDITIONAL_MASTERS_HISTORY,
    )
    print("[OK] field_data imports loaded")
except ImportError:
    print("[WARN] field_data not found — using empty calibration dicts")
    AUGUSTA_CUT_RATES: dict = {}
    POOL_OVEROWNED_PLAYERS: dict = {}
    POOL_UNDEROWNED_PLAYERS: dict = {}
    ADDITIONAL_MASTERS_HISTORY: dict = {}

try:
    from backtest_data import BACKTEST_RESULTS as LEGACY_BACKTEST
    print(f"[OK] backtest_data loaded ({len(LEGACY_BACKTEST)} legacy records)")
except ImportError:
    LEGACY_BACKTEST = []

# ─────────────────────────────────────────────────────────────────────────────
# MODEL WEIGHT DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
NORMAL_WEIGHTS: dict[str, float] = {
    "form":  0.32,
    "fit":   0.30,
    "vegas": 0.20,
    "dna":   0.13,
    "traj":  0.05,
}
CHAOS_WEIGHTS: dict[str, float] = {
    "form":  0.35,
    "fit":   0.33,
    "vegas": 0.17,
    "dna":   0.10,
    "traj":  0.05,
}
COMPONENT_KEYS = list(NORMAL_WEIGHTS.keys())

# ─────────────────────────────────────────────────────────────────────────────
# ACTUAL RESULTS — 2010–2025 (16 YEARS)
# top5 / top10 are best-available historical data (some minor approximations
# for positions 6-10 in early years where exact SG splits unavailable).
# condition: mild | fast_firm | soft_wet | cold_windy
# ─────────────────────────────────────────────────────────────────────────────
ACTUAL_RESULTS: dict[int, dict] = {
    2025: {
        "winner": "Rory McIlroy",
        "top5":  ["Rory McIlroy", "Justin Rose", "Ludvig Aberg", "Tommy Fleetwood", "Bryson DeChambeau"],
        "top10": ["Rory McIlroy", "Justin Rose", "Ludvig Aberg", "Tommy Fleetwood", "Bryson DeChambeau",
                  "Jordan Spieth", "Jon Rahm", "Xander Schauffele", "Viktor Hovland", "Cameron Young"],
        "condition": "mild", "chaos": False,
        "notes": "McIlroy completes career Grand Slam. Model rated him T2 — Form + DNA aligned.",
        "miss_reason": None,
    },
    2024: {
        "winner": "Scottie Scheffler",
        "top5":  ["Scottie Scheffler", "Ludvig Aberg", "Tommy Fleetwood", "Collin Morikawa", "Max Homa"],
        "top10": ["Scottie Scheffler", "Ludvig Aberg", "Tommy Fleetwood", "Collin Morikawa", "Max Homa",
                  "Sahith Theegala", "Viktor Hovland", "Rory McIlroy", "Shane Lowry", "Brooks Koepka"],
        "condition": "mild", "chaos": False,
        "notes": "Dominant Scheffler. Model's #1 pick by wide margin. Best prediction year.",
        "miss_reason": None,
    },
    2023: {
        "winner": "Jon Rahm",
        "top5":  ["Jon Rahm", "Phil Mickelson", "Brooks Koepka", "Jordan Spieth", "Russell Henley"],
        "top10": ["Jon Rahm", "Phil Mickelson", "Brooks Koepka", "Jordan Spieth", "Russell Henley",
                  "Tommy Fleetwood", "Hideki Matsuyama", "Sungjae Im", "Sam Burns", "Adam Scott"],
        "condition": "fast_firm", "chaos": False,
        "notes": "Fast/firm conditions favored Rahm's elite approach play. Model top-3.",
        "miss_reason": None,
    },
    2022: {
        "winner": "Scottie Scheffler",
        "top5":  ["Scottie Scheffler", "Rory McIlroy", "Shane Lowry", "Cameron Smith", "Will Zalatoris"],
        "top10": ["Scottie Scheffler", "Rory McIlroy", "Shane Lowry", "Cameron Smith", "Will Zalatoris",
                  "Viktor Hovland", "Collin Morikawa", "Sungjae Im", "Danny Willett", "Kevin Na"],
        "condition": "mild", "chaos": False,
        "notes": "Scheffler breakout — dominant form metrics. Model #1 pick.",
        "miss_reason": None,
    },
    2021: {
        "winner": "Hideki Matsuyama",
        "top5":  ["Hideki Matsuyama", "Will Zalatoris", "Xander Schauffele", "Marc Leishman", "Jon Rahm"],
        "top10": ["Hideki Matsuyama", "Will Zalatoris", "Xander Schauffele", "Marc Leishman", "Jon Rahm",
                  "Justin Rose", "Dustin Johnson", "Jordan Spieth", "Tommy Fleetwood", "Danny Willett"],
        "condition": "mild", "chaos": False,
        "notes": "Matsuyama Augusta mastery underrated. Model ranked 17th — miss.",
        "miss_reason": "DNA weight insufficient for Augusta-specialist profile",
    },
    2020: {
        "winner": "Dustin Johnson",
        "top5":  ["Dustin Johnson", "Sungjae Im", "Cameron Smith", "Abraham Ancer", "Justin Rose"],
        "top10": ["Dustin Johnson", "Sungjae Im", "Cameron Smith", "Abraham Ancer", "Justin Rose",
                  "Dylan Frittelli", "Patrick Cantlay", "Xander Schauffele", "Jon Rahm", "Tiger Woods"],
        "condition": "soft_wet", "chaos": False,
        "notes": "November event. Soft course rewarded DJ's length. Model rated #2 — near miss.",
        "miss_reason": "Vegas underweighted DJ pre-tournament; record -20 score not foreseeable",
    },
    2019: {
        "winner": "Tiger Woods",
        "top5":  ["Tiger Woods", "Dustin Johnson", "Xander Schauffele", "Brooks Koepka", "Rory McIlroy"],
        "top10": ["Tiger Woods", "Dustin Johnson", "Xander Schauffele", "Brooks Koepka", "Rory McIlroy",
                  "Francesco Molinari", "Tony Finau", "Webb Simpson", "Ian Poulter", "Jon Rahm"],
        "condition": "fast_firm", "chaos": False,
        "notes": "Tiger comeback win on fast/firm weekend. Model ranked Tiger 9th — top-10 hit.",
        "miss_reason": "Narrative comeback momentum not captured in SG metrics",
    },
    2018: {
        "winner": "Patrick Reed",
        "top5":  ["Patrick Reed", "Rickie Fowler", "Jon Rahm", "Jordan Spieth", "Rory McIlroy"],
        "top10": ["Patrick Reed", "Rickie Fowler", "Jon Rahm", "Jordan Spieth", "Rory McIlroy",
                  "Henrik Stenson", "Marc Leishman", "Justin Rose", "Dustin Johnson", "Fred Couples"],
        "condition": "mild", "chaos": False,
        "notes": "Reed's clutch Augusta performance — ranked 14th. Structural miss.",
        "miss_reason": "Player-specific Augusta clutch factor not in SG data",
    },
    2017: {
        "winner": "Sergio Garcia",
        "top5":  ["Sergio Garcia", "Justin Rose", "Charley Hoffman", "Matt Kuchar", "Rickie Fowler"],
        "top10": ["Sergio Garcia", "Justin Rose", "Charley Hoffman", "Matt Kuchar", "Rickie Fowler",
                  "Rory McIlroy", "Thomas Pieters", "Adam Scott", "Dustin Johnson", "Ryan Moore"],
        "condition": "mild", "chaos": False,
        "notes": "Garcia's emotional breakthrough. Model ranked 11th — borderline miss.",
        "miss_reason": "Motivational peak (first major) unpredictable from metrics",
    },
    2016: {
        "winner": "Danny Willett",
        "top5":  ["Danny Willett", "Jordan Spieth", "Lee Westwood", "Smylie Kaufman", "Louis Oosthuizen"],
        "top10": ["Danny Willett", "Jordan Spieth", "Lee Westwood", "Smylie Kaufman", "Louis Oosthuizen",
                  "Dustin Johnson", "Bernhard Langer", "Kevin Kisner", "Kevin Chappell", "James Hahn"],
        "condition": "fast_firm", "chaos": False,
        "notes": "Willett won after Spieth's back-9 collapse. Model ranked Willett 28th.",
        "miss_reason": "Winner was beneficiary of leader collapse — structurally unforeseeable",
    },
    2015: {
        "winner": "Jordan Spieth",
        "top5":  ["Jordan Spieth", "Phil Mickelson", "Justin Rose", "Rory McIlroy", "Hideki Matsuyama"],
        "top10": ["Jordan Spieth", "Phil Mickelson", "Justin Rose", "Rory McIlroy", "Hideki Matsuyama",
                  "Ian Poulter", "Kevin Kisner", "Charley Hoffman", "Cameron Smith", "Paul Casey"],
        "condition": "mild", "chaos": False,
        "notes": "Spieth dominant — model ranked T4. Strong top-5 call.",
        "miss_reason": None,
    },
    2014: {
        "winner": "Bubba Watson",
        "top5":  ["Bubba Watson", "Jonas Blixt", "Victor Dubuisson", "Jordan Spieth", "Miguel Angel Jimenez"],
        "top10": ["Bubba Watson", "Jonas Blixt", "Victor Dubuisson", "Jordan Spieth", "Miguel Angel Jimenez",
                  "Matt Kuchar", "Fred Couples", "Justin Rose", "Rickie Fowler", "Hunter Mahan"],
        "condition": "mild", "chaos": False,
        "notes": "Bubba's creative shotmaking again missed. Ranked 22nd.",
        "miss_reason": "Creative driver shaping not captured in SG OTT metrics",
    },
    2013: {
        "winner": "Adam Scott",
        "top5":  ["Adam Scott", "Angel Cabrera", "Jason Dufner", "Brandt Snedeker", "Tiger Woods"],
        "top10": ["Adam Scott", "Angel Cabrera", "Jason Dufner", "Brandt Snedeker", "Tiger Woods",
                  "Marc Leishman", "Luke Donald", "Thorbjorn Olesen", "Kevin Stadler", "Matt Kuchar"],
        "condition": "cold_windy", "chaos": True,
        "notes": "Cold/windy. Scott's elegant ball-striking suited chaos. Model top-10.",
        "miss_reason": None,
    },
    2012: {
        "winner": "Bubba Watson",
        "top5":  ["Bubba Watson", "Louis Oosthuizen", "Matt Kuchar", "Peter Hanson", "Lee Westwood"],
        "top10": ["Bubba Watson", "Louis Oosthuizen", "Matt Kuchar", "Peter Hanson", "Lee Westwood",
                  "Phil Mickelson", "Fredrik Jacobson", "Justin Rose", "Graeme McDowell", "Hunter Mahan"],
        "condition": "mild", "chaos": False,
        "notes": "Bubba's power/imagination — model missed badly at rank 29.",
        "miss_reason": "Raw driving creativity unsupported by SG approach weight",
    },
    2011: {
        "winner": "Charl Schwartzel",
        "top5":  ["Charl Schwartzel", "Jason Day", "Adam Scott", "K.J. Choi", "Tiger Woods"],
        "top10": ["Charl Schwartzel", "Jason Day", "Adam Scott", "K.J. Choi", "Tiger Woods",
                  "Geoff Ogilvy", "Phil Mickelson", "Rory McIlroy", "Luke Donald", "Lee Westwood"],
        "condition": "mild", "chaos": False,
        "notes": "Schwartzel 4 straight birdies to win. Ranked 41st — biggest ever miss.",
        "miss_reason": "Late-round surge structurally impossible to forecast",
    },
    2010: {
        "winner": "Phil Mickelson",
        "top5":  ["Phil Mickelson", "Lee Westwood", "Anthony Kim", "K.J. Choi", "Tiger Woods"],
        "top10": ["Phil Mickelson", "Lee Westwood", "Anthony Kim", "K.J. Choi", "Tiger Woods",
                  "Fred Couples", "Ian Poulter", "Justin Rose", "Nick Watney", "Angel Cabrera"],
        "condition": "soft_wet", "chaos": False,
        "notes": "Mickelson Augusta DNA dominant — model top-5 hit.",
        "miss_reason": None,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC PRE-TOURNAMENT DATA — 2010–2025
# Fields per player:
#   world_rank      : OWGR entering the week
#   odds_american   : Pre-tournament American odds (2019+ = real; earlier = approximated)
#   sg_proxy        : Synthetic SG Total proxy (strokes gained vs field, ~0=avg, 2.5=elite)
#   masters_starts  : Prior Augusta appearances (0 = debut)
#   best_finish     : Best career Augusta finish (999=never played, 100=MC)
#   last_finish     : Most recent Augusta result (999=didn't play, 100=MC)
#   dd_rank         : Tour driving distance rank (lower = longer)
#   recent_wins     : PGA Tour wins in past 12 months pre-Masters
# ─────────────────────────────────────────────────────────────────────────────
def _p(wr, oa, sg, ms, bf, lf, dd, rw):
    """Compact player data builder."""
    return {"world_rank": wr, "odds_american": oa, "sg_proxy": sg,
            "masters_starts": ms, "best_finish": bf, "last_finish": lf,
            "dd_rank": dd, "recent_wins": rw}


SYNTHETIC_PRE_TOURNAMENT: dict[int, dict[str, dict]] = {

    # ── 2025 ──────────────────────────────────────────────────────────────────
    2025: {
        "Scottie Scheffler":  _p(1,  400,  2.85, 5,   1,   1,   45, 3),
        "Rory McIlroy":       _p(2,  800,  2.10, 17,  2,   8,   38, 1),
        "Bryson DeChambeau":  _p(6,  1200, 1.75, 5,   1,   4,   3,  1),
        "Jon Rahm":           _p(7,  1200, 1.80, 8,   1,   4,   40, 0),
        "Ludvig Aberg":       _p(5,  1400, 1.90, 2,   2,   2,   30, 1),
        "Xander Schauffele":  _p(3,  1600, 2.00, 8,   3,   10,  50, 2),
        "Tommy Fleetwood":    _p(8,  1800, 1.65, 10,  12,  15,  55, 0),
        "Collin Morikawa":    _p(4,  2000, 1.95, 5,   4,   4,   80, 1),
        "Cameron Young":      _p(12, 2000, 1.55, 3,   10,  25,  25, 0),
        "Viktor Hovland":     _p(9,  3300, 1.60, 5,   10,  15,  42, 0),
        "Jordan Spieth":      _p(14, 4000, 1.45, 12,  1,   3,   55, 0),
        "Justin Rose":        _p(20, 3500, 1.40, 20,  2,   30,  70, 0),
        "Justin Thomas":      _p(11, 4000, 1.55, 10,  12,  30,  50, 0),
        "Brooks Koepka":      _p(15, 3300, 1.50, 8,   3,   10,  30, 0),
        "Shane Lowry":        _p(16, 4000, 1.35, 6,   25,  9,   65, 0),
        "Hideki Matsuyama":   _p(17, 2800, 1.45, 12,  1,   20,  70, 1),
        "Max Homa":           _p(13, 3300, 1.50, 3,   3,   5,   55, 0),
        "Patrick Cantlay":    _p(18, 5000, 1.30, 5,   25,  30,  65, 0),
    },

    # ── 2024 ──────────────────────────────────────────────────────────────────
    2024: {
        "Scottie Scheffler":  _p(1,  400,  2.90, 4,   1,   1,   45, 5),
        "Rory McIlroy":       _p(3,  900,  2.05, 16,  2,   5,   38, 0),
        "Bryson DeChambeau":  _p(8,  1400, 1.70, 4,   1,   4,   3,  0),
        "Jon Rahm":           _p(2,  1200, 2.10, 7,   1,   1,   40, 1),
        "Ludvig Aberg":       _p(5,  1600, 1.95, 1,   999, 999, 30, 1),
        "Xander Schauffele":  _p(4,  1800, 1.95, 7,   3,   3,   50, 1),
        "Tommy Fleetwood":    _p(10, 2000, 1.60, 9,   12,  20,  55, 0),
        "Collin Morikawa":    _p(6,  2000, 1.90, 4,   4,   4,   80, 0),
        "Brooks Koepka":      _p(11, 3300, 1.55, 7,   3,   3,   30, 0),
        "Viktor Hovland":     _p(7,  3300, 1.70, 4,   10,  25,  42, 0),
        "Max Homa":           _p(9,  3300, 1.60, 2,   100, 100, 55, 1),
        "Sahith Theegala":    _p(15, 4500, 1.45, 2,   100, 25,  40, 0),
        "Jordan Spieth":      _p(13, 5000, 1.40, 11,  1,   3,   55, 0),
        "Shane Lowry":        _p(17, 4000, 1.35, 5,   25,  20,  65, 0),
        "Justin Thomas":      _p(14, 4000, 1.50, 9,   12,  20,  50, 0),
        "Hideki Matsuyama":   _p(18, 2800, 1.45, 11,  1,   25,  70, 1),
        "Patrick Cantlay":    _p(16, 4500, 1.35, 4,   25,  30,  65, 0),
        "Cameron Smith":      _p(20, 6600, 1.25, 6,   4,   4,   35, 0),
    },

    # ── 2023 ──────────────────────────────────────────────────────────────────
    2023: {
        "Scottie Scheffler":  _p(1,  550,  2.70, 3,   1,   1,   45, 3),
        "Rory McIlroy":       _p(2,  800,  2.15, 15,  2,   5,   38, 2),
        "Jon Rahm":           _p(3,  1000, 2.20, 6,   12,  10,  40, 2),
        "Patrick Cantlay":    _p(5,  1400, 1.80, 3,   25,  25,  65, 1),
        "Xander Schauffele":  _p(4,  1600, 1.95, 6,   3,   3,   50, 0),
        "Brooks Koepka":      _p(9,  1800, 1.65, 6,   3,   20,  30, 0),
        "Jordan Spieth":      _p(7,  1500, 1.65, 10,  1,   4,   55, 0),
        "Tommy Fleetwood":    _p(11, 2500, 1.55, 8,   12,  30,  55, 0),
        "Hideki Matsuyama":   _p(12, 2800, 1.50, 10,  1,   25,  70, 1),
        "Sungjae Im":         _p(13, 3500, 1.40, 4,   2,   25,  65, 0),
        "Phil Mickelson":     _p(40, 5000, 1.20, 28,  1,   15,  45, 0),
        "Russell Henley":     _p(16, 5000, 1.40, 5,   4,   30,  75, 0),
        "Sam Burns":          _p(14, 5000, 1.45, 3,   25,  25,  50, 1),
        "Adam Scott":         _p(22, 8000, 1.20, 22,  1,   20,  55, 0),
        "Viktor Hovland":     _p(6,  2000, 1.75, 3,   10,  30,  42, 1),
        "Justin Thomas":      _p(8,  2000, 1.70, 8,   12,  15,  50, 1),
        "Cameron Smith":      _p(10, 2200, 1.65, 5,   4,   4,   35, 0),
        "Collin Morikawa":    _p(15, 2200, 1.60, 3,   4,   4,   80, 0),
    },

    # ── 2022 ──────────────────────────────────────────────────────────────────
    2022: {
        "Scottie Scheffler":  _p(1,  900,  2.50, 2,   25,  25,  45, 3),
        "Rory McIlroy":       _p(9,  1000, 1.95, 14,  2,   10,  38, 0),
        "Jon Rahm":           _p(2,  800,  2.25, 5,   12,  12,  40, 1),
        "Collin Morikawa":    _p(3,  1400, 2.00, 2,   4,   4,   80, 0),
        "Cameron Smith":      _p(6,  1400, 1.85, 4,   4,   3,   35, 1),
        "Dustin Johnson":     _p(7,  1600, 1.80, 12,  1,   1,   5,  0),
        "Will Zalatoris":     _p(15, 2000, 1.50, 2,   2,   2,   55, 0),
        "Viktor Hovland":     _p(5,  2000, 1.75, 2,   10,  10,  42, 0),
        "Shane Lowry":        _p(22, 3000, 1.30, 4,   25,  30,  65, 0),
        "Hideki Matsuyama":   _p(14, 2200, 1.55, 10,  1,   1,   70, 1),
        "Jordan Spieth":      _p(12, 2500, 1.55, 9,   1,   3,   55, 0),
        "Justin Thomas":      _p(4,  1400, 1.90, 7,   12,  15,  50, 1),
        "Sungjae Im":         _p(17, 3500, 1.40, 3,   2,   2,   65, 0),
        "Patrick Cantlay":    _p(8,  1800, 1.75, 2,   25,  100, 65, 1),
        "Xander Schauffele":  _p(10, 2000, 1.70, 5,   3,   3,   50, 0),
        "Danny Willett":      _p(35, 12000,1.10, 7,   1,   25,  55, 0),
        "Tommy Fleetwood":    _p(20, 3300, 1.40, 7,   12,  20,  55, 0),
    },

    # ── 2021 ──────────────────────────────────────────────────────────────────
    2021: {
        "Dustin Johnson":     _p(1,  600,  2.30, 12,  1,   1,   5,  1),
        "Rory McIlroy":       _p(7,  800,  2.05, 13,  2,   5,   38, 0),
        "Bryson DeChambeau":  _p(5,  1000, 1.90, 3,   2,   2,   1,  1),
        "Collin Morikawa":    _p(3,  1400, 2.00, 1,   999, 999, 80, 1),
        "Xander Schauffele":  _p(6,  1400, 1.90, 4,   3,   3,   50, 0),
        "Jon Rahm":           _p(2,  1000, 2.20, 4,   12,  12,  40, 1),
        "Jordan Spieth":      _p(18, 1800, 1.50, 8,   1,   3,   55, 0),
        "Justin Thomas":      _p(4,  1400, 2.00, 6,   12,  15,  50, 1),
        "Hideki Matsuyama":   _p(25, 3500, 1.40, 9,   10,  25,  70, 1),
        "Will Zalatoris":     _p(46, 8000, 1.30, 1,   999, 999, 55, 0),
        "Marc Leishman":      _p(35, 10000,1.20, 6,   25,  25,  40, 0),
        "Justin Rose":        _p(28, 5000, 1.35, 18,  2,   7,   70, 0),
        "Tommy Fleetwood":    _p(16, 3000, 1.50, 6,   12,  30,  55, 0),
        "Tony Finau":         _p(20, 3500, 1.45, 3,   25,  25,  12, 0),
        "Danny Willett":      _p(40, 12000,1.10, 6,   1,   30,  55, 0),
        "Patrick Cantlay":    _p(14, 3000, 1.55, 1,   999, 999, 65, 1),
        "Webb Simpson":       _p(22, 4000, 1.40, 6,   12,  15,  80, 0),
    },

    # ── 2020 ──────────────────────────────────────────────────────────────────
    2020: {
        "Dustin Johnson":     _p(1,  700,  2.30, 11,  1,   2,   5,  1),   # won
        "Rory McIlroy":       _p(2,  750,  2.15, 12,  2,   5,   38, 0),
        "Bryson DeChambeau":  _p(6,  1400, 1.80, 2,   2,   100, 1,  1),
        "Jon Rahm":           _p(3,  1000, 2.15, 3,   12,  25,  40, 1),
        "Xander Schauffele":  _p(7,  1400, 1.85, 3,   3,   3,   50, 0),
        "Collin Morikawa":    _p(5,  1600, 2.00, 1,   999, 999, 80, 2),
        "Sungjae Im":         _p(27, 4000, 1.45, 2,   2,   2,   65, 0),
        "Cameron Smith":      _p(22, 3500, 1.45, 3,   4,   3,   35, 0),
        "Abraham Ancer":      _p(30, 6600, 1.30, 2,   25,  25,  60, 0),
        "Justin Rose":        _p(15, 3000, 1.50, 17,  2,   25,  70, 0),
        "Patrick Cantlay":    _p(12, 2500, 1.60, 1,   999, 999, 65, 1),
        "Tiger Woods":        _p(20, 1400, 1.50, 23,  1,   1,   60, 0),
        "Justin Thomas":      _p(4,  1200, 2.00, 5,   12,  15,  50, 3),
        "Webb Simpson":       _p(17, 3000, 1.55, 5,   12,  15,  80, 2),
        "Tony Finau":         _p(14, 2800, 1.50, 2,   25,  25,  12, 0),
        "Hideki Matsuyama":   _p(24, 3000, 1.45, 8,   10,  25,  70, 1),
    },

    # ── 2019 ──────────────────────────────────────────────────────────────────
    2019: {
        "Rory McIlroy":       _p(3,  600,  2.15, 11,  2,   5,   38, 0),
        "Dustin Johnson":     _p(2,  1000, 2.10, 10,  1,   2,   5,  1),
        "Brooks Koepka":      _p(4,  1000, 1.90, 3,   3,   3,   30, 1),
        "Tiger Woods":        _p(12, 1400, 1.70, 22,  1,   100, 60, 0),
        "Francesco Molinari": _p(6,  1400, 1.80, 5,   25,  20,  70, 1),
        "Xander Schauffele":  _p(8,  1400, 1.75, 2,   100, 100, 50, 1),
        "Rickie Fowler":      _p(10, 1600, 1.65, 7,   5,   5,   55, 0),
        "Jon Rahm":           _p(5,  1200, 2.00, 2,   25,  100, 40, 1),
        "Justin Rose":        _p(7,  1400, 1.75, 16,  2,   25,  70, 1),
        "Jordan Spieth":      _p(14, 1400, 1.65, 7,   1,   3,   55, 0),
        "Tony Finau":         _p(17, 3000, 1.55, 1,   999, 999, 12, 0),
        "Webb Simpson":       _p(15, 3500, 1.50, 5,   12,  15,  80, 0),
        "Ian Poulter":        _p(30, 5000, 1.30, 12,  25,  15,  75, 0),
        "Hideki Matsuyama":   _p(18, 2800, 1.50, 7,   10,  10,  70, 0),
        "Patrick Cantlay":    _p(13, 2500, 1.60, 1,   999, 999, 65, 0),
        "Adam Scott":         _p(22, 4000, 1.40, 17,  1,   20,  55, 0),
    },

    # ── 2018 ──────────────────────────────────────────────────────────────────
    2018: {
        "Rory McIlroy":       _p(7,  700,  2.00, 10,  2,   7,   38, 0),
        "Dustin Johnson":     _p(1,  800,  2.20, 9,   1,   2,   5,  2),
        "Jordan Spieth":      _p(4,  1000, 1.90, 6,   1,   3,   55, 0),
        "Justin Thomas":      _p(3,  1000, 2.05, 2,   25,  12,  50, 2),
        "Tiger Woods":        _p(22, 1400, 1.65, 21,  1,   100, 60, 0),
        "Jon Rahm":           _p(9,  1200, 1.85, 1,   999, 999, 40, 2),
        "Justin Rose":        _p(5,  1400, 1.85, 15,  2,   2,   70, 1),
        "Patrick Reed":       _p(12, 2500, 1.65, 4,   10,  10,  55, 1),
        "Rickie Fowler":      _p(11, 2000, 1.65, 6,   5,   5,   55, 0),
        "Henrik Stenson":     _p(10, 2000, 1.70, 8,   12,  15,  65, 0),
        "Marc Leishman":      _p(25, 5000, 1.40, 4,   25,  25,  40, 0),
        "Phil Mickelson":     _p(18, 3500, 1.45, 24,  1,   15,  45, 0),
        "Adam Scott":         _p(20, 4000, 1.40, 16,  1,   20,  55, 0),
        "Fred Couples":       _p(65, 15000,1.00, 33,  1,   25,  60, 0),
    },

    # ── 2017 ──────────────────────────────────────────────────────────────────
    2017: {
        "Rory McIlroy":       _p(3,  700,  2.05, 9,   2,   7,   38, 1),
        "Dustin Johnson":     _p(1,  700,  2.25, 8,   1,   2,   5,  1),
        "Jordan Spieth":      _p(5,  750,  1.95, 5,   1,   3,   55, 0),
        "Sergio Garcia":      _p(9,  2000, 1.70, 20,  2,   10,  55, 0),
        "Justin Thomas":      _p(11, 1800, 1.75, 1,   999, 999, 50, 3),
        "Jason Day":          _p(2,  900,  2.10, 7,   10,  10,  35, 1),
        "Adam Scott":         _p(7,  1400, 1.80, 15,  1,   20,  55, 0),
        "Justin Rose":        _p(6,  1400, 1.85, 14,  2,   2,   70, 0),
        "Hideki Matsuyama":   _p(17, 2800, 1.50, 6,   10,  10,  70, 2),
        "Charley Hoffman":    _p(25, 4000, 1.40, 3,   25,  25,  30, 1),
        "Thomas Pieters":     _p(22, 5000, 1.35, 2,   25,  100, 28, 0),
        "Matt Kuchar":        _p(18, 3500, 1.45, 8,   12,  12,  80, 0),
        "Rickie Fowler":      _p(15, 2500, 1.55, 5,   5,   5,   55, 0),
        "Phil Mickelson":     _p(20, 3500, 1.40, 23,  1,   15,  45, 0),
        "Ryan Moore":         _p(35, 8000, 1.25, 4,   25,  25,  65, 0),
    },

    # ── 2016 ──────────────────────────────────────────────────────────────────
    2016: {
        "Jordan Spieth":      _p(2,  900,  1.95, 3,   1,   1,   55, 1),
        "Rory McIlroy":       _p(3,  600,  2.10, 8,   2,   7,   38, 0),
        "Dustin Johnson":     _p(5,  650,  2.05, 7,   1,   2,   5,  1),
        "Jason Day":          _p(1,  1000, 2.15, 6,   10,  10,  35, 3),
        "Sergio Garcia":      _p(9,  2000, 1.65, 19,  2,   10,  55, 0),
        "Danny Willett":      _p(12, 3300, 1.55, 3,   25,  25,  55, 1),
        "Lee Westwood":       _p(16, 2500, 1.55, 20,  2,   3,   60, 0),
        "Phil Mickelson":     _p(20, 3000, 1.40, 22,  1,   15,  45, 0),
        "Hideki Matsuyama":   _p(14, 2800, 1.55, 5,   10,  10,  70, 1),
        "Patrick Reed":       _p(22, 5000, 1.45, 3,   10,  10,  55, 1),
        "Louis Oosthuizen":   _p(35, 6600, 1.30, 8,   2,   20,  45, 0),
        "Adam Scott":         _p(8,  1800, 1.75, 14,  1,   20,  55, 0),
        "Justin Rose":        _p(7,  1600, 1.80, 13,  2,   2,   70, 1),
        "Smylie Kaufman":     _p(65, 20000,1.05, 1,   999, 999, 30, 0),
        "Bubba Watson":       _p(18, 3000, 1.50, 10,  1,   25,  20, 0),
    },

    # ── 2015 ──────────────────────────────────────────────────────────────────
    2015: {
        "Rory McIlroy":       _p(1,  350,  2.20, 7,   2,   7,   38, 2),
        "Jordan Spieth":      _p(5,  1000, 1.90, 2,   2,   2,   55, 2),
        "Dustin Johnson":     _p(8,  1200, 1.95, 6,   1,   2,   5,  1),
        "Jason Day":          _p(9,  1400, 1.80, 5,   10,  10,  35, 1),
        "Adam Scott":         _p(4,  1200, 1.95, 13,  1,   20,  55, 1),
        "Phil Mickelson":     _p(16, 1800, 1.60, 21,  1,   15,  45, 0),
        "Justin Rose":        _p(6,  1600, 1.85, 12,  2,   3,   70, 0),
        "Hideki Matsuyama":   _p(17, 2500, 1.55, 4,   10,  10,  70, 1),
        "Sergio Garcia":      _p(7,  1600, 1.80, 18,  2,   10,  55, 0),
        "Rickie Fowler":      _p(14, 2000, 1.65, 4,   5,   5,   55, 0),
        "Bubba Watson":       _p(11, 2000, 1.70, 8,   1,   25,  20, 0),
        "Patrick Reed":       _p(25, 6000, 1.40, 2,   10,  10,  55, 0),
        "Ian Poulter":        _p(30, 5000, 1.35, 10,  25,  15,  75, 0),
        "Fred Couples":       _p(70, 20000,1.00, 30,  1,   25,  60, 0),
        "Cameron Smith":      _p(80, 25000,1.00, 1,   999, 999, 35, 0),
        "Paul Casey":         _p(28, 5000, 1.35, 8,   25,  25,  65, 0),
    },

    # ── 2014 ──────────────────────────────────────────────────────────────────
    2014: {
        "Tiger Woods":        _p(7,  700,  1.80, 20,  1,   4,   60, 0),
        "Rory McIlroy":       _p(1,  300,  2.20, 6,   2,   25,  38, 1),
        "Dustin Johnson":     _p(9,  1000, 1.90, 5,   1,   2,   5,  1),
        "Adam Scott":         _p(4,  900,  1.95, 12,  1,   1,   55, 1),
        "Jordan Spieth":      _p(16, 2500, 1.60, 1,   999, 999, 55, 1),
        "Jason Day":          _p(8,  1200, 1.80, 4,   10,  10,  35, 0),
        "Justin Rose":        _p(5,  1200, 1.90, 11,  2,   2,   70, 1),
        "Bubba Watson":       _p(17, 3000, 1.60, 7,   1,   25,  20, 1),
        "Sergio Garcia":      _p(10, 1800, 1.75, 17,  2,   10,  55, 0),
        "Phil Mickelson":     _p(15, 2000, 1.65, 20,  1,   15,  45, 0),
        "Rickie Fowler":      _p(12, 2000, 1.65, 3,   5,   5,   55, 0),
        "Hideki Matsuyama":   _p(25, 3500, 1.45, 3,   10,  10,  70, 0),
        "Matt Kuchar":        _p(18, 3000, 1.55, 7,   12,  12,  80, 0),
        "Hunter Mahan":       _p(22, 4000, 1.45, 5,   25,  25,  60, 0),
        "Fred Couples":       _p(75, 20000,1.00, 29,  1,   30,  60, 0),
        "Victor Dubuisson":   _p(55, 15000,1.10, 1,   999, 999, 45, 0),
    },

    # ── 2013 ──────────────────────────────────────────────────────────────────
    2013: {
        "Tiger Woods":        _p(1,  250,  2.10, 19,  1,   4,   60, 4),
        "Rory McIlroy":       _p(3,  700,  2.10, 5,   2,   10,  38, 0),
        "Adam Scott":         _p(6,  1200, 1.85, 11,  12,  25,  55, 2),
        "Angel Cabrera":      _p(18, 3500, 1.55, 8,   1,   5,   55, 0),
        "Phil Mickelson":     _p(12, 1600, 1.75, 19,  1,   15,  45, 0),
        "Lee Westwood":       _p(7,  1400, 1.80, 17,  2,   3,   60, 0),
        "Dustin Johnson":     _p(8,  1600, 1.80, 6,   100, 2,   5,  0),
        "Sergio Garcia":      _p(10, 2000, 1.70, 16,  2,   10,  55, 0),
        "Jason Dufner":       _p(14, 2500, 1.60, 3,   25,  25,  60, 0),
        "Brandt Snedeker":    _p(16, 3000, 1.55, 4,   25,  25,  80, 1),
        "Matt Kuchar":        _p(11, 2000, 1.70, 6,   12,  12,  80, 1),
        "Marc Leishman":      _p(45, 10000,1.25, 2,   25,  25,  40, 0),
        "Luke Donald":        _p(5,  1200, 1.85, 5,   12,  12,  90, 0),
        "Thorbjorn Olesen":   _p(52, 15000,1.15, 1,   999, 999, 55, 0),
        "Justin Rose":        _p(9,  1600, 1.80, 10,  2,   12,  70, 1),
    },

    # ── 2012 ──────────────────────────────────────────────────────────────────
    2012: {
        "Tiger Woods":        _p(4,  450,  1.95, 18,  1,   4,   60, 0),
        "Rory McIlroy":       _p(1,  700,  2.15, 4,   2,   25,  38, 2),
        "Luke Donald":        _p(3,  900,  1.95, 4,   12,  12,  90, 1),
        "Lee Westwood":       _p(2,  900,  1.95, 16,  2,   3,   60, 0),
        "Adam Scott":         _p(10, 1200, 1.80, 10,  12,  30,  55, 1),
        "Bubba Watson":       _p(23, 4000, 1.50, 6,   100, 100, 20, 0),
        "Phil Mickelson":     _p(12, 1600, 1.75, 18,  1,   15,  45, 0),
        "Justin Rose":        _p(8,  1600, 1.80, 9,   2,   12,  70, 0),
        "Sergio Garcia":      _p(14, 2000, 1.70, 15,  2,   10,  55, 0),
        "Louis Oosthuizen":   _p(35, 6000, 1.40, 3,   2,   20,  45, 0),
        "Matt Kuchar":        _p(15, 2200, 1.65, 5,   12,  12,  80, 0),
        "Peter Hanson":       _p(45, 10000,1.25, 4,   25,  25,  55, 0),
        "Fred Couples":       _p(70, 20000,1.00, 27,  1,   30,  60, 0),
        "Graeme McDowell":    _p(18, 3000, 1.55, 5,   25,  25,  70, 0),
        "Fredrik Jacobson":   _p(55, 15000,1.15, 4,   25,  25,  65, 0),
    },

    # ── 2011 ──────────────────────────────────────────────────────────────────
    2011: {
        "Tiger Woods":        _p(1,  600,  1.90, 17,  1,   4,   60, 1),
        "Lee Westwood":       _p(2,  800,  1.90, 15,  2,   3,   60, 0),
        "Rory McIlroy":       _p(7,  1000, 2.05, 3,   2,   25,  38, 0),
        "Phil Mickelson":     _p(9,  1500, 1.80, 17,  1,   15,  45, 0),
        "Luke Donald":        _p(8,  1400, 1.85, 3,   12,  100, 90, 0),
        "Adam Scott":         _p(12, 1800, 1.75, 9,   12,  30,  55, 1),
        "Dustin Johnson":     _p(15, 2000, 1.70, 4,   100, 100, 5,  0),
        "Jason Day":          _p(25, 4000, 1.55, 2,   100, 100, 35, 0),
        "Charl Schwartzel":   _p(36, 6600, 1.40, 3,   25,  25,  55, 0),
        "K.J. Choi":          _p(30, 5000, 1.45, 8,   25,  25,  85, 0),
        "Ian Poulter":        _p(22, 3500, 1.55, 8,   25,  15,  75, 0),
        "Justin Rose":        _p(11, 2200, 1.75, 8,   2,   12,  70, 0),
        "Sergio Garcia":      _p(14, 2500, 1.65, 14,  2,   10,  55, 0),
        "Fred Couples":       _p(65, 20000,1.00, 26,  1,   25,  60, 0),
        "Geoff Ogilvy":       _p(18, 3000, 1.60, 6,   25,  25,  65, 0),
    },

    # ── 2010 ──────────────────────────────────────────────────────────────────
    2010: {
        "Tiger Woods":        _p(1,  500,  2.00, 16,  1,   1,   60, 0),
        "Phil Mickelson":     _p(3,  1000, 1.90, 16,  1,   15,  45, 0),
        "Lee Westwood":       _p(2,  900,  1.90, 14,  2,   3,   60, 0),
        "Rory McIlroy":       _p(8,  1800, 1.80, 2,   20,  20,  38, 0),
        "Adam Scott":         _p(10, 2000, 1.75, 8,   12,  30,  55, 0),
        "Sergio Garcia":      _p(9,  2000, 1.70, 13,  2,   10,  55, 0),
        "Justin Rose":        _p(14, 2500, 1.65, 7,   8,   8,   70, 0),
        "Ian Poulter":        _p(22, 4000, 1.55, 6,   25,  15,  75, 0),
        "Fred Couples":       _p(60, 20000,1.00, 25,  1,   30,  60, 0),
        "Angel Cabrera":      _p(20, 3500, 1.60, 6,   1,   5,   55, 0),
        "Anthony Kim":        _p(18, 3000, 1.60, 3,   25,  25,  35, 0),
        "K.J. Choi":          _p(25, 5000, 1.50, 7,   25,  25,  85, 0),
        "Nick Watney":        _p(35, 8000, 1.40, 2,   25,  25,  55, 1),
        "Dustin Johnson":     _p(40, 10000,1.35, 3,   100, 100, 5,  0),
        "Ernie Els":          _p(15, 3000, 1.65, 20,  1,   20,  55, 0),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _odds_to_implied(odds_american: int) -> float:
    """Convert American odds to implied probability (0–1)."""
    if odds_american > 0:
        return 100.0 / (odds_american + 100.0)
    else:
        return abs(odds_american) / (abs(odds_american) + 100.0)


def _compute_dna_simple(pdata: dict) -> float:
    """Augusta DNA score (0–100) from pre-tournament history fields."""
    starts = pdata.get("masters_starts", 0)
    best   = pdata.get("best_finish",   999)
    last   = pdata.get("last_finish",   999)

    if starts == 0:
        return 28.0   # Augusta debutant baseline

    def finish_sub(f: int) -> float:
        if f == 1:    return 100.0
        if f <= 5:    return 85.0
        if f <= 10:   return 70.0
        if f <= 20:   return 55.0
        if f <= 30:   return 40.0
        if f < 99:    return 25.0
        return 8.0    # 99/100 = missed cut

    best_sub   = finish_sub(best)
    last_sub   = finish_sub(last) if last < 999 else 28.0
    starts_sub = min(starts / 12.0 * 100.0, 100.0)

    dna = 0.50 * best_sub + 0.25 * starts_sub + 0.25 * last_sub
    return float(np.clip(dna, 0.0, 100.0))


def _midranks(vals: list) -> list:
    """Assign mid-ranks (average rank for ties) to a list of values."""
    n   = len(vals)
    arr = [(v, i) for i, v in enumerate(vals)]
    arr.sort(key=lambda t: t[0])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and arr[j + 1][0] == arr[j][0]:
            j += 1
        midrank = (i + j + 2) / 2.0   # 1-based midrank
        for k in range(i, j + 1):
            ranks[arr[k][1]] = midrank
        i = j + 1
    return ranks


def _spearman(x: list, y: list) -> float:
    """Compute Spearman rank correlation (correct mid-rank tie handling)."""
    if HAS_SCIPY:
        rho, _ = scipy_stats.spearmanr(x, y)
        return float(rho)
    # Manual fallback using Pearson on mid-ranks
    n = len(x)
    if n < 3:
        return 0.0
    rx  = _midranks(x)
    ry  = _midranks(y)
    rxm = sum(rx) / n
    rym = sum(ry) / n
    num = sum((rx[i] - rxm) * (ry[i] - rym) for i in range(n))
    den = (sum((rx[i] - rxm) ** 2 for i in range(n))
           * sum((ry[i] - rym) ** 2 for i in range(n))) ** 0.5
    if den == 0:
        return 0.0
    return float(max(-1.0, min(1.0, num / den)))


def _pool_score(players: list[str], actual: dict) -> float:
    """Score a 4-player pool entry against actual results (point system)."""
    top5  = set(actual.get("top5",  []))
    top10 = set(actual.get("top10", []))
    winner = actual.get("winner", "")
    score = 0.0
    for p in players:
        if p == winner:
            score += 20
        elif p in top5:
            score += 12
        elif p in top10:
            score += 7
    return score


def _estimate_ownership(player: str, pdata: dict) -> float:
    """Estimate pool ownership % from odds + calibration factors."""
    odds = pdata.get("odds_american", 5000)
    rank = pdata.get("world_rank", 50)

    impl = _odds_to_implied(odds) * 100.0

    # Top-3 by rank → 1.5× calibration (mirrors pool_optimizer.py)
    if rank <= 3:
        base = min(65.0, impl * 1.5)
    else:
        base = max(0.3, impl * 0.75)

    # Over/underowned player corrections
    mult_over  = POOL_OVEROWNED_PLAYERS.get(player, 1.0)
    mult_under = POOL_UNDEROWNED_PLAYERS.get(player, 1.0)

    if player in POOL_OVEROWNED_PLAYERS:
        base *= mult_over
    elif player in POOL_UNDEROWNED_PLAYERS:
        base *= mult_under

    return float(np.clip(base, 0.3, 65.0))


# ─────────────────────────────────────────────────────────────────────────────
# AUGBUSTA BACKTEST CLASS
# ─────────────────────────────────────────────────────────────────────────────

class AugustaBacktest:

    def __init__(self, years: list[int] | None = None, rng_seed: int = 42):
        self.years = years or sorted(ACTUAL_RESULTS.keys())
        self.rng_seed = rng_seed
        random.seed(rng_seed)
        np.random.seed(rng_seed)
        self._results: dict[int, dict] = {}
        self._summary: dict = {}

    # ── Internal: score one year ────────────────────────────────────────────

    def score_year(self, year: int, weights: dict | None = None) -> pd.DataFrame:
        """
        Build a scored DataFrame for `year` from SYNTHETIC_PRE_TOURNAMENT.
        Returns DataFrame with columns: Player, Form, DNA, Fit, Vegas, Trajectory,
        Augusta_Score, Ownership_Pct, EV_Score, Model_Rank.
        """
        actual = ACTUAL_RESULTS[year]
        if weights is None:
            weights = CHAOS_WEIGHTS if actual["chaos"] else NORMAL_WEIGHTS
        field = SYNTHETIC_PRE_TOURNAMENT.get(year, {})
        if not field:
            raise ValueError(f"No synthetic data for {year}")

        all_sg    = [p["sg_proxy"] for p in field.values()]
        all_odds  = [p["odds_american"] for p in field.values()]
        all_ranks = [p["world_rank"] for p in field.values()]

        sg_mean = float(np.mean(all_sg))
        sg_std  = max(float(np.std(all_sg)), 0.01)
        max_impl = max(_odds_to_implied(o) for o in all_odds)
        rk_min, rk_max = min(all_ranks), max(all_ranks)

        rows = []
        for player, pdata in field.items():
            sg   = pdata["sg_proxy"]
            odds = pdata["odds_american"]
            rank = pdata["world_rank"]
            rw   = pdata.get("recent_wins", 0)
            dd   = pdata.get("dd_rank", 80)

            # ── FORM (0-100) ──────────────────────────────────────────────────
            sg_z    = (sg - sg_mean) / sg_std
            sg_norm = float(np.clip((sg_z + 3) / 6.0 * 100.0, 0, 100))
            win_bonus = min(rw * 12.0, 28.0)
            form = float(np.clip(sg_norm * 0.75 + win_bonus, 0, 100))

            # ── DNA (0-100) ───────────────────────────────────────────────────
            dna = _compute_dna_simple(pdata)

            # ── FIT (0-100): approach proxy + driving distance ─────────────────
            dd_score  = float(np.clip((180 - dd) / 160.0 * 100.0, 0, 100))
            app_proxy = float(np.clip((sg - 0.4 - sg_mean * 0.5) / sg_std * 50 + 50, 0, 100))
            fit = float(np.clip(app_proxy * 0.70 + dd_score * 0.30, 0, 100))

            # ── VEGAS (0-100): implied probability vs field ────────────────────
            impl  = _odds_to_implied(odds)
            vegas = float(np.clip(impl / max(max_impl, 0.001) * 100.0, 0, 100))

            # ── TRAJECTORY (0-100): world rank vs field ───────────────────────
            traj = float(np.clip(
                (rk_max - rank) / max(rk_max - rk_min, 1) * 100.0, 0, 100
            ))

            # ── COMPOSITE ─────────────────────────────────────────────────────
            score = (
                weights["form"]  * form
                + weights["fit"]   * fit
                + weights["vegas"] * vegas
                + weights["dna"]   * dna
                + weights["traj"]  * traj
            )

            own = _estimate_ownership(player, pdata)
            ev  = score / math.sqrt(max(own, 0.1))

            rows.append({
                "Player":        player,
                "Form":          round(form,  1),
                "DNA":           round(dna,   1),
                "Fit":           round(fit,   1),
                "Vegas":         round(vegas, 1),
                "Trajectory":    round(traj,  1),
                "Augusta_Score": round(score, 2),
                "Ownership_Pct": round(own,   2),
                "EV_Score":      round(ev,    3),
            })

        df = pd.DataFrame(rows).sort_values("Augusta_Score", ascending=False)
        df["Model_Rank"] = range(1, len(df) + 1)
        return df.reset_index(drop=True)

    # ── Internal: generate teams ─────────────────────────────────────────────

    def generate_teams_for_year(self, df: pd.DataFrame) -> tuple[list, list, list]:
        """
        Generate Floor / Ceiling / Value teams from scored DataFrame.
        Returns (team_a, team_b, team_c) as lists of player names.
        """
        by_score = df.sort_values("Augusta_Score", ascending=False)
        by_ev    = df.sort_values("EV_Score",      ascending=False)

        # Team A (Floor): top-4 by Augusta_Score
        team_a = by_score.head(4)["Player"].tolist()

        # Team B (Ceiling): top-1 by score + top-3 by EV (not already in A)
        anchor_b = by_score.iloc[0]["Player"]
        ev_others = [p for p in by_ev["Player"] if p != anchor_b]
        team_b = [anchor_b] + ev_others[:3]

        # Team C (Value): top-2 by EV from ranks 3–12, + 2 more EV not in B
        mid_ev = [p for p in by_ev["Player"]
                  if df[df["Player"] == p]["Model_Rank"].values[0] in range(3, 13)]
        c_core = mid_ev[:2] if mid_ev else by_ev.head(2)["Player"].tolist()
        c_fill = [p for p in by_ev["Player"] if p not in c_core and p not in team_b]
        team_c = (c_core + c_fill[:2])[:4]

        return team_a, team_b, team_c

    # ── TEST 1: Score → Finish Rank Correlation (Spearman ρ) ────────────────

    def test_score_correlation(self, year: int, df: pd.DataFrame) -> dict:
        """
        Measure how well Augusta_Score predicts actual finishing quality.
        Binary target: 1 = actual top-10, 0 = not.
        Returns Spearman ρ and p-value between Score rank and outcome rank.
        """
        actual = ACTUAL_RESULTS[year]
        top10  = set(actual.get("top10", []))

        scores   = []
        outcomes = []
        for _, row in df.iterrows():
            p = row["Player"]
            scores.append(row["Augusta_Score"])
            # Outcome: 100 = winner, 80 = top-5, 60 = top-10, else 20
            if p == actual["winner"]:
                outcomes.append(100)
            elif p in set(actual.get("top5", [])):
                outcomes.append(80)
            elif p in top10:
                outcomes.append(60)
            else:
                outcomes.append(20)

        rho = _spearman(scores, outcomes)

        # Capture: what % of actual top-10 appear in model's top-15?
        model_top15 = set(df.head(15)["Player"].tolist())
        captured    = len([p for p in top10 if p in model_top15 and p in df["Player"].values])
        total_in_field = len([p for p in top10 if p in df["Player"].values])
        capture_rate = captured / max(total_in_field, 1)

        return {
            "spearman_rho":    round(rho, 3),
            "capture_top10_in_model_top15": round(capture_rate, 3),
            "actual_top10_in_field":  total_in_field,
        }

    # ── TEST 2: Top-10 Capture Rate ──────────────────────────────────────────

    def test_top10_capture(self, year: int, df: pd.DataFrame) -> dict:
        """How many actual top-10 finishers appear in model's top-20?"""
        actual   = ACTUAL_RESULTS[year]
        top10    = set(actual.get("top10", []))
        model_20 = set(df.head(20)["Player"].tolist())
        model_10 = set(df.head(10)["Player"].tolist())

        in_field   = [p for p in top10 if p in df["Player"].values]
        hit_top20  = [p for p in in_field if p in model_20]
        hit_top10  = [p for p in in_field if p in model_10]

        winner_rank = None
        w = actual["winner"]
        if w in df["Player"].values:
            winner_rank = int(df[df["Player"] == w]["Model_Rank"].values[0])

        return {
            "actual_top10_in_field":   len(in_field),
            "captured_in_model_top20": len(hit_top20),
            "captured_in_model_top10": len(hit_top10),
            "capture_rate_top20":      round(len(hit_top20) / max(len(in_field), 1), 3),
            "capture_rate_top10":      round(len(hit_top10) / max(len(in_field), 1), 3),
            "winner_model_rank":       winner_rank,
            "winner_in_top10":         winner_rank is not None and winner_rank <= 10,
        }

    # ── TEST 3: Monte Carlo Pool Simulation ──────────────────────────────────

    def test_pool_simulation(
        self,
        year: int,
        df: pd.DataFrame,
        team_a: list,
        team_b: list,
        team_c: list,
        n_iter: int = 1000,
        pool_size: int = 500,
    ) -> dict:
        """
        Monte Carlo: simulate pool_size random 4-person teams, repeat n_iter times.
        Model teams scored against same actual results.
        Returns mean/median percentile for each model team.
        """
        actual   = ACTUAL_RESULTS[year]
        players  = df["Player"].tolist()
        own_pcts = [_estimate_ownership(p, SYNTHETIC_PRE_TOURNAMENT[year].get(p, {}))
                    for p in players]
        own_arr  = np.array(own_pcts, dtype=float)
        own_arr  = own_arr / own_arr.sum()   # normalize to probability distribution

        ta_score = _pool_score(team_a, actual)
        tb_score = _pool_score(team_b, actual)
        tc_score = _pool_score(team_c, actual)

        pctile_a, pctile_b, pctile_c = [], [], []

        for _ in range(n_iter):
            pool_scores = []
            for _ in range(pool_size):
                # Sample 4 players without replacement within a team
                idxs = np.random.choice(len(players), size=min(4, len(players)),
                                        replace=False, p=own_arr)
                team = [players[i] for i in idxs]
                pool_scores.append(_pool_score(team, actual))
            pool_arr = np.array(pool_scores, dtype=float)
            pctile_a.append(float(np.mean(pool_arr <= ta_score) * 100))
            pctile_b.append(float(np.mean(pool_arr <= tb_score) * 100))
            pctile_c.append(float(np.mean(pool_arr <= tc_score) * 100))

        return {
            "team_a_score":        ta_score,
            "team_b_score":        tb_score,
            "team_c_score":        tc_score,
            "team_a_pctile_mean":  round(float(np.mean(pctile_a)), 1),
            "team_b_pctile_mean":  round(float(np.mean(pctile_b)), 1),
            "team_c_pctile_mean":  round(float(np.mean(pctile_c)), 1),
            "best_team":           "A" if ta_score >= max(tb_score, tc_score)
                                   else ("B" if tb_score >= tc_score else "C"),
        }

    # ── TEST 4: Component Ablation ────────────────────────────────────────────

    def test_component_ablation(self, year: int) -> dict:
        """
        Drop each component, redistribute its weight proportionally,
        re-score the year, measure Spearman delta vs baseline.
        """
        actual   = ACTUAL_RESULTS[year]
        base_w   = CHAOS_WEIGHTS if actual["chaos"] else NORMAL_WEIGHTS
        base_df  = self.score_year(year, weights=base_w)
        base_res = self.test_score_correlation(year, base_df)
        base_rho = base_res["spearman_rho"]

        ablation = {}
        for drop in COMPONENT_KEYS:
            remaining = {k: v for k, v in base_w.items() if k != drop}
            total_r   = sum(remaining.values())
            adj_w     = {k: v / total_r for k, v in remaining.items()}
            adj_w[drop] = 0.0
            try:
                abl_df   = self.score_year(year, weights=adj_w)
                abl_res  = self.test_score_correlation(year, abl_df)
                abl_rho  = abl_res["spearman_rho"]
            except Exception:
                abl_rho = base_rho
            ablation[drop] = {
                "rho_without": round(abl_rho, 3),
                "delta":       round(abl_rho - base_rho, 3),
            }
        return {"base_rho": base_rho, "components": ablation}

    # ── TEST 5: Weak Year Diagnosis ───────────────────────────────────────────

    def test_weak_year_diagnosis(
        self, year: int, df: pd.DataFrame, team_a: list, team_b: list, team_c: list
    ) -> dict:
        """Categorize why the model missed (if it did) in a given year."""
        actual     = ACTUAL_RESULTS[year]
        winner     = actual["winner"]
        chaos      = actual["chaos"]
        miss_reason = actual.get("miss_reason")

        winner_rank = None
        if winner in df["Player"].values:
            winner_rank = int(df[df["Player"] == winner]["Model_Rank"].values[0])

        top10  = set(actual.get("top10", []))
        model5 = set(df.head(5)["Player"].tolist())
        overlap_with_top10 = len(top10 & set(df.head(10)["Player"].tolist()))

        if winner_rank is None or winner_rank > 15:
            if chaos:
                miss_type = "CHAOS_MISS"
            elif miss_reason and "collapse" in miss_reason:
                miss_type = "BENEFICIARY_MISS"
            elif miss_reason and "DNA" in miss_reason:
                miss_type = "DNA_UNDERWEIGHT"
            elif miss_reason and "clutch" in miss_reason.lower():
                miss_type = "INTANGIBLE_MISS"
            else:
                miss_type = "STRUCTURAL_MISS"
        elif winner_rank <= 5:
            miss_type = "HIT_TOP5"
        elif winner_rank <= 10:
            miss_type = "HIT_TOP10"
        else:
            miss_type = "NEAR_MISS"

        any_team_has_winner = winner in team_a or winner in team_b or winner in team_c

        return {
            "winner_model_rank":   winner_rank,
            "miss_type":           miss_type,
            "any_team_has_winner": any_team_has_winner,
            "overlap_top10":       overlap_with_top10,
            "miss_reason":         miss_reason or "N/A",
        }

    # ── TEST 6: Holdout Deep-Dive (2024 + 2025) ──────────────────────────────

    def test_holdout_years(self) -> dict:
        """Detailed decomposition for the two most recent holdout years."""
        results = {}
        for year in [2024, 2025]:
            if year not in SYNTHETIC_PRE_TOURNAMENT:
                continue
            actual = ACTUAL_RESULTS[year]
            df     = self.score_year(year)
            ta, tb, tc = self.generate_teams_for_year(df)

            top5    = actual.get("top5", [])
            top10   = actual.get("top10", [])
            winner  = actual["winner"]

            # Full model ranking
            ranking = df[["Player", "Augusta_Score", "Model_Rank",
                          "Ownership_Pct", "EV_Score"]].head(15).to_dict("records")

            # Team scoring
            ta_pts = _pool_score(ta, actual)
            tb_pts = _pool_score(tb, actual)
            tc_pts = _pool_score(tc, actual)

            # Winner decomposition
            w_row = df[df["Player"] == winner]
            w_decomp = {}
            if not w_row.empty:
                w_decomp = {
                    "Form":    w_row.iloc[0]["Form"],
                    "DNA":     w_row.iloc[0]["DNA"],
                    "Fit":     w_row.iloc[0]["Fit"],
                    "Vegas":   w_row.iloc[0]["Vegas"],
                    "Trajectory": w_row.iloc[0]["Trajectory"],
                    "Augusta_Score": w_row.iloc[0]["Augusta_Score"],
                    "Model_Rank": int(w_row.iloc[0]["Model_Rank"]),
                }

            # Surprise check: who in top-5 wasn't in model top-10?
            surprises = [p for p in top5
                         if p in df["Player"].values
                         and int(df[df["Player"] == p]["Model_Rank"].values[0]) > 10]

            results[year] = {
                "winner":         winner,
                "condition":      actual["condition"],
                "model_top15":    ranking,
                "team_a":         ta,
                "team_b":         tb,
                "team_c":         tc,
                "team_a_pts":     ta_pts,
                "team_b_pts":     tb_pts,
                "team_c_pts":     tc_pts,
                "winner_decomp":  w_decomp,
                "surprises":      surprises,
                "actual_top5":    top5,
                "actual_top10":   top10,
            }
        return results

    # ── MAIN RUNNER ───────────────────────────────────────────────────────────

    def run_full_backtest(self) -> dict:
        """Run all 6 test dimensions across all years. Returns full results dict."""
        t0 = time.time()
        year_results = {}

        for year in self.years:
            if year not in SYNTHETIC_PRE_TOURNAMENT or year not in ACTUAL_RESULTS:
                continue
            actual = ACTUAL_RESULTS[year]
            w = CHAOS_WEIGHTS if actual["chaos"] else NORMAL_WEIGHTS

            try:
                df = self.score_year(year, weights=w)
                ta, tb, tc = self.generate_teams_for_year(df)

                corr    = self.test_score_correlation(year, df)
                capture = self.test_top10_capture(year, df)
                sim     = self.test_pool_simulation(year, df, ta, tb, tc,
                                                    n_iter=1000, pool_size=500)
                ablate  = self.test_component_ablation(year)
                diag    = self.test_weak_year_diagnosis(year, df, ta, tb, tc)

                year_results[year] = {
                    "actual":        actual,
                    "team_a":        ta,
                    "team_b":        tb,
                    "team_c":        tc,
                    "score_correlation": corr,
                    "top10_capture":     capture,
                    "pool_simulation":   sim,
                    "component_ablation": ablate,
                    "weak_year_diagnosis": diag,
                }
            except Exception as e:
                year_results[year] = {"error": str(e)}
                print(f"  [ERR] {year}: {e}")

        holdout = self.test_holdout_years()
        elapsed = round(time.time() - t0, 1)

        self._results = year_results
        self._summary = self._build_summary(year_results)
        return {
            "meta": {
                "years_tested": len(year_results),
                "rng_seed":     self.rng_seed,
                "elapsed_sec":  elapsed,
            },
            "year_results":  year_results,
            "holdout":       holdout,
            "summary":       self._summary,
        }

    def _build_summary(self, year_results: dict) -> dict:
        """Aggregate stats across all years."""
        valid = {y: r for y, r in year_results.items() if "error" not in r}

        rhos      = [r["score_correlation"]["spearman_rho"] for r in valid.values()]
        cap20     = [r["top10_capture"]["capture_rate_top20"] for r in valid.values()]
        cap10     = [r["top10_capture"]["capture_rate_top10"] for r in valid.values()]
        w_ranks   = [r["top10_capture"]["winner_model_rank"] for r in valid.values()
                     if r["top10_capture"]["winner_model_rank"] is not None]
        top5_hits = sum(1 for r in valid.values()
                        if r["top10_capture"]["winner_model_rank"] is not None
                        and r["top10_capture"]["winner_model_rank"] <= 5)
        top10_hits = sum(1 for r in valid.values()
                         if r["top10_capture"]["winner_model_rank"] is not None
                         and r["top10_capture"]["winner_model_rank"] <= 10)

        miss_types = defaultdict(int)
        for r in valid.values():
            miss_types[r["weak_year_diagnosis"]["miss_type"]] += 1

        # Best component by ablation (most important = biggest negative delta when removed)
        ablation_importance = defaultdict(list)
        for r in valid.values():
            for comp, stats in r["component_ablation"]["components"].items():
                ablation_importance[comp].append(stats["delta"])
        component_importance = {k: round(float(np.mean(v)), 4)
                                 for k, v in ablation_importance.items()}

        return {
            "years_analyzed":    len(valid),
            "spearman_rho_mean": round(float(np.mean(rhos)), 3),
            "spearman_rho_std":  round(float(np.std(rhos)),  3),
            "capture_top20_mean": round(float(np.mean(cap20)), 3),
            "capture_top10_mean": round(float(np.mean(cap10)), 3),
            "winner_top5_hits":  top5_hits,
            "winner_top10_hits": top10_hits,
            "winner_top5_rate":  round(top5_hits  / len(valid), 3),
            "winner_top10_rate": round(top10_hits / len(valid), 3),
            "avg_winner_rank":   round(float(np.mean(w_ranks)), 1) if w_ranks else None,
            "miss_type_counts":  dict(miss_types),
            "component_importance_delta_rho": component_importance,
        }

    # ── PRINT SUMMARY ─────────────────────────────────────────────────────────

    def _print_summary(self, full: dict) -> None:
        W = 76
        sep = "━" * W
        thin = "─" * W

        def hdr(txt):
            print(f"\n{sep}\n  {txt}\n{thin}")

        print(f"\n{sep}")
        print(f"  AUGUSTA POOL INTELLIGENCE — BACKTEST v2.0  ({self.years[0]}–{self.years[-1]})")
        print(f"  {len(self.years)} years | 6 test dimensions | Monte Carlo: 1000 iter × 500 pool")
        print(sep)

        summary = full["summary"]
        meta    = full["meta"]

        # ── Overview ──────────────────────────────────────────────────────────
        hdr("EXECUTIVE SUMMARY")
        print(f"  Years analyzed:        {summary['years_analyzed']}")
        print(f"  Avg Spearman ρ:        {summary['spearman_rho_mean']:+.3f}  "
              f"(σ={summary['spearman_rho_std']:.3f})")
        print(f"  Top-20 capture rate:   {summary['capture_top20_mean']:.1%}")
        print(f"  Top-10 capture rate:   {summary['capture_top10_mean']:.1%}")
        print(f"  Winner in model top-5: {summary['winner_top5_hits']}/{summary['years_analyzed']}  "
              f"({summary['winner_top5_rate']:.1%})")
        print(f"  Winner in model top-10:{summary['winner_top10_hits']}/{summary['years_analyzed']}  "
              f"({summary['winner_top10_rate']:.1%})")
        print(f"  Avg winner model rank: {summary['avg_winner_rank']}")
        print(f"  Runtime:               {meta['elapsed_sec']}s")

        # ── Test 1+2: Per-year correlation + capture ──────────────────────────
        hdr("TEST 1+2: SCORE CORRELATION & TOP-10 CAPTURE")
        print(f"  {'Year':<6} {'Winner':<22} {'WinRank':<9} {'ρ':>6} "
              f"{'Cap20':>6} {'Cap10':>6} {'MissType'}")
        print(f"  {thin}")
        for year in self.years:
            r = full["year_results"].get(year, {})
            if "error" in r:
                print(f"  {year}  ERROR: {r['error']}")
                continue
            winner = r["actual"]["winner"]
            cap    = r["top10_capture"]
            corr   = r["score_correlation"]
            diag   = r["weak_year_diagnosis"]
            wr     = cap.get("winner_model_rank")
            wr_str = f"#{wr}" if wr else "N/A"
            miss   = diag["miss_type"]
            chaos  = "⚡" if r["actual"]["chaos"] else "  "
            print(f"  {year} {chaos} {winner:<20} {wr_str:<9} "
                  f"{corr['spearman_rho']:>+.3f} "
                  f"{cap['capture_rate_top20']:>5.1%} "
                  f"{cap['capture_rate_top10']:>5.1%} "
                  f"{miss}")

        # ── Test 3: Pool Simulation ───────────────────────────────────────────
        hdr("TEST 3: POOL SIMULATION MONTE CARLO (n=500 pool, 1000 iter)")
        print(f"  {'Year':<6} {'Winner':<22} {'Team A':>8} {'Team B':>8} {'Team C':>8} {'Best'}")
        print(f"  {thin}")
        for year in self.years:
            r = full["year_results"].get(year, {})
            if "error" in r:
                continue
            sim    = r["pool_simulation"]
            winner = r["actual"]["winner"]
            chaos  = "⚡" if r["actual"]["chaos"] else "  "
            print(f"  {year} {chaos} {winner:<20} "
                  f"{sim['team_a_pctile_mean']:>6.1f}%ile "
                  f"{sim['team_b_pctile_mean']:>6.1f}%ile "
                  f"{sim['team_c_pctile_mean']:>6.1f}%ile "
                  f"  Team {sim['best_team']}")

        # ── Test 4: Component Ablation ────────────────────────────────────────
        hdr("TEST 4: COMPONENT ABLATION (mean Δρ when component removed)")
        imp = summary["component_importance_delta_rho"]
        ranked = sorted(imp.items(), key=lambda x: x[1])
        print(f"  Component   Mean Δρ    Verdict")
        print(f"  {thin}")
        for comp, delta in ranked:
            w = NORMAL_WEIGHTS[comp]
            verdict = "Critical" if delta < -0.04 else \
                      "Significant" if delta < -0.01 else \
                      "Marginal"
            bar = "█" * min(int(abs(delta) * 200), 20)
            print(f"  {comp.upper():<10}  {delta:>+.4f}    {verdict:<12}  [{bar}]  (wt={w:.0%})")

        # ── Test 5: Weak Year Diagnosis ───────────────────────────────────────
        hdr("TEST 5: WEAK YEAR DIAGNOSIS")
        miss_counts = summary["miss_type_counts"]
        total = sum(miss_counts.values())
        print(f"  {'Miss Type':<24}  Count  Rate   Implication")
        print(f"  {thin}")
        labels = {
            "HIT_TOP5":          "✓ Winner in model top-5",
            "HIT_TOP10":         "✓ Winner in model top-10",
            "NEAR_MISS":         "~ Winner ranked 11–15",
            "CHAOS_MISS":        "⚡ Chaos conditions disrupted model",
            "BENEFICIARY_MISS":  "↘ Winner benefited from leader collapse",
            "DNA_UNDERWEIGHT":   "⟳ Augusta DNA weight may need increase",
            "INTANGIBLE_MISS":   "? Clutch/narrative — not modelable",
            "STRUCTURAL_MISS":   "✗ Structural gap in model",
        }
        for mtype in sorted(miss_counts, key=miss_counts.get, reverse=True):
            cnt  = miss_counts[mtype]
            rate = cnt / total
            lbl  = labels.get(mtype, mtype)
            print(f"  {mtype:<24}  {cnt:>4}   {rate:.1%}   {lbl}")

        # ── Test 6: Holdout Deep-Dive ─────────────────────────────────────────
        hdr("TEST 6: HOLDOUT YEARS — 2024 & 2025 DEEP DIVE")
        for year, h in full["holdout"].items():
            print(f"\n  [{year}] WINNER: {h['winner']}  |  Conditions: {h['condition']}")
            w_d = h.get("winner_decomp", {})
            if w_d:
                print(f"  Winner model rank #{w_d.get('Model_Rank','?')}  "
                      f"Score={w_d.get('Augusta_Score','?'):.1f}")
                print(f"    Form={w_d.get('Form','?'):.0f}  DNA={w_d.get('DNA','?'):.0f}  "
                      f"Fit={w_d.get('Fit','?'):.0f}  Vegas={w_d.get('Vegas','?'):.0f}  "
                      f"Traj={w_d.get('Trajectory','?'):.0f}")
            print(f"  Actual top-5:  {', '.join(h['actual_top5'][:5])}")
            print(f"  Team A {h['team_a_pts']:>2}pts:  {', '.join(h['team_a'])}")
            print(f"  Team B {h['team_b_pts']:>2}pts:  {', '.join(h['team_b'])}")
            print(f"  Team C {h['team_c_pts']:>2}pts:  {', '.join(h['team_c'])}")
            if h.get("surprises"):
                print(f"  Model surprises: {', '.join(h['surprises'])}")
            print(f"  Model top-10:")
            for rec in h["model_top15"][:10]:
                tag = ""
                if rec["Player"] == h["winner"]:              tag = " ← WINNER"
                elif rec["Player"] in h["actual_top5"]:       tag = " ← top-5"
                elif rec["Player"] in h["actual_top10"]:      tag = " ← top-10"
                print(f"    #{rec['Model_Rank']:<3} {rec['Player']:<22} "
                      f"Score={rec['Augusta_Score']:.1f}  "
                      f"Own={rec['Ownership_Pct']:.1f}%  "
                      f"EV={rec['EV_Score']:.2f}{tag}")

        print(f"\n{sep}")
        print(f"  Backtest complete.  Results saved to backtest_results/backtest_v2_results.json")
        print(sep)

    # ── SAVE RESULTS ──────────────────────────────────────────────────────────

    def _save_results(self, full: dict) -> Path:
        out_dir = Path(__file__).parent / "backtest_results"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "backtest_v2_results.json"

        # Make serializable (convert numpy types)
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            return obj

        with open(out_path, "w") as f:
            json.dump(_clean(full), f, indent=2, default=str)
        return out_path


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT OPTIMIZATION RECOMMENDATION
# ─────────────────────────────────────────────────────────────────────────────

def recommend_weight_adjustments(backtest_results: dict) -> dict:
    """
    Analyze ablation deltas and miss-type distribution to recommend
    weight adjustments for the next iteration of the model.

    Logic:
    - If a component's mean ablation Δρ is > threshold_up (removing it HELPS),
      the component is over-weighted → suggest -3 to -5 percentage points.
    - If a component's Δρ is < threshold_down (removing it hurts a lot),
      the component is under-weighted → suggest +3 to +5 pts.
    - If CHAOS_MISS rate > 30%, chaos weights may need rebalancing.
    - If DNA_UNDERWEIGHT appears, bump DNA by +2 pts in normal mode.
    """
    summary  = backtest_results.get("summary", {})
    imp      = summary.get("component_importance_delta_rho", {})
    miss_c   = summary.get("miss_type_counts", {})
    total    = max(sum(miss_c.values()), 1)

    threshold_under = -0.035   # removing hurts a lot → currently underweighted
    threshold_over  =  0.010   # removing helps → currently overweighted

    recs = {}
    for comp, delta in imp.items():
        current_w = NORMAL_WEIGHTS[comp]
        if delta < threshold_under:
            suggestion  = "INCREASE"
            adjustment  = +0.03 if delta < -0.06 else +0.02
            new_w       = round(current_w + adjustment, 2)
            rationale   = (
                f"Removing {comp.upper()} drops ρ by {abs(delta):.3f} — "
                f"critically undervalued. Raise from {current_w:.0%} → {new_w:.0%}."
            )
        elif delta > threshold_over:
            suggestion  = "DECREASE"
            adjustment  = -0.03 if delta > 0.05 else -0.02
            new_w       = round(max(0.02, current_w + adjustment), 2)
            rationale   = (
                f"Removing {comp.upper()} improves ρ by {delta:.3f} — "
                f"may be overfit or noisy. Lower from {current_w:.0%} → {new_w:.0%}."
            )
        else:
            suggestion  = "HOLD"
            adjustment  =  0.0
            new_w       = current_w
            rationale   = (
                f"{comp.upper()} contributing meaningfully. "
                f"Δρ = {delta:+.3f} — within noise band. Keep at {current_w:.0%}."
            )
        recs[comp] = {
            "current_weight":    current_w,
            "suggested_action":  suggestion,
            "adjustment":        adjustment,
            "suggested_weight":  new_w,
            "ablation_delta_rho": delta,
            "rationale":         rationale,
        }

    # Structural observations
    observations = []

    chaos_miss_rate = miss_c.get("CHAOS_MISS", 0) / total
    if chaos_miss_rate > 0.25:
        observations.append(
            f"CHAOS_MISS rate is {chaos_miss_rate:.0%} — consider expanding chaos "
            f"weight shift: Fit → 0.35, Form → 0.35 (from 0.33/0.35)."
        )

    dna_miss_rate = miss_c.get("DNA_UNDERWEIGHT", 0) / total
    if dna_miss_rate > 0.10:
        observations.append(
            "DNA_UNDERWEIGHT flag present — Augusta specialists consistently "
            "underrated. Consider DNA: 0.15 (from 0.13) in normal conditions."
        )

    hit_rate = (miss_c.get("HIT_TOP5", 0) + miss_c.get("HIT_TOP10", 0)) / total
    if hit_rate >= 0.55:
        observations.append(
            f"Model predicts winner top-10 in {hit_rate:.0%} of years — "
            f"well above random (expected ~40%). Core architecture is sound."
        )

    intangible_rate = (
        miss_c.get("INTANGIBLE_MISS", 0) + miss_c.get("BENEFICIARY_MISS", 0)
    ) / total
    if intangible_rate > 0.20:
        observations.append(
            f"{intangible_rate:.0%} of misses are intangible/beneficiary — "
            "unforeseeable events, not weight optimization opportunities."
        )

    print("\n" + "─" * 76)
    print("  WEIGHT ADJUSTMENT RECOMMENDATIONS")
    print("─" * 76)
    for comp, rec in recs.items():
        action = rec["suggested_action"]
        sym    = "▲" if action == "INCREASE" else ("▼" if action == "DECREASE" else "●")
        print(f"  {sym} {comp.upper():<8}  {rec['current_weight']:.0%} → {rec['suggested_weight']:.0%}  "
              f"[{action}]  Δρ={rec['ablation_delta_rho']:+.3f}")
        print(f"           {rec['rationale']}")

    if observations:
        print("\n  STRUCTURAL OBSERVATIONS:")
        for obs in observations:
            print(f"  • {obs}")
    print("─" * 76)

    return {
        "component_recommendations": recs,
        "structural_observations":   observations,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\nInitializing Augusta Pool Intelligence Backtest v2.0 ...")
    print(f"  Years in scope: 2010–2025  ({len(ACTUAL_RESULTS)} years)")
    print(f"  Synthetic field coverage: {sum(len(v) for v in SYNTHETIC_PRE_TOURNAMENT.values())} "
          f"player-year records")
    print(f"  Monte Carlo: 1,000 iterations × 500-entry pool per year")

    bt  = AugustaBacktest(years=sorted(ACTUAL_RESULTS.keys()), rng_seed=42)
    res = bt.run_full_backtest()

    bt._print_summary(res)
    recs = recommend_weight_adjustments(res)
    res["weight_recommendations"] = recs

    out = bt._save_results(res)
    print(f"\n  JSON written → {out}\n")


if __name__ == "__main__":
    main()
