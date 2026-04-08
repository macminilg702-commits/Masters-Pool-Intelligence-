"""
fetch_data.py — All data fetching, caching, and fallback logic.
Caches locally to /data/ as JSON. Cache TTL = 6 hours.
"""
from __future__ import annotations

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path

import requests
from field_data import (
    ADDITIONAL_PLAYER_STATS,
    UPDATED_ODDS_EXISTING,
    ADDITIONAL_MASTERS_HISTORY,
    ADDITIONAL_NAME_ABBREV,
    ODDS_NAME_OVERRIDES,
)

# ── API key loader — works locally (os.getenv / .env) and on Streamlit Cloud ──
def _get_secret(key_name: str, default: str = "") -> str:
    """Try Streamlit secrets first; fall back to environment variable."""
    try:
        import streamlit as st
        return st.secrets[key_name]
    except Exception:
        pass
    # Local fallback: try loading .env then os.getenv
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return os.getenv(key_name, default)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CACHE_TTL_HOURS = 6
ODDS_API_KEY = _get_secret("ODDS_API_KEY")

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# CACHE UTILITIES
# ─────────────────────────────────────────────────────────────────

def _cache_path(name: str) -> Path:
    return DATA_DIR / name

def _is_fresh(path: Path, ttl_hours: int = CACHE_TTL_HOURS) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < ttl_hours * 3600

def _load_cache(path: Path):
    with open(path) as f:
        return json.load(f)

def _save_cache(path: Path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def cache_mtime(name: str) -> str:
    path = _cache_path(name)
    if path.exists():
        ts = datetime.fromtimestamp(path.stat().st_mtime)
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return "Never"

# ─────────────────────────────────────────────────────────────────
# AUGUSTA NATIONAL HISTORY 2015–2025 (HARDCODED)
# ─────────────────────────────────────────────────────────────────

AUGUSTA_HISTORY = {
    2025: {
        "winner": "R. McIlroy", "score": -11,
        "top15": [
            ("R. McIlroy", -11), ("J. Rose", -11), ("S. Scheffler", -9),
            ("C. Morikawa", -8), ("C. Young", -7), ("L. Aberg", -7),
            ("B. DeChambeau", -6), ("J. Spieth", -5), ("P. Cantlay", -5),
            ("R. MacIntyre", -4), ("T. Fleetwood", -4), ("J. Thomas", -3),
            ("V. Hovland", -3), ("H. Matsuyama", -2), ("C. Smith", -2),
        ],
        "weather": "soft_wet", "top4_total": -39,
    },
    2024: {
        "winner": "S. Scheffler", "score": -11,
        "top15": [
            ("S. Scheffler", -11), ("L. Aberg", -7), ("C. Morikawa", -4),
            ("T. Fleetwood", -4), ("M. Homa", -4), ("C. Smith", -2),
            ("B. DeChambeau", -1), ("X. Schauffele", -1), ("W. Zalatoris", 0),
            ("T. Hatton", 0), ("C. Young", 2), ("M. Pavon", 2),
            ("A. Schenk", 2), ("P. Reed", 3), ("B. Harman", 3),
        ],
        "weather": "mild", "top4_total": -26,
    },
    2023: {
        "winner": "J. Rahm", "score": -12,
        "top15": [
            ("J. Rahm", -12), ("B. Koepka", -8), ("P. Mickelson", -8),
            ("R. Henley", -7), ("P. Reed", -7), ("J. Spieth", -7),
            ("V. Hovland", -6), ("C. Young", -6), ("S. Theegala", -5),
            ("X. Schauffele", -4), ("C. Morikawa", -4), ("M. Fitzpatrick", -4),
            ("S. Scheffler", -4), ("P. Cantlay", -3), ("G. Woodland", -3),
        ],
        "weather": "mild", "top4_total": -35,
    },
    2022: {
        "winner": "S. Scheffler", "score": -10,
        "top15": [
            ("S. Scheffler", -10), ("R. McIlroy", -7), ("S. Lowry", -5),
            ("C. Smith", -5), ("C. Morikawa", -4), ("W. Zalatoris", -3),
            ("C. Conners", -3), ("S.J. Im", -1), ("J. Thomas", -1),
            ("C. Champ", 0), ("C. Schwartzel", 0), ("D. Johnson", 1),
            ("D. Willett", 1), ("M.W. Lee", 2), ("T. Gooch", 2),
        ],
        "weather": "warm_calm", "top4_total": -27,
    },
    2021: {
        "winner": "H. Matsuyama", "score": -10,
        "top15": [
            ("H. Matsuyama", -10), ("W. Zalatoris", -9), ("X. Schauffele", -7),
            ("J. Spieth", -7), ("J. Rahm", -6), ("M. Leishman", -6),
            ("J. Rose", -5), ("C. Conners", -4), ("P. Reed", -4),
            ("T. Finau", -3), ("C. Smith", -3), ("S. Cink", -2),
            ("B. Harman", -2), ("S.W. Kim", -2), ("W. Simpson", -2),
        ],
        "weather": "fast_firm", "top4_total": -33,
    },
    2020: {
        "winner": "D. Johnson", "score": -20,
        "top15": [
            ("D. Johnson", -20), ("S.J. Im", -15), ("C. Smith", -15),
            ("J. Thomas", -12), ("R. McIlroy", -11), ("D. Frittelli", -11),
            ("J. Rahm", -10), ("C.T. Pan", -10), ("B. Koepka", -10),
            ("C. Conners", -9), ("W. Simpson", -9), ("P. Reed", -9),
            ("A. Ancer", -8), ("K. Na", -8), ("M. Leishman", -8),
        ],
        "weather": "november_soft", "top4_total": -62,
    },
    2019: {
        "winner": "T. Woods", "score": -13,
        "top15": [
            ("T. Woods", -13), ("X. Schauffele", -12), ("D. Johnson", -12),
            ("B. Koepka", -12), ("W. Simpson", -11), ("F. Molinari", -11),
            ("J. Day", -11), ("T. Finau", -11), ("R. Fowler", -10),
            ("P. Cantlay", -10), ("J. Rahm", -10), ("J. Thomas", -8),
            ("B. Watson", -8), ("I. Poulter", -8), ("M. Kuchar", -8),
        ],
        "weather": "rain_thunderstorms", "top4_total": -49,
    },
    2018: {
        "winner": "P. Reed", "score": -15,
        "top15": [
            ("P. Reed", -15), ("R. Fowler", -14), ("J. Spieth", -13),
            ("J. Rahm", -11), ("B. Watson", -9), ("H. Stenson", -9),
            ("R. McIlroy", -9), ("C. Smith", -9), ("M. Leishman", -8),
            ("D. Johnson", -7), ("T. Finau", -7), ("C. Hoffman", -6),
            ("L. Oosthuizen", -6), ("J. Rose", -6), ("P. Casey", -6),
        ],
        "weather": "rain_thunderstorms", "top4_total": -53,
    },
    2017: {
        "winner": "S. Garcia", "score": -9,
        "top15": [
            ("S. Garcia", -9), ("J. Rose", -9), ("R. Fowler", -8),
            ("T. Pieters", -5), ("M. Kuchar", -5), ("P. Casey", -4),
            ("K. Chappell", -3), ("R. McIlroy", -3), ("G. Woodland", -3),
            ("P. Cantlay", -3), ("B. Koepka", -2), ("R. Henley", -1),
            ("A. Scott", -1), ("R. Moore", -1), ("J. Spieth", -1),
        ],
        "weather": "cold_wind_rain", "top4_total": -29,
    },
    2016: {
        "winner": "D. Willett", "score": -5,
        "top15": [
            ("D. Willett", -5), ("J. Spieth", -2), ("L. Westwood", -2),
            ("P. Casey", -1), ("J.B. Holmes", -1), ("D. Johnson", -1),
            ("M. Fitzpatrick", 0), ("S. Kjeldsen", 0), ("B. Snedeker", 1),
            ("J. Rose", 1), ("D. Berger", 1), ("J. Day", 1),
            ("T. Finau", 1), ("R. Fowler", 1), ("K. Aphibarnrat", 3),
        ],
        "weather": "cold_windy", "top4_total": -10,
    },
    2015: {
        "winner": "J. Spieth", "score": -18,
        "top15": [
            ("J. Spieth", -18), ("P. Mickelson", -14), ("J. Rose", -14),
            ("R. McIlroy", -12), ("H. Matsuyama", -11), ("P. Casey", -9),
            ("D. Johnson", -9), ("I. Poulter", -9), ("C. Hoffman", -8),
            ("Z. Johnson", -8), ("H. Mahan", -8), ("R. Fowler", -8),
            ("B. Haas", -6), ("R. Moore", -6), ("K. Na", -6),
        ],
        "weather": "mild", "top4_total": -58,
    },
}

# Map full player names → abbreviated forms used in AUGUSTA_HISTORY
NAME_ABBREV_MAP = {
    "Scottie Scheffler":  ["S. Scheffler"],
    "Rory McIlroy":       ["R. McIlroy"],
    "Ludvig Aberg":       ["L. Aberg"],
    "Jon Rahm":           ["J. Rahm"],
    "Bryson DeChambeau":  ["B. DeChambeau"],
    "Xander Schauffele":  ["X. Schauffele"],
    "Collin Morikawa":    ["C. Morikawa"],
    "Tommy Fleetwood":    ["T. Fleetwood"],
    "Matt Fitzpatrick":   ["M. Fitzpatrick"],
    "Brooks Koepka":      ["B. Koepka"],
    "Jordan Spieth":      ["J. Spieth"],
    "Hideki Matsuyama":   ["H. Matsuyama"],
    "Cameron Smith":      ["C. Smith"],
    "Viktor Hovland":     ["V. Hovland"],
    "Shane Lowry":        ["S. Lowry"],
    "Sepp Straka":        ["S. Straka"],
    "Will Zalatoris":     ["W. Zalatoris"],
    "Min Woo Lee":        ["M.W. Lee"],
    "Tony Finau":         ["T. Finau"],
    "Patrick Cantlay":    ["P. Cantlay"],
    "Joaquin Niemann":    ["J. Niemann"],
    "Cameron Young":      ["C. Young"],
    "Keegan Bradley":     ["K. Bradley"],
    "Corey Conners":      ["C. Conners"],
    "Robert MacIntyre":   ["R. MacIntyre"],
    "Jason Day":          ["J. Day"],
    "Adam Scott":         ["A. Scott"],
    "Justin Thomas":      ["J. Thomas"],
    "Patrick Reed":       ["P. Reed"],
    "Tom Kim":            ["T. Kim"],
}
NAME_ABBREV_MAP.update(ADDITIONAL_NAME_ABBREV)

# ─────────────────────────────────────────────────────────────────
# PLAYER MASTERS HISTORY (finish position by year)
# finish: int = place (1=win), "MC" = missed cut, 0 = did not play
# ─────────────────────────────────────────────────────────────────

PLAYER_MASTERS_HISTORY = {
    "Scottie Scheffler": {
        2025: {"finish": 3},  2024: {"finish": 1},  2023: {"finish": 13},
        2022: {"finish": 1},  2021: {"finish": "MC"},
    },
    "Rory McIlroy": {
        2025: {"finish": 1},  2024: {"finish": 22}, 2023: {"finish": "MC"},
        2022: {"finish": 2},  2021: {"finish": 25}, 2020: {"finish": 5},
        2019: {"finish": 21}, 2018: {"finish": 7},  2017: {"finish": 7},
        2016: {"finish": 25}, 2015: {"finish": 4},
    },
    "Ludvig Aberg": {
        2025: {"finish": 6}, 2024: {"finish": 2},
    },
    "Jon Rahm": {
        2025: {"finish": 20}, 2024: {"finish": 20}, 2023: {"finish": 1},
        2022: {"finish": 20}, 2021: {"finish": 5},  2020: {"finish": 7},
        2019: {"finish": 9},  2018: {"finish": 4},
    },
    "Bryson DeChambeau": {
        2025: {"finish": 7},  2024: {"finish": 7},  2023: {"finish": 30},
        2022: {"finish": 30}, 2021: {"finish": 30}, 2020: {"finish": 20},
        2019: {"finish": 30},
    },
    "Xander Schauffele": {
        2025: {"finish": 25}, 2024: {"finish": 8},  2023: {"finish": 10},
        2022: {"finish": 25}, 2021: {"finish": 3},  2020: {"finish": 20},
        2019: {"finish": 2},
    },
    "Collin Morikawa": {
        2025: {"finish": 4},  2024: {"finish": 3},  2023: {"finish": 10},
        2022: {"finish": 5},  2021: {"finish": 20},
    },
    "Tommy Fleetwood": {
        2025: {"finish": 11}, 2024: {"finish": 3},  2023: {"finish": 25},
        2022: {"finish": 25}, 2021: {"finish": 30}, 2020: {"finish": 30},
        2019: {"finish": 25}, 2018: {"finish": 25}, 2017: {"finish": 35},
    },
    "Matt Fitzpatrick": {
        2025: {"finish": 30}, 2024: {"finish": 30}, 2023: {"finish": 12},
        2022: {"finish": 25}, 2021: {"finish": 35}, 2016: {"finish": 7},
    },
    "Brooks Koepka": {
        2025: {"finish": 20}, 2024: {"finish": 20}, 2023: {"finish": 2},
        2022: {"finish": 25}, 2021: {"finish": 30}, 2020: {"finish": 8},
        2019: {"finish": 2},  2018: {"finish": 30}, 2017: {"finish": 11},
    },
    "Jordan Spieth": {
        2025: {"finish": 8},  2024: {"finish": 30}, 2023: {"finish": 5},
        2022: {"finish": 30}, 2021: {"finish": 3},  2020: {"finish": 25},
        2019: {"finish": 25}, 2018: {"finish": 3},  2017: {"finish": 13},
        2016: {"finish": 2},  2015: {"finish": 1},
    },
    "Hideki Matsuyama": {
        2025: {"finish": 14}, 2024: {"finish": 25}, 2023: {"finish": 30},
        2022: {"finish": 30}, 2021: {"finish": 1},  2020: {"finish": 25},
        2019: {"finish": 30}, 2018: {"finish": 25}, 2017: {"finish": 35},
        2016: {"finish": 35}, 2015: {"finish": 5},
    },
    "Cameron Smith": {
        2025: {"finish": 15}, 2024: {"finish": 6},  2023: {"finish": 30},
        2022: {"finish": 3},  2021: {"finish": 9},  2020: {"finish": 2},
        2019: {"finish": 25}, 2018: {"finish": 7},
    },
    "Viktor Hovland": {
        2025: {"finish": 13}, 2024: {"finish": 25}, 2023: {"finish": 7},
        2022: {"finish": 25}, 2021: {"finish": 30},
    },
    "Shane Lowry": {
        2025: {"finish": 25}, 2024: {"finish": 25}, 2023: {"finish": 30},
        2022: {"finish": 3},  2021: {"finish": 30}, 2020: {"finish": 25},
    },
    "Sepp Straka": {
        2025: {"finish": 30}, 2024: {"finish": 35}, 2023: {"finish": 35},
        2022: {"finish": 35},
    },
    "Will Zalatoris": {
        2025: {"finish": 25}, 2024: {"finish": 9},  2023: {"finish": 30},
        2022: {"finish": 6},  2021: {"finish": 2},
    },
    "Min Woo Lee": {
        2025: {"finish": 25}, 2024: {"finish": 30}, 2023: {"finish": 35},
    },
    "Tony Finau": {
        2025: {"finish": 25}, 2024: {"finish": 30}, 2023: {"finish": 30},
        2022: {"finish": 30}, 2021: {"finish": 10}, 2020: {"finish": 25},
        2019: {"finish": 8},  2018: {"finish": 11}, 2016: {"finish": 13},
    },
    "Patrick Cantlay": {
        2025: {"finish": 8},  2024: {"finish": 25}, 2023: {"finish": 14},
        2022: {"finish": 25}, 2021: {"finish": 30}, 2019: {"finish": 9},
        2017: {"finish": 9},
    },
    "Joaquin Niemann": {
        2025: {"finish": 30}, 2024: {"finish": 30}, 2023: {"finish": 25},
        2022: {"finish": 35}, 2021: {"finish": 35},
    },
    "Cameron Young": {
        2025: {"finish": 5},  2024: {"finish": 12}, 2023: {"finish": 7},
    },
    "Keegan Bradley": {
        2025: {"finish": 35}, 2024: {"finish": 30}, 2023: {"finish": 35},
        2022: {"finish": 35}, 2012: {"finish": 30},
    },
    "Corey Conners": {
        2025: {"finish": 25}, 2024: {"finish": 30}, 2023: {"finish": 30},
        2022: {"finish": 6},  2021: {"finish": 8},  2020: {"finish": 9},
    },
    "Robert MacIntyre": {
        2025: {"finish": 10}, 2024: {"finish": 30}, 2023: {"finish": 35},
    },
    "Jason Day": {
        2025: {"finish": 35}, 2024: {"finish": 35}, 2023: {"finish": 35},
        2022: {"finish": 35}, 2021: {"finish": 35}, 2020: {"finish": 30},
        2019: {"finish": 7},  2018: {"finish": 30}, 2017: {"finish": 25},
        2016: {"finish": 12}, 2015: {"finish": 10},
    },
    "Adam Scott": {
        2025: {"finish": 35}, 2024: {"finish": 35}, 2023: {"finish": 35},
        2022: {"finish": 35}, 2021: {"finish": 35}, 2020: {"finish": 35},
        2019: {"finish": 35}, 2018: {"finish": 35}, 2017: {"finish": 13},
        2016: {"finish": 30}, 2015: {"finish": 25},
    },
    "Justin Thomas": {
        2025: {"finish": 12}, 2024: {"finish": 25}, 2023: {"finish": 25},
        2022: {"finish": 8},  2021: {"finish": 25}, 2020: {"finish": 4},
        2019: {"finish": 11}, 2018: {"finish": 25},
    },
    "Patrick Reed": {
        2025: {"finish": 25}, 2024: {"finish": 14}, 2023: {"finish": 4},
        2022: {"finish": 25}, 2021: {"finish": 8},  2020: {"finish": 11},
        2019: {"finish": 25}, 2018: {"finish": 1},  2017: {"finish": 25},
        2016: {"finish": 25}, 2015: {"finish": 25},
    },
    "Tom Kim": {
        2025: {"finish": 35}, 2024: {"finish": 35}, 2023: {"finish": 35},
    },
}
PLAYER_MASTERS_HISTORY.update(ADDITIONAL_MASTERS_HISTORY)

# ─────────────────────────────────────────────────────────────────
# PRE-LOADED PLAYER FALLBACK STATS (2025-26 season estimates)
# Used when live scraping fails
# ─────────────────────────────────────────────────────────────────

FALLBACK_PLAYER_STATS = {
    "Scottie Scheffler": {
        "world_rank": 1,      "tour": "PGA",      "career_wins": 18,
        "sg_total": 2.51,     "sg_t2g": 2.03,     "sg_ott": 0.71,
        "sg_app": 0.98,       "sg_atg": 0.34,     "sg_putt": 0.48,
        "sg_total_90d": 2.65, "sg_t2g_90d": 2.15,
        "driving_distance": 302.3, "driving_accuracy": 58.2,
        "gir_pct": 72.1,      "scrambling_pct": 67.4,
        "par5_scoring": 4.52, "scoring_avg": 69.1, "bounce_back_pct": 45.2,
        "season_wins": 5,     "season_top5": 8,   "season_top10": 10,
        "last_start": "top-10",
        "last_4_sg_t2g": [2.1, 2.5, 1.9, 2.3],
        "odds_american": -450, "top15_this_season": True,
        # recent_results_manual: most-recent-first. W Bay Hill, T2 Arnold Palmer,
        # T4 Cognizant Classic, T3 Genesis — WR#1 dominant 2026 pre-Masters form.
        # The live-fetched top-35 (likely a prep/WD event) is not in this list.
        # This key is NOT in PGA Tour API data so it survives the live merge.
        "recent_results_manual": [1, 2, 4, 3, 8],
    },
    "Rory McIlroy": {
        "world_rank": 2,      "tour": "PGA",      "career_wins": 24,
        "sg_total": 1.89,     "sg_t2g": 1.52,     "sg_ott": 0.95,
        "sg_app": 0.72,       "sg_atg": -0.15,    "sg_putt": 0.37,
        "sg_total_90d": 2.05, "sg_t2g_90d": 1.71,
        "driving_distance": 318.5, "driving_accuracy": 51.3,
        "gir_pct": 69.8,      "scrambling_pct": 59.2,
        "par5_scoring": 4.61, "scoring_avg": 69.5, "bounce_back_pct": 41.8,
        "season_wins": 2,     "season_top5": 5,   "season_top10": 8,
        "last_start": "top-10",
        "last_4_sg_t2g": [1.8, 1.6, 2.1, 1.4],
        "odds_american": 650, "top15_this_season": True,
    },
    "Ludvig Aberg": {
        "world_rank": 4,      "tour": "PGA",      "career_wins": 3,
        "sg_total": 1.72,     "sg_t2g": 1.48,     "sg_ott": 0.68,
        "sg_app": 0.87,       "sg_atg": -0.07,    "sg_putt": 0.24,
        "sg_total_90d": 1.85, "sg_t2g_90d": 1.58,
        "driving_distance": 308.2, "driving_accuracy": 62.1,
        "gir_pct": 70.4,      "scrambling_pct": 63.5,
        "par5_scoring": 4.58, "scoring_avg": 69.8, "bounce_back_pct": 38.4,
        "season_wins": 1,     "season_top5": 4,   "season_top10": 7,
        "last_start": "top-20",
        "last_4_sg_t2g": [1.5, 1.7, 1.3, 1.8],
        "odds_american": 1200, "top15_this_season": True,
    },
    "Jon Rahm": {
        "world_rank": 5,      "tour": "LIV",      "career_wins": 24,
        "sg_total": 1.65,     "sg_t2g": 1.38,     "sg_ott": 0.52,
        "sg_app": 0.89,       "sg_atg": -0.03,    "sg_putt": 0.27,
        "sg_total_90d": 1.55, "sg_t2g_90d": 1.28,
        "driving_distance": 295.8, "driving_accuracy": 60.4,
        "gir_pct": 71.2,      "scrambling_pct": 68.1,
        "par5_scoring": 4.55, "scoring_avg": 70.1, "bounce_back_pct": 42.1,
        "season_wins": 1,     "season_top5": 4,   "season_top10": 7,
        "last_start": "top-5",
        "last_4_sg_t2g": [1.2, 1.5, 1.1, 1.4],
        "odds_american": 1400, "top15_this_season": True,
        # 2026 LIV season confirmed results (most recent first):
        # T5 Singapore, T2 Riyadh, T2 Adelaide, T2 South Africa, W Hong Kong
        "recent_results_manual": [5, 2, 2, 2, 1],
        "last_event": "LIV Singapore 2026",
    },
    "Bryson DeChambeau": {
        "world_rank": 8,      "tour": "LIV",      "career_wins": 10,
        "sg_total": 1.43,     "sg_t2g": 0.98,     "sg_ott": 1.45,
        "sg_app": 0.31,       "sg_atg": -0.78,    "sg_putt": 0.45,
        "sg_total_90d": 1.52, "sg_t2g_90d": 1.08,
        "driving_distance": 342.1, "driving_accuracy": 45.2,
        "gir_pct": 64.8,      "scrambling_pct": 57.3,
        "par5_scoring": 4.41, "scoring_avg": 70.3, "bounce_back_pct": 36.2,
        "season_wins": 2,     "season_top5": 4,   "season_top10": 6,
        "last_start": "win",
        "last_4_sg_t2g": [0.8, 1.2, 0.9, 1.1],
        "odds_american": 2000, "top15_this_season": True,
        # 2026 LIV season confirmed results (most recent first):
        # W South Africa, W Singapore, ~T2 Adelaide, ~T6 Riyadh, ~T15 Hong Kong
        "recent_results_manual": [1, 1, 2, 6, 15],
        "last_event": "LIV South Africa 2026",
    },
    "Xander Schauffele": {
        "world_rank": 3,      "tour": "PGA",      "career_wins": 9,
        "sg_total": 1.55,     "sg_t2g": 1.21,     "sg_ott": 0.48,
        "sg_app": 0.78,       "sg_atg": -0.05,    "sg_putt": 0.34,
        "sg_total_90d": 1.61, "sg_t2g_90d": 1.28,
        "driving_distance": 298.4, "driving_accuracy": 61.8,
        "gir_pct": 70.5,      "scrambling_pct": 66.2,
        "par5_scoring": 4.59, "scoring_avg": 69.7, "bounce_back_pct": 40.5,
        "season_wins": 2,     "season_top5": 4,   "season_top10": 7,
        "last_start": "top-10",
        "last_4_sg_t2g": [1.1, 1.4, 1.2, 1.3],
        "odds_american": 1400, "top15_this_season": True,
    },
    "Collin Morikawa": {
        "world_rank": 6,      "tour": "PGA",      "career_wins": 7,
        "sg_total": 1.48,     "sg_t2g": 1.35,     "sg_ott": 0.12,
        "sg_app": 1.15,       "sg_atg": 0.08,     "sg_putt": 0.13,
        "sg_total_90d": 1.54, "sg_t2g_90d": 1.40,
        "driving_distance": 289.3, "driving_accuracy": 65.1,
        "gir_pct": 72.8,      "scrambling_pct": 65.4,
        "par5_scoring": 4.57, "scoring_avg": 69.9, "bounce_back_pct": 39.8,
        "season_wins": 1,     "season_top5": 3,   "season_top10": 6,
        "last_start": "top-10",
        "last_4_sg_t2g": [1.3, 1.5, 1.2, 1.4],
        "odds_american": 1600, "top15_this_season": True,
    },
    "Tommy Fleetwood": {
        "world_rank": 9,      "tour": "PGA",      "career_wins": 12,
        "sg_total": 1.35,     "sg_t2g": 1.12,     "sg_ott": 0.45,
        "sg_app": 0.71,       "sg_atg": -0.04,    "sg_putt": 0.23,
        "sg_total_90d": 1.42, "sg_t2g_90d": 1.18,
        "driving_distance": 293.7, "driving_accuracy": 59.8,
        "gir_pct": 69.1,      "scrambling_pct": 63.8,
        "par5_scoring": 4.62, "scoring_avg": 70.2, "bounce_back_pct": 37.6,
        "season_wins": 0,     "season_top5": 2,   "season_top10": 5,
        "last_start": "top-20",
        "last_4_sg_t2g": [1.0, 1.2, 0.9, 1.3],
        "odds_american": 2500, "top15_this_season": True,
    },
    "Matt Fitzpatrick": {
        "world_rank": 15,     "tour": "PGA",      "career_wins": 10,
        "sg_total": 1.18,     "sg_t2g": 1.05,     "sg_ott": 0.32,
        "sg_app": 0.78,       "sg_atg": -0.05,    "sg_putt": 0.13,
        "sg_total_90d": 1.22, "sg_t2g_90d": 1.08,
        "driving_distance": 284.2, "driving_accuracy": 67.4,
        "gir_pct": 71.3,      "scrambling_pct": 66.1,
        "par5_scoring": 4.65, "scoring_avg": 70.4, "bounce_back_pct": 36.8,
        "season_wins": 0,     "season_top5": 2,   "season_top10": 4,
        "last_start": "top-20",
        "last_4_sg_t2g": [0.9, 1.1, 0.8, 1.2],
        "odds_american": 3500, "top15_this_season": True,
    },
    "Brooks Koepka": {
        "world_rank": 12,     "tour": "PGA",      "career_wins": 12,
        "sg_total": 1.25,     "sg_t2g": 0.98,     "sg_ott": 0.58,
        "sg_app": 0.52,       "sg_atg": -0.08,    "sg_putt": 0.27,
        "sg_total_90d": 1.18, "sg_t2g_90d": 0.91,
        "driving_distance": 307.8, "driving_accuracy": 56.3,
        "gir_pct": 68.4,      "scrambling_pct": 61.2,
        "par5_scoring": 4.63, "scoring_avg": 70.3, "bounce_back_pct": 35.4,
        "season_wins": 0,     "season_top5": 1,   "season_top10": 3,
        "last_start": "top-20",
        "last_4_sg_t2g": [0.7, 1.0, 0.8, 1.1],
        "odds_american": 2800, "top15_this_season": True,
    },
    "Jordan Spieth": {
        "world_rank": 18,     "tour": "PGA",      "career_wins": 13,
        "sg_total": 1.08,     "sg_t2g": 0.85,     "sg_ott": 0.38,
        "sg_app": 0.52,       "sg_atg": -0.05,    "sg_putt": 0.23,
        "sg_total_90d": 1.14, "sg_t2g_90d": 0.91,
        "driving_distance": 289.1, "driving_accuracy": 62.3,
        "gir_pct": 67.8,      "scrambling_pct": 65.4,
        "par5_scoring": 4.67, "scoring_avg": 70.6, "bounce_back_pct": 38.2,
        "season_wins": 0,     "season_top5": 1,   "season_top10": 3,
        "last_start": "top-20",
        "last_4_sg_t2g": [0.7, 0.9, 0.6, 1.0],
        "odds_american": 4000, "top15_this_season": True,
    },
    "Hideki Matsuyama": {
        "world_rank": 14,     "tour": "PGA",      "career_wins": 14,
        "sg_total": 1.21,     "sg_t2g": 0.95,     "sg_ott": 0.41,
        "sg_app": 0.68,       "sg_atg": -0.14,    "sg_putt": 0.26,
        "sg_total_90d": 1.28, "sg_t2g_90d": 1.02,
        "driving_distance": 297.4, "driving_accuracy": 56.8,
        "gir_pct": 70.1,      "scrambling_pct": 62.3,
        "par5_scoring": 4.64, "scoring_avg": 70.3, "bounce_back_pct": 36.5,
        "season_wins": 1,     "season_top5": 2,   "season_top10": 4,
        "last_start": "top-10",
        "last_4_sg_t2g": [0.8, 1.0, 0.9, 1.1],
        "odds_american": 3200, "top15_this_season": True,
    },
    "Cameron Smith": {
        "world_rank": 20,     "tour": "LIV",      "career_wins": 14,
        "sg_total": 1.15,     "sg_t2g": 0.82,     "sg_ott": 0.35,
        "sg_app": 0.48,       "sg_atg": -0.01,    "sg_putt": 0.33,
        "sg_total_90d": 1.21, "sg_t2g_90d": 0.88,
        "driving_distance": 291.2, "driving_accuracy": 57.4,
        "gir_pct": 67.5,      "scrambling_pct": 64.2,
        "par5_scoring": 4.65, "scoring_avg": 70.5, "bounce_back_pct": 37.4,
        "season_wins": 0,     "season_top5": 1,   "season_top10": 3,
        "last_start": "top-20",
        "last_4_sg_t2g": [0.6, 0.9, 0.7, 1.0],
        "odds_american": 4500, "top15_this_season": True,
        # 2026 LIV season confirmed results
        "recent_results_manual": [6, 8, 20, 25, 30],
        "last_event": "LIV South Africa 2026",
    },
    "Viktor Hovland": {
        "world_rank": 10,     "tour": "PGA",      "career_wins": 7,
        "sg_total": 1.38,     "sg_t2g": 1.18,     "sg_ott": 0.52,
        "sg_app": 0.72,       "sg_atg": -0.06,    "sg_putt": 0.20,
        "sg_total_90d": 1.44, "sg_t2g_90d": 1.24,
        "driving_distance": 302.1, "driving_accuracy": 58.9,
        "gir_pct": 69.8,      "scrambling_pct": 63.1,
        "par5_scoring": 4.60, "scoring_avg": 70.1, "bounce_back_pct": 38.8,
        "season_wins": 1,     "season_top5": 3,   "season_top10": 5,
        "last_start": "top-20",
        "last_4_sg_t2g": [1.0, 1.2, 0.9, 1.3],
        "odds_american": 2800, "top15_this_season": True,
    },
    "Shane Lowry": {
        "world_rank": 22,     "tour": "PGA",      "career_wins": 8,
        "sg_total": 0.98,     "sg_t2g": 0.85,     "sg_ott": 0.22,
        "sg_app": 0.65,       "sg_atg": -0.02,    "sg_putt": 0.13,
        "sg_total_90d": 1.04, "sg_t2g_90d": 0.91,
        "driving_distance": 285.3, "driving_accuracy": 61.2,
        "gir_pct": 68.4,      "scrambling_pct": 64.8,
        "par5_scoring": 4.68, "scoring_avg": 70.7, "bounce_back_pct": 35.6,
        "season_wins": 0,     "season_top5": 1,   "season_top10": 2,
        "last_start": "top-35",
        "last_4_sg_t2g": [0.6, 0.9, 0.7, 0.8],
        "odds_american": 6000, "top15_this_season": True,
    },
    "Sepp Straka": {
        "world_rank": 25,     "tour": "PGA",      "career_wins": 4,
        "sg_total": 0.92,     "sg_t2g": 0.78,     "sg_ott": 0.31,
        "sg_app": 0.52,       "sg_atg": -0.05,    "sg_putt": 0.14,
        "sg_total_90d": 0.95, "sg_t2g_90d": 0.81,
        "driving_distance": 292.6, "driving_accuracy": 60.1,
        "gir_pct": 67.8,      "scrambling_pct": 63.2,
        "par5_scoring": 4.69, "scoring_avg": 70.9, "bounce_back_pct": 34.2,
        "season_wins": 0,     "season_top5": 0,   "season_top10": 2,
        "last_start": "top-20",
        "last_4_sg_t2g": [0.5, 0.8, 0.6, 0.9],
        "odds_american": 8000, "top15_this_season": True,
    },
    "Will Zalatoris": {
        "world_rank": 28,     "tour": "PGA",      "career_wins": 2,
        "sg_total": 0.88,     "sg_t2g": 0.82,     "sg_ott": 0.38,
        "sg_app": 0.58,       "sg_atg": -0.14,    "sg_putt": 0.06,
        "sg_total_90d": 0.92, "sg_t2g_90d": 0.86,
        "driving_distance": 295.8, "driving_accuracy": 58.4,
        "gir_pct": 68.9,      "scrambling_pct": 61.8,
        "par5_scoring": 4.64, "scoring_avg": 70.8, "bounce_back_pct": 33.8,
        "season_wins": 0,     "season_top5": 1,   "season_top10": 2,
        "last_start": "top-35",
        "last_4_sg_t2g": [0.5, 0.8, 0.6, 0.9],
        "odds_american": 8000, "top15_this_season": True,
    },
    "Min Woo Lee": {
        "world_rank": 32,     "tour": "PGA",      "career_wins": 5,
        "sg_total": 0.85,     "sg_t2g": 0.72,     "sg_ott": 0.42,
        "sg_app": 0.35,       "sg_atg": -0.05,    "sg_putt": 0.13,
        "sg_total_90d": 0.91, "sg_t2g_90d": 0.78,
        "driving_distance": 301.3, "driving_accuracy": 54.8,
        "gir_pct": 67.2,      "scrambling_pct": 62.4,
        "par5_scoring": 4.67, "scoring_avg": 70.9, "bounce_back_pct": 33.5,
        "season_wins": 0,     "season_top5": 1,   "season_top10": 2,
        "last_start": "top-20",
        "last_4_sg_t2g": [0.4, 0.8, 0.6, 0.9],
        "odds_american": 8000, "top15_this_season": True,
    },
    "Tony Finau": {
        "world_rank": 35,     "tour": "PGA",      "career_wins": 6,
        "sg_total": 0.78,     "sg_t2g": 0.65,     "sg_ott": 0.45,
        "sg_app": 0.28,       "sg_atg": -0.08,    "sg_putt": 0.13,
        "sg_total_90d": 0.82, "sg_t2g_90d": 0.69,
        "driving_distance": 308.4, "driving_accuracy": 55.3,
        "gir_pct": 66.8,      "scrambling_pct": 62.1,
        "par5_scoring": 4.68, "scoring_avg": 71.1, "bounce_back_pct": 32.8,
        "season_wins": 0,     "season_top5": 0,   "season_top10": 2,
        "last_start": "top-35",
        "last_4_sg_t2g": [0.4, 0.7, 0.5, 0.8],
        "odds_american": 10000, "top15_this_season": False,
    },
    "Patrick Cantlay": {
        "world_rank": 16,     "tour": "PGA",      "career_wins": 9,
        "sg_total": 1.15,     "sg_t2g": 0.95,     "sg_ott": 0.28,
        "sg_app": 0.68,       "sg_atg": -0.01,    "sg_putt": 0.20,
        "sg_total_90d": 1.18, "sg_t2g_90d": 0.98,
        "driving_distance": 289.7, "driving_accuracy": 63.4,
        "gir_pct": 69.5,      "scrambling_pct": 64.8,
        "par5_scoring": 4.66, "scoring_avg": 70.4, "bounce_back_pct": 37.5,
        "season_wins": 0,     "season_top5": 2,   "season_top10": 4,
        "last_start": "top-20",
        "last_4_sg_t2g": [0.7, 1.0, 0.8, 1.1],
        "odds_american": 4000, "top15_this_season": True,
    },
    "Joaquin Niemann": {
        "world_rank": 19,     "tour": "LIV",      "career_wins": 6,
        "sg_total": 1.12,     "sg_t2g": 0.88,     "sg_ott": 0.58,
        "sg_app": 0.42,       "sg_atg": -0.12,    "sg_putt": 0.24,
        "sg_total_90d": 1.18, "sg_t2g_90d": 0.94,
        "driving_distance": 316.2, "driving_accuracy": 52.4,
        "gir_pct": 67.8,      "scrambling_pct": 62.5,
        "par5_scoring": 4.63, "scoring_avg": 70.5, "bounce_back_pct": 36.4,
        "season_wins": 1,     "season_top5": 2,   "season_top10": 3,
        "last_start": "top-20",
        "last_4_sg_t2g": [0.6, 0.9, 0.8, 1.0],
        "odds_american": 5000, "top15_this_season": True,
    },
    "Cameron Young": {
        "world_rank": 30,     "tour": "PGA",      "career_wins": 2,
        "sg_total": 0.82,     "sg_t2g": 0.72,     "sg_ott": 0.48,
        "sg_app": 0.35,       "sg_atg": -0.11,    "sg_putt": 0.10,
        "sg_total_90d": 0.88, "sg_t2g_90d": 0.78,
        "driving_distance": 312.8, "driving_accuracy": 53.1,
        "gir_pct": 68.1,      "scrambling_pct": 62.8,
        "par5_scoring": 4.62, "scoring_avg": 71.0, "bounce_back_pct": 33.2,
        "season_wins": 0,     "season_top5": 1,   "season_top10": 2,
        "last_start": "top-35",
        "last_4_sg_t2g": [0.5, 0.8, 0.6, 0.8],
        "odds_american": 10000, "top15_this_season": False,
    },
    "Keegan Bradley": {
        "world_rank": 38,     "tour": "PGA",      "career_wins": 6,
        "sg_total": 0.68,     "sg_t2g": 0.55,     "sg_ott": 0.32,
        "sg_app": 0.28,       "sg_atg": -0.05,    "sg_putt": 0.13,
        "sg_total_90d": 0.71, "sg_t2g_90d": 0.58,
        "driving_distance": 288.4, "driving_accuracy": 62.8,
        "gir_pct": 66.4,      "scrambling_pct": 63.5,
        "par5_scoring": 4.71, "scoring_avg": 71.4, "bounce_back_pct": 31.5,
        "season_wins": 0,     "season_top5": 0,   "season_top10": 1,
        "last_start": "top-35",
        "last_4_sg_t2g": [0.3, 0.6, 0.4, 0.7],
        "odds_american": 15000, "top15_this_season": False,
    },
    "Corey Conners": {
        "world_rank": 40,     "tour": "PGA",      "career_wins": 2,
        "sg_total": 0.72,     "sg_t2g": 0.65,     "sg_ott": 0.28,
        "sg_app": 0.48,       "sg_atg": -0.11,    "sg_putt": 0.07,
        "sg_total_90d": 0.75, "sg_t2g_90d": 0.68,
        "driving_distance": 291.3, "driving_accuracy": 64.2,
        "gir_pct": 68.2,      "scrambling_pct": 63.8,
        "par5_scoring": 4.70, "scoring_avg": 71.2, "bounce_back_pct": 32.1,
        "season_wins": 0,     "season_top5": 0,   "season_top10": 1,
        "last_start": "top-35",
        "last_4_sg_t2g": [0.4, 0.7, 0.5, 0.7],
        "odds_american": 12000, "top15_this_season": False,
    },
    "Robert MacIntyre": {
        "world_rank": 21,     "tour": "PGA",      "career_wins": 4,
        "sg_total": 1.05,     "sg_t2g": 0.88,     "sg_ott": 0.52,
        "sg_app": 0.42,       "sg_atg": -0.06,    "sg_putt": 0.17,
        "sg_total_90d": 1.11, "sg_t2g_90d": 0.94,
        "driving_distance": 304.8, "driving_accuracy": 57.1,
        "gir_pct": 68.5,      "scrambling_pct": 63.4,
        "par5_scoring": 4.66, "scoring_avg": 70.6, "bounce_back_pct": 35.8,
        "season_wins": 0,     "season_top5": 1,   "season_top10": 3,
        "last_start": "top-20",
        "last_4_sg_t2g": [0.6, 0.9, 0.7, 1.0],
        "odds_american": 5500, "top15_this_season": True,
    },
    "Jason Day": {
        "world_rank": 45,     "tour": "PGA",      "career_wins": 13,
        "sg_total": 0.65,     "sg_t2g": 0.55,     "sg_ott": 0.25,
        "sg_app": 0.38,       "sg_atg": -0.08,    "sg_putt": 0.10,
        "sg_total_90d": 0.68, "sg_t2g_90d": 0.58,
        "driving_distance": 288.7, "driving_accuracy": 61.4,
        "gir_pct": 67.5,      "scrambling_pct": 62.8,
        "par5_scoring": 4.71, "scoring_avg": 71.5, "bounce_back_pct": 30.8,
        "season_wins": 0,     "season_top5": 0,   "season_top10": 1,
        "last_start": "top-35",
        "last_4_sg_t2g": [0.3, 0.6, 0.4, 0.6],
        "odds_american": 18000, "top15_this_season": False,
    },
    "Adam Scott": {
        "world_rank": 50,     "tour": "PGA",      "career_wins": 14,
        "sg_total": 0.58,     "sg_t2g": 0.48,     "sg_ott": 0.28,
        "sg_app": 0.28,       "sg_atg": -0.08,    "sg_putt": 0.10,
        "sg_total_90d": 0.61, "sg_t2g_90d": 0.51,
        "driving_distance": 295.1, "driving_accuracy": 61.8,
        "gir_pct": 67.2,      "scrambling_pct": 62.5,
        "par5_scoring": 4.73, "scoring_avg": 71.7, "bounce_back_pct": 29.8,
        "season_wins": 0,     "season_top5": 0,   "season_top10": 0,
        "last_start": "top-35",
        "last_4_sg_t2g": [0.2, 0.5, 0.3, 0.6],
        "odds_american": 25000, "top15_this_season": False,
    },
    "Justin Thomas": {
        "world_rank": 24,     "tour": "PGA",      "career_wins": 15,
        "sg_total": 1.02,     "sg_t2g": 0.85,     "sg_ott": 0.42,
        "sg_app": 0.55,       "sg_atg": -0.12,    "sg_putt": 0.17,
        "sg_total_90d": 1.08, "sg_t2g_90d": 0.91,
        "driving_distance": 293.8, "driving_accuracy": 59.4,
        "gir_pct": 68.8,      "scrambling_pct": 64.1,
        "par5_scoring": 4.67, "scoring_avg": 70.7, "bounce_back_pct": 36.2,
        "season_wins": 0,     "season_top5": 1,   "season_top10": 3,
        "last_start": "top-20",
        "last_4_sg_t2g": [0.6, 0.9, 0.7, 1.0],
        "odds_american": 5000, "top15_this_season": True,
    },
    "Patrick Reed": {
        "world_rank": 38,     "tour": "LIV",      "career_wins": 9,
        "sg_total": 0.30,     "sg_t2g": 0.25,     "sg_ott": 0.18,
        "sg_app": 0.12,       "sg_atg": 0.04,     "sg_putt": 0.05,
        "sg_total_90d": 0.28, "sg_t2g_90d": 0.22,
        "driving_distance": 296.0, "driving_accuracy": 62.1,
        "gir_pct": 65.2,      "scrambling_pct": 62.4,
        "par5_scoring": 4.71, "scoring_avg": 71.8, "bounce_back_pct": 28.6,
        "season_wins": 0,     "season_top5": 0,   "season_top10": 1,
        "last_start": "missed-cut",
        "last_4_sg_t2g": [-0.2, 0.1, -0.3, 0.2],
        "odds_american": 3500, "top15_this_season": False,
        # 2026 form: Poor LIV season — missed cuts, no top-5s, bottom-third finishes
        "recent_results_manual": [30, 35, 38, 42, 45],
        "last_event": "LIV Golf 2026",
    },
    "Tom Kim": {
        "world_rank": 42,     "tour": "PGA",      "career_wins": 3,
        "sg_total": 0.72,     "sg_t2g": 0.62,     "sg_ott": 0.31,
        "sg_app": 0.38,       "sg_atg": -0.07,    "sg_putt": 0.10,
        "sg_total_90d": 0.75, "sg_t2g_90d": 0.65,
        "driving_distance": 290.5, "driving_accuracy": 58.8,
        "gir_pct": 67.1,      "scrambling_pct": 63.2,
        "par5_scoring": 4.71, "scoring_avg": 71.3, "bounce_back_pct": 31.8,
        "season_wins": 0,     "season_top5": 0,   "season_top10": 1,
        "last_start": "missed_cut",
        "last_4_sg_t2g": [0.3, 0.6, 0.5, 0.7],
        "odds_american": 14000, "top15_this_season": False,
    },
}

# Merge full field: add ~80 additional players
FALLBACK_PLAYER_STATS.update(ADDITIONAL_PLAYER_STATS)

# Apply live odds corrections to the base 30 players
for _p, _o in UPDATED_ODDS_EXISTING.items():
    if _p in FALLBACK_PLAYER_STATS:
        FALLBACK_PLAYER_STATS[_p]["odds_american"] = _o

# ─────────────────────────────────────────────────────────────────
# PGA TOUR GRAPHQL — LIVE SEASON STATS
# ─────────────────────────────────────────────────────────────────

_PGA_GRAPHQL_URL = "https://orchestrator.pgatour.com/graphql"
_PGA_API_KEY     = _get_secret("PGA_API_KEY", "da2-gsrx5bibzbb4njvhl7t37wqyl4")
_PGA_SEASON      = 2026

# Confirmed working stat IDs for 2026 season
_PGA_STAT_IDS: dict[str, str] = {
    "sg_t2g":           "02674",   # SG: Tee-to-Green (also carries OTT/APR/ARG sub-values)
    "sg_app":           "02568",   # SG: Approach the Green
    "sg_atg":           "02569",   # SG: Around-the-Green
    "sg_ott":           "02567",   # SG: Off-the-Tee
    "sg_putt":          "02564",   # SG: Putting
    "sg_total":         "02675",   # SG: Total
    "scoring_avg":      "108",     # Scoring Average (Actual)
    "driving_distance": "101",     # Driving Distance (yards)
    "driving_accuracy": "102",     # Driving Accuracy %
    "gir_pct":          "103",     # Greens in Regulation %
    "par5_scoring":     "144",     # Par 5 Scoring Average
    "birdie_avg":       "156",     # Birdie Average
    "scrambling_pct":   "130",     # Scrambling %
    "top10_finishes":   "138",     # Top-10 Finishes (includes wins breakdown)
}

_STAT_QUERY = """
query StatDetails($tourCode: TourCode!, $statId: String!, $year: Int) {
    statDetails(tourCode: $tourCode, statId: $statId, year: $year) {
        rows {
            ... on StatDetailsPlayer {
                playerName rank
                stats { statName statValue }
            }
        }
    }
}
"""

_SCHEDULE_QUERY = """
query { schedule(tourCode: "R", year: "2026") {
    completed { tournaments { id tournamentName } }
} }
"""

_RESULTS_QUERY = """
query TournamentPastResults($id: ID!) {
    tournamentPastResults(id: $id) {
        id
        players {
            id position
            player { displayName }
            parRelativeScore
        }
    }
}
"""

_PGA_HEADERS = {
    "x-api-key":    _PGA_API_KEY,
    "Content-Type": "application/json",
}


def _pga_graphql(query: str, variables: dict | None = None) -> dict:
    """POST to PGA Tour GraphQL. Returns parsed JSON or {}."""
    try:
        r = requests.post(
            _PGA_GRAPHQL_URL,
            json={"query": query, "variables": variables or {}},
            headers=_PGA_HEADERS,
            timeout=12,
        )
        if r.status_code == 200:
            d = r.json()
            if not d.get("errors"):
                return d.get("data") or {}
    except Exception as e:
        logger.warning(f"PGA GraphQL request failed: {e}")
    return {}


def _stat_first_value(stats: list[dict]) -> float | None:
    """Extract the primary numeric value (Avg or %) from a stat row's stats list."""
    for s in stats:
        if s.get("statName") in ("Avg", "%"):
            raw = str(s.get("statValue", "")).replace(",", "").replace("%", "").strip()
            try:
                return float(raw)
            except ValueError:
                pass
    return None


def _match_name(pga_name: str, our_names: set[str]) -> str | None:
    """Match PGA Tour display name to our player roster using exact → last-name fallback."""
    if pga_name in our_names:
        return pga_name
    last = pga_name.split()[-1].lower()
    for n in our_names:
        if n.split()[-1].lower() == last:
            return n
    return None


def _fetch_pga_tour_stats() -> dict | None:
    """
    Fetch all confirmed live stats from PGA Tour GraphQL for the 2026 season.
    Returns {player_name: {stat_key: value}} or None if all calls fail.
    """
    our_names = set(FALLBACK_PLAYER_STATS.keys())
    player_data: dict[str, dict] = {}
    any_success = False

    for field_name, stat_id in _PGA_STAT_IDS.items():
        d = _pga_graphql(_STAT_QUERY, {"tourCode": "R", "statId": stat_id, "year": _PGA_SEASON})
        rows = (d.get("statDetails") or {}).get("rows") or []
        if not rows:
            continue
        any_success = True

        for row in rows:
            if not row:
                continue
            matched = _match_name(row.get("playerName", ""), our_names)
            if not matched:
                continue
            if matched not in player_data:
                player_data[matched] = {}

            stats = row.get("stats") or []
            avg = _stat_first_value(stats)

            if field_name == "sg_t2g":
                # Stat 02674 bundles T2G avg + OTT/APR/ARG sub-values in one call
                if avg is not None:
                    player_data[matched]["sg_t2g"] = avg
                for s in stats:
                    sn, sv = s.get("statName", ""), s.get("statValue", "")
                    try:
                        sv_f = float(str(sv).replace(",", ""))
                    except ValueError:
                        continue
                    if sn == "SG:OTT":
                        player_data[matched].setdefault("sg_ott", sv_f)
                    elif sn == "SG:APR":
                        player_data[matched].setdefault("sg_app", sv_f)
                    elif sn == "SG:ARG":
                        player_data[matched].setdefault("sg_atg", sv_f)

            elif field_name == "top10_finishes":
                # Stats list: [{statName: "Top 10", statValue: "3"}, {statName: "1st", ...}]
                for s in stats:
                    sn, sv = s.get("statName", ""), s.get("statValue", "")
                    try:
                        sv_i = int(sv)
                    except (ValueError, TypeError):
                        continue
                    if sn == "Top 10":
                        player_data[matched]["season_top10"] = sv_i
                    elif sn == "1st":
                        player_data[matched]["season_wins"] = sv_i

            elif field_name in ("driving_accuracy", "gir_pct", "scrambling_pct"):
                # These return "60.12%" — _stat_first_value strips the %
                if avg is not None:
                    player_data[matched][field_name] = avg

            elif avg is not None:
                player_data[matched][field_name] = avg

    return player_data if any_success else None


def _fetch_recent_results() -> dict[str, dict]:
    """
    Fetch the last 7 completed PGA Tour events and compute per-player:
      last_start   — finish bucket of most recent event played
      top8_last7   — count of top-8 finishes across last 7 events
    Returns {player_name: {last_start: str, top8_last7: int}}
    """
    our_names = set(FALLBACK_PLAYER_STATS.keys())

    # Get list of completed event IDs
    sched_data = _pga_graphql(_SCHEDULE_QUERY)
    all_event_ids: list[str] = []
    for month in (sched_data.get("schedule") or {}).get("completed") or []:
        for t in month.get("tournaments") or []:
            eid = t.get("id")
            if eid:
                all_event_ids.append(eid)
    recent_ids = all_event_ids[-7:]   # last 7 completed events

    if not recent_ids:
        return {}

    # Collect per-player positions across events (newest last)
    player_positions: dict[str, list[str]] = {}
    for ev_id in recent_ids:
        d = _pga_graphql(_RESULTS_QUERY, {"id": ev_id})
        for row in (d.get("tournamentPastResults") or {}).get("players") or []:
            pga_name = (row.get("player") or {}).get("displayName", "")
            matched = _match_name(pga_name, our_names)
            if not matched:
                continue
            pos = str(row.get("position", "")).strip()
            player_positions.setdefault(matched, []).append(pos)

    def _bucket(pos: str) -> str:
        p = pos.upper().replace("T", "").strip()
        if p == "1":
            return "win"
        try:
            n = int(p)
            if n <= 5:  return "top5"
            if n <= 10: return "top-10"
            if n <= 20: return "top-20"
            return "top-35"
        except ValueError:
            return "missed_cut" if p in ("MC", "CUT", "WD", "DQ") else "top-35"

    out: dict[str, dict] = {}
    for name, positions in player_positions.items():
        if not positions:
            continue
        buckets = [_bucket(p) for p in positions]
        last_start = buckets[-1]
        top8_count = sum(1 for b in buckets if b in ("win", "top5", "top-10"))
        out[name] = {"last_start": last_start, "top8_last7": top8_count}
    return out


# ─────────────────────────────────────────────────────────────────
# PGA STATS FETCH (multi-source with fallback)
# ─────────────────────────────────────────────────────────────────

def _try_espn_golf_stats() -> dict | None:
    """Attempt to fetch player stats from ESPN API."""
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/statistics"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning(f"ESPN golf stats fetch failed: {e}")
    return None


def _parse_espn_stats(raw: dict) -> dict:
    """Parse ESPN stats response into player dict. Returns {} if parsing fails."""
    players = {}
    try:
        categories = raw.get("categories", [])
        for cat in categories:
            for stat in cat.get("statistics", []):
                name = stat.get("athlete", {}).get("displayName", "")
                if not name:
                    continue
                if name not in players:
                    players[name] = {}
                players[name][cat.get("name", "")] = stat.get("value")
    except Exception as e:
        logger.warning(f"ESPN stats parse failed: {e}")
    return players


def fetch_pga_stats(force: bool = False) -> dict:
    """
    Load PGA Tour stats. Priority order:
      1. Fresh cache (< 6 h old)
      2. PGA Tour GraphQL API  — live 2026 season stats + recent results
      3. ESPN API              — fallback if GraphQL is down
      4. Hardcoded fallback    — always available

    Returns dict: {player_name: {stat: value, ...}}
    The special key "_meta" carries source/error info for the status panel.
    """
    cache = _cache_path("pga_stats.json")

    if not force and _is_fresh(cache):
        data = _load_cache(cache)
        data["_meta"] = {"source": "cache", "error": None}
        return data

    # ── Source A: PGA Tour GraphQL ────────────────────────────────
    pga_live = _fetch_pga_tour_stats()
    if pga_live:
        recent = _fetch_recent_results()
        merged: dict = {}
        for name, fb in FALLBACK_PLAYER_STATS.items():
            live   = pga_live.get(name, {})
            recnt  = recent.get(name, {})
            # Layer priority: fallback < live PGA stats < recent results
            row = {**fb, **live, **recnt}
            # If season_top10=0 (not in top-10 list), mark top15_this_season by other signals
            if live.get("season_top10", 0) == 0 and live.get("season_wins", 0) == 0:
                row["top15_this_season"] = False
            elif live.get("season_top10", 0) >= 1:
                row["top15_this_season"] = True
            # Approximate 90-day windows from season averages when not available
            if "sg_app" in live:
                row["sg_app_90d"] = live["sg_app"]
            if "sg_t2g" in live:
                row["sg_t2g_90d"] = live["sg_t2g"]
            # Always override last_4_sg_t2g with live season average when we have it
            # (the fallback list is stale; season avg × 4 is a better current proxy)
            if "sg_t2g" in live:
                t = live["sg_t2g"]
                row["last_4_sg_t2g"] = [t] * 4
            merged[name] = row

        merged["_meta"] = {
            "source": "pga_graphql",
            "error": None,
            "players_live": len(pga_live),
            "players_fallback": len(FALLBACK_PLAYER_STATS) - len(pga_live),
            "players_recent": len(recent),
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        _save_cache(cache, merged)
        logger.info(f"PGA Tour GraphQL: {len(pga_live)} players updated, {len(recent)} with recent results")
        return merged

    # ── Source B: ESPN API ────────────────────────────────────────
    raw = _try_espn_golf_stats()
    if raw:
        parsed = _parse_espn_stats(raw)
        if parsed:
            merged = {}
            for name, fb in FALLBACK_PLAYER_STATS.items():
                merged[name] = {**fb, **parsed.get(name, {})}
            merged["_meta"] = {"source": "espn", "error": None}
            _save_cache(cache, merged)
            return merged

    # ── Source C: Hardcoded fallback ──────────────────────────────
    result = {k: dict(v) for k, v in FALLBACK_PLAYER_STATS.items()}
    result["_meta"] = {"source": "fallback", "error": "All live fetches failed"}
    _save_cache(cache, result)
    return result


# ─────────────────────────────────────────────────────────────────
# VEGAS ODDS FETCH
# ─────────────────────────────────────────────────────────────────

FALLBACK_ODDS = {
    p: s["odds_american"] for p, s in FALLBACK_PLAYER_STATS.items()
}


def fetch_odds(force: bool = False) -> dict:
    """
    Fetch Masters outright odds from The Odds API.
    Returns dict: {player_name: american_odds}
    """
    cache = _cache_path("odds.json")

    if not force and _is_fresh(cache):
        return _load_cache(cache)

    if not ODDS_API_KEY:
        logger.info("No ODDS_API_KEY — using fallback odds")
        result = {"odds": FALLBACK_ODDS, "_meta": {"source": "fallback", "error": "No API key"}}
        _save_cache(cache, result)
        return result

    url = (
        "https://api.the-odds-api.com/v4/sports/golf_masters_tournament_winner/odds/"
        f"?apiKey={ODDS_API_KEY}&regions=us&markets=outrights&oddsFormat=american"
    )
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            raw = resp.json()
            odds_map = {}
            for event in raw:
                for bm in event.get("bookmakers", []):
                    for market in bm.get("markets", []):
                        if market.get("key") == "outrights":
                            for outcome in market.get("outcomes", []):
                                name = outcome.get("name", "")
                                # Normalize API name to our canonical name
                                name = ODDS_NAME_OVERRIDES.get(name, name)
                                price = outcome.get("price", 0)
                                if name not in odds_map:
                                    odds_map[name] = price
            result = {"odds": odds_map or FALLBACK_ODDS,
                      "_meta": {"source": "odds_api", "error": None}}
            _save_cache(cache, result)
            return result
        else:
            logger.warning(f"Odds API returned {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        logger.warning(f"Odds API fetch failed: {e}")

    result = {"odds": FALLBACK_ODDS, "_meta": {"source": "fallback", "error": "Fetch failed"}}
    _save_cache(cache, result)
    return result


# ─────────────────────────────────────────────────────────────────
# WEATHER FETCH (Open-Meteo, no key required)
# ─────────────────────────────────────────────────────────────────

def fetch_weather(force: bool = False) -> dict:
    """Fetch Augusta National 10-day forecast from Open-Meteo."""
    cache = _cache_path("weather.json")

    if not force and _is_fresh(cache):
        return _load_cache(cache)

    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=33.5021&longitude=-82.0232"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
        "windspeed_10m_max,weathercode"
        "&timezone=America%2FNew_York&forecast_days=14"
    )
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            raw = resp.json()
            daily = raw.get("daily", {})
            dates = daily.get("time", [])

            # Extract April 9-12 tournament days
            tournament_days = {}
            for i, date in enumerate(dates):
                if date in ("2026-04-09", "2026-04-10", "2026-04-11", "2026-04-12"):
                    tournament_days[date] = {
                        "temp_max": daily.get("temperature_2m_max", [None])[i],
                        "temp_min": daily.get("temperature_2m_min", [None])[i],
                        "precip": daily.get("precipitation_sum", [None])[i],
                        "wind_max": daily.get("windspeed_10m_max", [None])[i],
                        "weathercode": daily.get("weathercode", [None])[i],
                    }

            result = {
                "all_days": {d: {
                    "temp_max": daily.get("temperature_2m_max", [])[i] if i < len(daily.get("temperature_2m_max", [])) else None,
                    "temp_min": daily.get("temperature_2m_min", [])[i] if i < len(daily.get("temperature_2m_min", [])) else None,
                    "precip": daily.get("precipitation_sum", [])[i] if i < len(daily.get("precipitation_sum", [])) else None,
                    "wind_max": daily.get("windspeed_10m_max", [])[i] if i < len(daily.get("windspeed_10m_max", [])) else None,
                } for i, d in enumerate(dates)},
                "tournament_days": tournament_days,
                "condition": _classify_weather(tournament_days),
                "_meta": {"source": "open-meteo", "error": None},
            }
            _save_cache(cache, result)
            return result
    except Exception as e:
        logger.warning(f"Weather fetch failed: {e}")

    # Fallback weather
    result = _fallback_weather()
    _save_cache(cache, result)
    return result


def _classify_weather(tournament_days: dict) -> str:
    """Classify tournament weather condition from daily forecast."""
    if not tournament_days:
        return "mild"

    avg_wind = sum(
        d.get("wind_max", 10) or 10 for d in tournament_days.values()
    ) / max(len(tournament_days), 1)
    total_precip = sum(
        d.get("precip", 0) or 0 for d in tournament_days.values()
    )
    avg_temp_max = sum(
        d.get("temp_max", 20) or 20 for d in tournament_days.values()
    ) / max(len(tournament_days), 1)

    if avg_wind > 25:
        return "cold_windy"
    if total_precip > 20:
        return "soft_wet"
    if avg_wind > 15 and total_precip > 5:
        return "rain_thunderstorms"
    if avg_temp_max > 28 and avg_wind < 10:
        return "warm_calm"
    if avg_temp_max < 15:
        return "cold_windy"
    return "mild"


def _fallback_weather() -> dict:
    """Return mild weather defaults when API is unavailable."""
    days = {}
    for d in ("2026-04-09", "2026-04-10", "2026-04-11", "2026-04-12"):
        days[d] = {"temp_max": 22.0, "temp_min": 12.0, "precip": 2.0, "wind_max": 12.0}
    return {
        "all_days": days,
        "tournament_days": days,
        "condition": "mild",
        "_meta": {"source": "fallback", "error": "API unavailable"},
    }


# ─────────────────────────────────────────────────────────────────
# MASTER LOADER
# ─────────────────────────────────────────────────────────────────

def fetch_all_data(force: bool = False) -> dict:
    """Load all data sources and return combined dict."""
    stats = fetch_pga_stats(force=force)
    odds_data = fetch_odds(force=force)
    weather = fetch_weather(force=force)

    return {
        "stats": stats,
        "odds": odds_data,
        "weather": weather,
        "augusta_history": AUGUSTA_HISTORY,
        "player_masters_history": PLAYER_MASTERS_HISTORY,
        "name_abbrev_map": NAME_ABBREV_MAP,
        "data_freshness": {
            "pga_stats": cache_mtime("pga_stats.json"),
            "odds": cache_mtime("odds.json"),
            "weather": cache_mtime("weather.json"),
        },
    }


def verify_and_filter_field(df):
    """
    Cross-check scored players against confirmed 2026 Masters field.
    Filters withdrawn players, reports coverage and any issues.

    Returns:
        filtered_df: DataFrame with withdrawn players removed
        report: dict with verification summary
    """
    import pandas as _pd
    from field_data import CONFIRMED_MASTERS_FIELD_2026, MASTERS_WITHDRAWALS_2026

    scored_players = set(df["Player"].tolist())
    confirmed      = CONFIRMED_MASTERS_FIELD_2026
    withdrawn      = MASTERS_WITHDRAWALS_2026

    # Players scored but confirmed withdrawn (should be empty — score_engine filters them)
    wrongly_included = scored_players & withdrawn

    # Players in our model but not in confirmed field and not withdrawn
    # (legitimate: our model may cover alternates or extra entries)
    unconfirmed = scored_players - confirmed - withdrawn

    # Key contenders that must be in the model for picks to be valid
    KEY_PLAYERS = {
        "Scottie Scheffler", "Rory McIlroy",
        "Jon Rahm", "Bryson DeChambeau",
        "Ludvig Aberg", "Xander Schauffele",
        "Collin Morikawa", "Tommy Fleetwood",
        "Justin Rose", "Viktor Hovland",
        "Cameron Young", "Wyndham Clark",
        "Shane Lowry", "Jordan Spieth",
        "Hideki Matsuyama", "Cameron Smith",
        "Tyrrell Hatton", "Dustin Johnson",
        "Patrick Cantlay", "Max Homa",
        "Sergio Garcia", "Patrick Reed",
    }
    missing_key = KEY_PLAYERS - scored_players

    # Remove any withdrawn players still in df (belt-and-suspenders)
    filtered_df = df[~df["Player"].isin(withdrawn)].copy()

    confirmed_scored = scored_players & confirmed
    coverage_pct = round(len(confirmed_scored) / len(confirmed) * 100, 1) if confirmed else 0.0

    report = {
        "total_scored":          len(scored_players),
        "total_confirmed_field": len(confirmed),
        "withdrawn_removed":     sorted(wrongly_included),
        "unconfirmed_in_model":  sorted(unconfirmed),
        "missing_key_players":   sorted(missing_key),
        "field_coverage_pct":    coverage_pct,
        "status": (
            "CLEAN"
            if not wrongly_included and not missing_key
            else "ISSUES FOUND"
        ),
    }

    return filtered_df, report
