"""
score_engine.py — Composite Augusta Score model.
Components: Form (32%) + Course Fit (30%) + Augusta DNA (13%) + Vegas (20%) + Trajectory (5%).
Injury multiplier applied to final composite (not additive).

Form sub-weights  : SG T2G 50% | Last start 18% | Top-8 recency 12% | Season results 20%
DNA sub-weights   : Weighted history 50% | Best finish 25% | Starts count 15% | Last result 10%
Fit sub-weights   : SG App Season 18% | SG App 90d 12% | Par-5 28% | Bogey avoid 16% |
                    SG ATG 18% | Drive dist 5% | Sunday scoring 3%  (normalized to 100%)
Vegas sub-weights : Implied prob 70% | Model-market divergence 30%
Trajectory        : 60-day world rank change → ±8/±4/0 mapped to 0-100 band
Chaos Coefficient : Wind>15 mph or Temp<55°F → rebalance: Form 35%, Fit 33%, DNA 10%, Vegas 17%, Traj 5%
Chalk Penalty     : Scaled — odds < +600 → −2 pts | odds < +800 → −1 pt | +800+ → 0 (toggleable)
2026 course adj   : +3 pts top-20 DD (tree loss holes 3/10/11/15/16) | +2 pts top-20 SG App (firm greens)
Augusta Cut Rate  : Penalty applied for players with historical cut rate < 0.75 at Augusta.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from fetch_data import PLAYER_MASTERS_HISTORY, FALLBACK_PLAYER_STATS
from field_data import AUGUSTA_CUT_RATES, CUT_RATE_FLOOR, MASTERS_WITHDRAWALS_2026

# ── Current market odds (American format) ─────────────────────────
# Takes precedence over odds.json when computing Vegas component.
# Update weekly as markets move closer to tournament.
CURRENT_ODDS_OVERRIDE: dict[str, int] = {
    "Scottie Scheffler":    500,
    "Jon Rahm":             950,
    "Bryson DeChambeau":   1000,
    "Rory McIlroy":        1200,
    "Ludvig Aberg":        1500,
    "Xander Schauffele":   1600,
    "Matt Fitzpatrick":    2000,
    "Cameron Young":       2000,
    "Tommy Fleetwood":     2200,
    "Collin Morikawa":     3000,
    "Justin Rose":         3000,
    "Robert MacIntyre":    3300,
    "Patrick Reed":        3500,
    "Hideki Matsuyama":    4000,
    "Min Woo Lee":         4000,
    "Brooks Koepka":       4500,
    "Jordan Spieth":       4500,
    "Chris Gotterup":      4500,
    "Shane Lowry":         5500,
    "Viktor Hovland":      5500,
    "Russell Henley":      5500,
    "Akshay Bhatia":       6000,
    "Patrick Cantlay":     6500,
    "Jacob Bridgeman":     6500,
    "Nicolai Hojgaard":    6500,
    "Jake Knapp":          7000,
    "Adam Scott":          7000,
    "Tyrrell Hatton":      7000,
    "Cameron Smith":       8000,
}

# ── Putting concern adjustments (applied to fit_raw before normalization) ──
PUTTING_CONCERN: dict[str, float] = {
    "Rory McIlroy": -4.0,   # ranked outside top 100 SG:putting 2026
}
from player_context import (
    get_injury_multiplier, get_injury_status,
    get_pretournament_events, played_tune_up, events_label,
    PRE_MASTERS_PENALTY,
    PRETOURNAMENT_SCHEDULE,
)
from player_form_data import RANKING_CHANGE_60D, SG_APP_90D, SUNDAY_SCORING_DIFF

# ─────────────────────────────────────────────────────────────────
# 2026 COURSE CHANGE BONUSES
# Hurricane Helene tree loss → wider corridors on 3, 10, 11, 15, 16
# Rebuilt greens 1, 8, 15, 16 → firmer, more demanding approach
# ─────────────────────────────────────────────────────────────────
DD_COURSE_BONUS  = 3.0   # pts for top-20 driving distance
APP_COURSE_BONUS = 2.0   # pts for top-20 SG Approach
TOP_N_COURSE_ADJ = 20    # "top-20 on tour" threshold

# ─────────────────────────────────────────────────────────────────
# FORM SUB-COMPONENT WEIGHTS
# season_results increased to 0.20: 81% of Masters winners
# had already won earlier that season — most undercounted
# signal in 15-year backtest analysis
# ─────────────────────────────────────────────────────────────────
FORM_WEIGHTS = {
    "t2g":            0.50,   # SG T2G last 4 events
    "last_start":     0.18,   # Last start finish quality
    "top8_recency":   0.12,   # Top-8 in last 7 starts (proxied by season_top10)
    "season_results": 0.20,   # Season wins / top-5 count
}

# ─────────────────────────────────────────────────────────────────
# DNA SUB-COMPONENT WEIGHTS
# ─────────────────────────────────────────────────────────────────
DNA_WEIGHTS = {
    "history":    0.50,   # Weighted finish history, recency-decayed
    "best":       0.25,   # Best-ever Augusta finish
    "starts":     0.15,   # Prior Augusta starts count
    "last":       0.10,   # Most recent Augusta appearance result
}


# ─────────────────────────────────────────────────────────────────
# RECENCY MULTIPLIERS FOR AUGUSTA DNA
# ─────────────────────────────────────────────────────────────────

RECENCY_MULT = {
    2025: 1.0, 2024: 0.9, 2023: 0.8, 2022: 0.7, 2021: 0.6,
    2020: 0.5, 2019: 0.4, 2018: 0.3, 2017: 0.2, 2016: 0.15, 2015: 0.1,
}

FINISH_POINTS = {
    "win": 15, "top5": 10, "top10": 7, "top25": 4, "top40": 2,
    "made_cut": 1, "missed_cut": -2,
}


# ─────────────────────────────────────────────────────────────────
# HARD FILTER FLAGS
# ─────────────────────────────────────────────────────────────────

def compute_flags(row: pd.Series) -> list[str]:
    flags = []
    if row.get("world_rank", 999) > 25:
        flags.append("Rank > 25")
    masters_hist = PLAYER_MASTERS_HISTORY.get(row.get("name", ""), {})
    if not masters_hist:
        flags.append("No Masters starts")
    if row.get("career_wins", 0) < 4:
        flags.append("< 4 career wins")
    if not row.get("top15_this_season", False):
        flags.append("No top-15 this season")
    if row.get("sg_total", 0) < 0.67:
        flags.append("SG Total < +0.67")
    if row.get("sg_app", 0) < 0.84:
        flags.append("SG App < +0.84")
    if row.get("sg_ott", 0) < 0.60:
        flags.append("SG OTT < +0.60")
    return flags


# ─────────────────────────────────────────────────────────────────
# COMPONENT A: RECENT FORM (0–100)
# ─────────────────────────────────────────────────────────────────

def compute_form_raw(player_stats: dict, all_players_stats: list[dict]) -> float:
    """
    Weighted Recent Form score. Each sub-component is scaled 0–100 before weighting.
    Returns a weighted composite (0–100 range); caller normalizes across field.

    Sub-weights (from FORM_WEIGHTS):
      SG T2G last 4 events    55%
      Last start finish       20%
      Top-8 recency           15%  (proxied by season_top10 if top8_last7 unavailable)
      Season wins / top-5     10%
    """
    # ── Sub-component 1: SG T2G last 4 events (55%) ──────────────────────────
    last4 = player_stats.get("last_4_sg_t2g", [])
    sg_t2g_sum = sum(last4) if last4 else (player_stats.get("sg_t2g", 0) * 4)

    # Field z-score
    field_sums = []
    for p in all_players_stats:
        pv = p.get("last_4_sg_t2g", [])
        field_sums.append(sum(pv) if pv else p.get("sg_t2g", 0) * 4)
    field_mean = np.mean(field_sums) if field_sums else 0.0
    field_std  = max(float(np.std(field_sums)), 0.01) if field_sums else 1.0

    z = (sg_t2g_sum - field_mean) / field_std
    t2g_sub = float(np.clip((z + 3) / 6 * 100, 0, 100))

    # Activity penalty if player has fewer than 4 recent data points
    if last4 and len(last4) < 4:
        t2g_sub = max(0.0, t2g_sub - 20.0)
    elif not last4:
        t2g_sub = max(0.0, t2g_sub - 15.0)

    # ── Sub-component 2: Last start finish quality (20%) ──────────────────────
    last_start = str(player_stats.get("last_start", "")).lower().strip()
    _last_lut = {
        "win":        100.0,
        "top5":        90.0, "top-5":        90.0,
        "top-10":      72.0, "top_10":       72.0,
        "top-20":      52.0, "top_20":       52.0,
        "top-35":      34.0, "top_35":       34.0,
        "missed_cut":   8.0, "mc":            8.0,
    }
    last_sub = _last_lut.get(last_start, 44.0)   # unknown → slightly below average

    # ── Sub-component 3: Top-8 recency in last 7 starts (15%) ────────────────
    # Use 'top8_last7' if available; fall back to season_top10 as proxy.
    # 5 top-10s in a season ≈ elite consistency = 100 pts.
    top8_count = player_stats.get("top8_last7", player_stats.get("season_top10", 0))
    top8_sub = float(np.clip(top8_count / 5.0 * 100, 0, 100))

    # ── Sub-component 4: Season wins / top-5 count (10%) ─────────────────────
    wins = player_stats.get("season_wins", 0)
    top5 = player_stats.get("season_top5", 0)
    # Wins count most (3 wins ≈ elite season = 100). Incremental top-5s add value.
    season_raw = wins * 25.0 + max(0, top5 - wins) * 10.0
    season_sub = float(np.clip(season_raw / 75.0 * 100, 0, 100))

    # ── Weighted composite ────────────────────────────────────────────────────
    form_raw = (
        FORM_WEIGHTS["t2g"]            * t2g_sub
        + FORM_WEIGHTS["last_start"]   * last_sub
        + FORM_WEIGHTS["top8_recency"] * top8_sub
        + FORM_WEIGHTS["season_results"] * season_sub
    )

    # ── Recent win bonus ──────────────────────────────────────────────────────
    # 81% of Masters winners won earlier that same season.
    # A win in the last 2 events before Augusta is the
    # strongest single pre-tournament signal in the data.
    recent_win_bonus = 0
    season_wins = int(player_stats.get("season_wins", 0))
    recent_results = player_stats.get("recent_results_manual", [])

    # For LIV players: use recent_results_manual[0] as the most-recent-start signal
    # (stored most-recent-first; 1 = win)
    _last_was_win = (
        last_start == "win"
        or (recent_results and recent_results[0] == 1)
    )
    _consec_wins = sum(1 for r in recent_results if r == 1)

    if _last_was_win:
        recent_win_bonus = 15  # won most recent event
        # Stack bonus for back-to-back wins (e.g. DeChambeau — 2 consec LIV wins)
        if _consec_wins >= 2:
            recent_win_bonus += 5  # → +20 total
    elif season_wins >= 1 and last_start in ("top5", "top-5", "top-10", "top_10"):
        recent_win_bonus = 8   # won earlier, still in form
    elif season_wins >= 2:
        recent_win_bonus = 5   # multiple wins, proven closer

    form_raw += recent_win_bonus

    return form_raw


# ─────────────────────────────────────────────────────────────────
# COMPONENT B: AUGUSTA DNA (0–100)
# ─────────────────────────────────────────────────────────────────

def _finish_points(finish) -> float:
    if finish == "MC" or finish == "missed_cut":
        return FINISH_POINTS["missed_cut"]
    try:
        f = int(finish)
    except (ValueError, TypeError):
        return 0
    if f == 1:
        return FINISH_POINTS["win"]
    if f <= 5:
        return FINISH_POINTS["top5"]
    if f <= 10:
        return FINISH_POINTS["top10"]
    if f <= 25:
        return FINISH_POINTS["top25"]
    if f <= 40:
        return FINISH_POINTS["top40"]
    if f <= 70:
        return FINISH_POINTS["made_cut"]
    return 0  # did not play / no data


def compute_dna_raw(player_name: str) -> float:
    """
    Weighted Augusta DNA score. Each sub-component is scaled 0–1 before weighting.
    Returns a weighted composite (0–1 range); caller normalizes across field.

    Sub-weights (from DNA_WEIGHTS):
      Weighted finish history (recency-decayed)   50%
      Best-ever Augusta finish                    25%
      Prior Augusta starts count                  15%
      Last Augusta appearance result              10%
    """
    history = PLAYER_MASTERS_HISTORY.get(player_name, {})
    if not history:
        return 0.0

    # ── Sub-component 1: Weighted finish history, recency-decayed (50%) ──────
    weighted_hist = 0.0
    for year, result in history.items():
        mult = RECENCY_MULT.get(int(year), 0.05)
        pts  = _finish_points(result.get("finish", 0))
        weighted_hist += pts * mult
    # Approximate max: winning every recent year ≈ 15×(1.0+0.9+0.8+0.7+0.6) = 30
    hist_sub = float(np.clip(weighted_hist / 25.0, 0.0, 1.0))

    # ── Sub-component 2: Best-ever Augusta finish (25%) ──────────────────────
    all_pts = [_finish_points(r.get("finish", 0)) for r in history.values()]
    best_pts = max(all_pts) if all_pts else 0.0
    # Win=15, Top-5=10, Top-10=7, Top-25=4, Top-40=2, MadeCut=1, MC=-2
    # Normalize: win → 1.0, mc → 0.0  (shift -2→0, 15→1.0 via range 17)
    best_sub = float(np.clip((best_pts + 2) / 17.0, 0.0, 1.0))

    # ── Sub-component 3: Prior starts count (15%) ────────────────────────────
    # Count years with non-trivial finish entry (present in history)
    n_starts = len(history)
    # 10+ starts = full Augusta veteran = 1.0
    starts_sub = float(np.clip(n_starts / 10.0, 0.0, 1.0))

    # ── Sub-component 4: Last Augusta appearance result (10%) ────────────────
    sorted_years = sorted(history.keys(), reverse=True)
    if sorted_years:
        last_pts = _finish_points(history[sorted_years[0]].get("finish", 0))
        last_sub = float(np.clip((last_pts + 2) / 17.0, 0.0, 1.0))
    else:
        last_sub = 0.0

    # ── Weighted composite ────────────────────────────────────────────────────
    return (
        DNA_WEIGHTS["history"] * hist_sub
        + DNA_WEIGHTS["best"]  * best_sub
        + DNA_WEIGHTS["starts"] * starts_sub
        + DNA_WEIGHTS["last"]  * last_sub
    )


# ─────────────────────────────────────────────────────────────────
# COMPONENT C: COURSE FIT (0–100)
# ─────────────────────────────────────────────────────────────────

def compute_fit_raw(player_stats: dict, all_players_stats: list[dict],
                    weights: dict | None = None) -> float:
    """
    Weighted Course Fit composite. Returns field-weighted score (0–100 range).

    Default weights (2026 spec — 7 sub-components, normalized to 100%):
      SG Approach season-long  18%  (sg_app_season / sg_app)
      SG Approach last 90 days 12%  (sg_app_90d  — isolates recent peak iron quality)
      Par-5 scoring average    28%  (4 reachable par-5s create up to 16-stroke separation)
      Bogey avoidance rate     16%
      SG Around Green          18%  (scrambling uniquely predictive at Augusta)
      Driving Distance          5%  (reduced post Hurricane-Helene analysis)
      Sunday scoring diff       3%  (redundant with Form component — reduced)

    NOTE: SG Putting is intentionally excluded — Augusta Masters winners rank
    an average of 98th in SG Putting; it is not a positive predictor.
    Legacy key 'sg_app' is automatically mapped to 'sg_app_season'.
    """
    if weights is None:
        weights = {
            "sg_app_season": 0.18,
            "sg_app_90d":    0.12,
            "par5":          0.28,
            "bogey_avoid":   0.16,
            "sg_atg":        0.18,
            "drive_dist":    0.05,
            "sunday_scoring": 0.03,
        }

    # Legacy compat: if caller passes 'sg_app' (old key), treat as 'sg_app_season'
    if "sg_app" in weights and "sg_app_season" not in weights:
        weights = dict(weights)
        weights["sg_app_season"] = weights.pop("sg_app")

    # Normalize weights to 1.0 (handles user sliders that may not sum to 1)
    weights = _normalize_weights(weights)

    def _field_norm(key: str, invert: bool = False) -> dict[str, float]:
        """Field-normalize a stat to 0–100."""
        vals = {p.get("_name", ""): p.get(key, 0) or 0 for p in all_players_stats}
        mn, mx = min(vals.values()), max(vals.values())
        spread = mx - mn if mx != mn else 1.0
        normed = {}
        for pname, v in vals.items():
            n = (v - mn) / spread * 100
            normed[pname] = (100 - n) if invert else n
        return normed

    name = player_stats.get("_name", "")

    # Field-normalize each sub-component
    sg_app_season_n  = _field_norm("sg_app")           # season SG App (uses standard key)
    sg_app_90d_n     = _field_norm("sg_app_90d")       # 90-day SG App (enriched in score_players)
    par5_n           = _field_norm("par5_scoring", invert=True)
    bogey_n          = _field_norm("scoring_avg",  invert=True)
    sg_atg_n         = _field_norm("sg_atg")
    drive_dist_n     = _field_norm("driving_distance")
    # Sunday scoring diff: more negative = better. Invert so higher is better.
    sunday_n         = _field_norm("sunday_scoring_diff", invert=True)

    score = (
        weights.get("sg_app_season",  0.180) * sg_app_season_n.get(name, 50)
        + weights.get("sg_app_90d",   0.120) * sg_app_90d_n.get(name, 50)
        + weights.get("par5",         0.280) * par5_n.get(name, 50)
        + weights.get("bogey_avoid",  0.160) * bogey_n.get(name, 50)
        + weights.get("sg_atg",       0.180) * sg_atg_n.get(name, 50)
        + weights.get("drive_dist",   0.050) * drive_dist_n.get(name, 50)
        + weights.get("sunday_scoring", 0.030) * sunday_n.get(name, 50)
    )
    return score


# ─────────────────────────────────────────────────────────────────
# COMPONENT E: RANKING TRAJECTORY (0–100)
# ─────────────────────────────────────────────────────────────────

def compute_trajectory_raw(player_name: str) -> float:
    """
    World ranking trajectory score based on 60-day rank change.
    Returns a raw score (0–100 band); normalized across field in score_players().

    Tier mapping:
      +10 or better  → 85  (surging — significant momentum)
      +5 to +9       → 70  (improving — moderate positive trend)
      +1 to +4       → 58  (slight upward drift)
       0             → 50  (neutral / stable)
      -1 to -4       → 42  (slight decline)
      -5 to -9       → 30  (declining form)
      -10 or worse   → 15  (sharp regression)

    Players not in RANKING_CHANGE_60D default to 50 (neutral).
    """
    change = RANKING_CHANGE_60D.get(player_name, 0)
    if change >= 10:
        return 85.0
    elif change >= 5:
        return 70.0
    elif change >= 1:
        return 58.0
    elif change == 0:
        return 50.0
    elif change >= -4:
        return 42.0
    elif change >= -9:
        return 30.0
    else:
        return 15.0


# ─────────────────────────────────────────────────────────────────
# COMPONENT D: VEGAS CALIBRATION (0–100)
# ─────────────────────────────────────────────────────────────────

def american_to_implied_prob(odds: float) -> float:
    """Convert American odds to raw implied probability."""
    if odds < 0:
        return (-odds) / (-odds + 100)
    else:
        return 100 / (odds + 100)


def remove_vig(probs: dict[str, float]) -> dict[str, float]:
    """Normalize probabilities to sum to 1.0 (remove vig)."""
    total = sum(probs.values())
    if total == 0:
        return probs
    return {k: v / total for k, v in probs.items()}


def compute_vegas_scores(
    odds_map: dict[str, float],
    player_names: list[str],
    model_scores: dict[str, float] | None = None,
    sub_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Vegas Calibration score (0–100). Blends implied market probability with a
    model-vs-market divergence signal.

    sub_weights: {"implied_prob": 0.70, "divergence": 0.30}
      - implied_prob: pure vig-removed market win probability, normalized 0-100
      - divergence:   bonus/penalty based on how much our (form+dna+fit) model
                      agrees with or exceeds the market's assessment.
                      Positive divergence (model > market) boosts the Vegas score.

    model_scores: pre-computed 3-factor (form+dna+fit) weighted score per player.
                  If None, divergence component is skipped.
    """
    if sub_weights is None:
        sub_weights = {"implied_prob": 0.70, "divergence": 0.30}

    # Resolve odds with partial-name fallback
    raw_probs = {
        name: american_to_implied_prob(odds_map.get(name, 50000))
        for name in player_names
    }
    default_prob = american_to_implied_prob(50000)
    for name in player_names:
        if raw_probs[name] == default_prob:
            for odds_name, odds_val in odds_map.items():
                if name.split()[-1].lower() in odds_name.lower():
                    raw_probs[name] = american_to_implied_prob(odds_val)
                    break

    clean = remove_vig(raw_probs)
    max_prob = max(clean.values()) if clean else 1.0
    implied_scores = {name: (clean[name] / max_prob) * 100 for name in player_names}

    # Pure implied-probability only when no model scores provided
    if model_scores is None or sub_weights.get("divergence", 0) == 0:
        return implied_scores

    # Divergence component: model_score - market_score, capped ±30 pts, mapped 0-100
    # Neutral (no divergence) → 50.  Model +30 over market → 80.  Model -30 → 20.
    w_imp = sub_weights["implied_prob"]
    w_div = sub_weights["divergence"]
    result = {}
    for name in player_names:
        market_s = implied_scores[name]
        model_s  = model_scores.get(name, market_s)   # default: no divergence
        raw_div  = float(np.clip(model_s - market_s, -30, 30))
        div_score = 50.0 + raw_div                     # 20–80 range
        result[name] = w_imp * market_s + w_div * div_score
    return result


# ─────────────────────────────────────────────────────────────────
# WEATHER MODIFIER
# ─────────────────────────────────────────────────────────────────

def apply_weather_modifier(score: float, condition: str, player_stats: dict,
                           fit_weights: dict) -> float:
    """Apply weather-based multiplier to composite score. Range: 0.9–1.1."""
    dd = player_stats.get("driving_distance", 295)
    par5 = player_stats.get("par5_scoring", 4.65)
    sg_t2g = player_stats.get("sg_t2g", 0.8)

    modifier = 1.0

    if condition in ("soft_wet", "november_soft"):
        # Long hitters and par-5 birdie machines benefit
        dd_bonus = np.clip((dd - 295) / 50 * 0.05, -0.03, 0.05)
        par5_bonus = np.clip((4.70 - par5) / 0.3 * 0.04, -0.02, 0.04)
        modifier = 1.0 + dd_bonus + par5_bonus

    elif condition == "fast_firm":
        # Approach precision and bogey avoidance rewarded
        sg_app = player_stats.get("sg_app", 0.5)
        approach_bonus = np.clip(sg_app / 1.15 * 0.05, -0.03, 0.05)
        scoring_avg = player_stats.get("scoring_avg", 70.5)
        bogey_bonus = np.clip((71.5 - scoring_avg) / 2.0 * 0.04, -0.02, 0.04)
        modifier = 1.0 + approach_bonus + bogey_bonus

    elif condition in ("cold_windy", "rain_thunderstorms", "cold_wind_rain"):
        # Ball-strikers (SG T2G) rewarded; short hitters penalized in wind
        t2g_bonus = np.clip(sg_t2g / 2.0 * 0.04, -0.03, 0.04)
        dd_penalty = np.clip((295 - dd) / 50 * 0.03, 0, 0.03)
        modifier = 1.0 + t2g_bonus - dd_penalty

    return np.clip(score * modifier, 0, 110)


# ─────────────────────────────────────────────────────────────────
# CHAOS COEFFICIENT
# ─────────────────────────────────────────────────────────────────

def _detect_chaos_mode(weather: dict) -> bool:
    """
    Return True if conditions warrant Chaos Coefficient weight rebalancing.
    Triggers when average wind > 15 mph OR average high temperature < 55°F.
    Under chaos: Form 35%, Fit 30%, DNA 15%, Vegas 15%, Trajectory 5%.
    """
    # Wind: try direct key, then compute from tournament_days
    avg_wind = weather.get("avg_wind_mph", None)
    if avg_wind is None:
        days = weather.get("tournament_days", {})
        if days:
            winds = [d.get("wind_max", 0) for d in days.values()
                     if isinstance(d.get("wind_max"), (int, float))]
            avg_wind = sum(winds) / len(winds) if winds else 0.0
        else:
            avg_wind = 0.0

    # Temperature: try °F directly, else convert from °C tournament highs
    avg_temp_f = weather.get("avg_temp_f", None)
    if avg_temp_f is None:
        days = weather.get("tournament_days", {})
        if days:
            temps_c = [d.get("temp_max", 20) for d in days.values()
                       if isinstance(d.get("temp_max"), (int, float))]
            avg_temp_c = sum(temps_c) / len(temps_c) if temps_c else 20.0
            avg_temp_f = avg_temp_c * 9.0 / 5.0 + 32.0
        else:
            avg_temp_f = 65.0  # default: mild Augusta spring

    return float(avg_wind) > 15.0 or float(avg_temp_f) < 55.0


# ─────────────────────────────────────────────────────────────────
# NORMALIZE HELPERS
# ─────────────────────────────────────────────────────────────────

def _normalize_series(values: list[float], low: float = 0, high: float = 100) -> list[float]:
    arr = np.array(values, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return [50.0] * len(values)
    return list((arr - mn) / (mx - mn) * (high - low) + low)


def _normalize_weights(w: dict) -> dict:
    total = sum(w.values())
    if total == 0:
        n = len(w)
        return {k: 1.0 / n for k in w}
    return {k: v / total for k, v in w.items()}


# ─────────────────────────────────────────────────────────────────
# MASTER SCORING FUNCTION
# ─────────────────────────────────────────────────────────────────

def score_players(
    data: dict,
    component_weights: dict | None = None,
    fit_weights: dict | None = None,
    apply_chalk_penalty: bool = True,
) -> pd.DataFrame:
    """
    Score all players. Returns a DataFrame sorted by Augusta_Score descending.

    component_weights: {form, dna, fit, vegas, trajectory}
      Defaults: Form 32% | Fit 27% | DNA 18% | Vegas 18% | Trajectory 5%
      Under Chaos Coefficient (wind>15 or temp<55°F):
        Form 35% | Fit 30% | DNA 15% | Vegas 15% | Trajectory 5%
    fit_weights: {sg_app_season, sg_app_90d, par5, bogey_avoid, sg_atg, drive_dist, sunday_scoring}
      Defaults (normalized from raw 20/15/18/18/14/7/10)
      Legacy key 'sg_app' is remapped to 'sg_app_season'.
      NOTE: three_putt key is silently ignored if present (removed from model).
    apply_chalk_penalty: If True, subtract 5 pts from players with odds < +600.
    """
    weather = data.get("weather", {})
    chaos_active = _detect_chaos_mode(weather)

    # Chaos Coefficient: override weights when conditions are extreme
    # DNA reduced to 0.13: 76% of Masters top-10 finishers 2010-2025
    # had zero prior Augusta top-10s — history is over-weighted
    # relative to skill profile fit.
    # Fit increased to 0.30: par-5 and scrambling are most
    # Augusta-specific predictors with available data.
    # Vegas increased to 0.20: market is efficient at Augusta
    # and captures LIV player form we can't measure directly.
    if component_weights is None:
        if chaos_active:
            component_weights = {
                "form": 0.35, "dna": 0.10, "fit": 0.33,
                "vegas": 0.17, "trajectory": 0.05,
            }
        else:
            component_weights = {
                "form": 0.32, "dna": 0.13, "fit": 0.30,
                "vegas": 0.20, "trajectory": 0.05,
            }
    component_weights = _normalize_weights(component_weights)

    if fit_weights is None:
        # par5 increased to 0.28: 4 reachable par-5s at Augusta
        # create up to 16 stroke separation over 72 holes.
        # sg_atg increased to 0.18: scrambling uniquely predictive
        # at Augusta vs other tour stops.
        # sunday_scoring reduced: redundant with Form component.
        fit_weights = {
            "sg_app_season": 0.18, "sg_app_90d": 0.12,
            "par5": 0.28, "bogey_avoid": 0.16,
            "sg_atg": 0.18, "drive_dist": 0.05,
            "sunday_scoring": 0.03,
        }
    # Strip any legacy keys before normalizing
    fit_weights = {k: v for k, v in fit_weights.items() if k != "three_putt"}
    # Legacy compat: 'sg_app' → 'sg_app_season'
    if "sg_app" in fit_weights and "sg_app_season" not in fit_weights:
        fit_weights["sg_app_season"] = fit_weights.pop("sg_app")
    fit_weights = _normalize_weights(fit_weights)

    stats_raw = data.get("stats", {})
    odds_data = data.get("odds", {})
    _base_odds_map = odds_data.get("odds", {}) if isinstance(odds_data, dict) else {}
    # Merge current odds override — takes precedence over stale cached odds.json values
    odds_map = {**_base_odds_map, **CURRENT_ODDS_OVERRIDE}
    condition = weather.get("condition", "mild")

    # Build player list (exclude _meta and withdrawn players)
    player_names = [
        k for k in stats_raw
        if not k.startswith("_") and k not in MASTERS_WITHDRAWALS_2026
    ]
    if not player_names:
        player_names = [
            k for k in FALLBACK_PLAYER_STATS
            if k not in MASTERS_WITHDRAWALS_2026
        ]

    # Enrich stats with _name, sg_app_90d, sunday_scoring_diff for fit sub-components
    all_stats = []
    for name in player_names:
        s = dict(stats_raw.get(name, FALLBACK_PLAYER_STATS.get(name, {})))
        s["_name"] = name
        # Apply current odds override so EV / Odds_American columns stay consistent
        if name in CURRENT_ODDS_OVERRIDE:
            s["odds_american"] = CURRENT_ODDS_OVERRIDE[name]
        # Inject 90-day SG App (fallback: 90% of season value)
        s["sg_app_90d"] = SG_APP_90D.get(
            name, round(s.get("sg_app", 0.5) * 0.90, 3)
        )
        # Inject Sunday scoring diff (fallback: neutral -0.5 strokes)
        s["sunday_scoring_diff"] = SUNDAY_SCORING_DIFF.get(name, -0.50)
        all_stats.append(s)

    # ── Raw component scores ──────────────────────────────────────────────────
    form_raws       = [compute_form_raw(s, all_stats) for s in all_stats]
    dna_raws        = [compute_dna_raw(name) for name in player_names]
    fit_raws        = [compute_fit_raw(s, all_stats, fit_weights) for s in all_stats]
    trajectory_raws = [compute_trajectory_raw(name) for name in player_names]

    # Apply targeted Fit adjustments before normalization
    for i, name in enumerate(player_names):
        if name in PUTTING_CONCERN:
            fit_raws[i] = max(0.0, fit_raws[i] + PUTTING_CONCERN[name])

    # Normalize to 0–100 across the field
    form_scores       = _normalize_series(form_raws)
    dna_scores        = _normalize_series(dna_raws)
    fit_scores        = fit_raws   # already 0–100 from weighted 0–100 sub-components
    trajectory_scores = _normalize_series(trajectory_raws)

    # ── 3-factor model score for Vegas divergence (no circular dependency) ────
    # Weight form/dna/fit proportionally, excluding Vegas & Trajectory
    _sum3 = component_weights["form"] + component_weights["dna"] + component_weights["fit"]
    _sum3 = _sum3 if _sum3 > 0 else 1.0
    three_factor_scores = {
        name: (
            component_weights["form"] / _sum3 * form_scores[i]
            + component_weights["dna"]  / _sum3 * dna_scores[i]
            + component_weights["fit"]  / _sum3 * fit_scores[i]
        )
        for i, name in enumerate(player_names)
    }

    vegas_scores_map = compute_vegas_scores(
        odds_map, player_names,
        model_scores=three_factor_scores,
        sub_weights={"implied_prob": 0.70, "divergence": 0.30},
    )
    vegas_scores = [vegas_scores_map.get(name, 50.0) for name in player_names]

    # ── 2026 Course Change Adjustments (discrete top-20 bonuses) ─────────────
    # +3 pts: top-20 driving distance (wider corridors, holes 3/10/11/15/16)
    # +2 pts: top-20 SG Approach (firm rebuilt greens 1/8/15/16)
    dd_ranked  = sorted(range(len(player_names)),
                        key=lambda i: all_stats[i].get("driving_distance", 0) or 0,
                        reverse=True)
    app_ranked = sorted(range(len(player_names)),
                        key=lambda i: all_stats[i].get("sg_app", 0) or 0,
                        reverse=True)
    top20_dd_idx  = set(dd_ranked[:TOP_N_COURSE_ADJ])
    top20_app_idx = set(app_ranked[:TOP_N_COURSE_ADJ])

    course_bonuses = []
    for i in range(len(player_names)):
        bonus = (DD_COURSE_BONUS if i in top20_dd_idx else 0.0) + \
                (APP_COURSE_BONUS if i in top20_app_idx else 0.0)
        course_bonuses.append(bonus)

    # ── Pre-compute win probs once ────────────────────────────────────────────
    raw_prob_map = {
        n: american_to_implied_prob(odds_map.get(n, 50000))
        for n in player_names
    }
    clean_probs = remove_vig(raw_prob_map)

    rows = []
    for i, name in enumerate(player_names):
        s = all_stats[i]
        win_prob_pct = clean_probs.get(name, 0.0) * 100

        # ── Pre-tournament schedule penalty (applied to Form component only) ──
        # PRE_MASTERS_PENALTY is now a dict; player-specific values override _default.
        schedule = get_pretournament_events(name)
        _no_tune_up = schedule is not None and not (schedule["valero"] or schedule["houston"])
        if _no_tune_up:
            if isinstance(PRE_MASTERS_PENALTY, dict):
                _penalty = PRE_MASTERS_PENALTY.get(name, PRE_MASTERS_PENALTY.get("_default", -3.0))
            else:
                _penalty = float(PRE_MASTERS_PENALTY)
            form_score_used = max(0.0, form_scores[i] + _penalty)
        else:
            form_score_used = form_scores[i]

        # ── Composite (with adjusted form score, trajectory, 5 components) ──
        composite = (
            component_weights["form"]       * form_score_used
            + component_weights["dna"]      * dna_scores[i]
            + component_weights["fit"]      * fit_scores[i]
            + component_weights["vegas"]    * vegas_scores[i]
            + component_weights.get("trajectory", 0.0) * trajectory_scores[i]
        )

        # ── Chalk Penalty: scaled by how short the odds are ──────────────────
        # odds < +600 → −2 pts  |  odds < +800 → −1 pt  |  +800+ → no penalty
        odds_val = s.get("odds_american", 50000)
        chalk_penalty_applied = False
        if apply_chalk_penalty and isinstance(odds_val, (int, float)) and 0 < odds_val:
            if odds_val < 600:
                composite = max(0.0, composite - 2.0)
                chalk_penalty_applied = True
            elif odds_val < 800:
                composite = max(0.0, composite - 1.0)
                chalk_penalty_applied = True

        # ── Weather modifier ──
        composite_adj = apply_weather_modifier(composite, condition, s, fit_weights)

        # ── 2026 Course Change Bonus (additive, before injury multiplier) ──
        firm_green_bonus = course_bonuses[i]   # column kept as Firm_Green_Bonus for UI compat
        composite_adj = min(110.0, composite_adj + firm_green_bonus)

        # ── Augusta cut rate modifier ─────────────────────────────────────────
        # A missed cut costs a team approximately +20 strokes relative to a
        # player who makes it and finishes T30. Players with poor Augusta cut
        # history carry real aggregate score risk regardless of talent.
        cut_rate = AUGUSTA_CUT_RATES.get(name, 0.75)
        if cut_rate < CUT_RATE_FLOOR:
            cut_penalty = (CUT_RATE_FLOOR - cut_rate) * 40
            composite_adj = max(0.0, composite_adj - cut_penalty)
        elif cut_rate < 0.75:
            cut_penalty = (0.75 - cut_rate) * 20
            composite_adj = max(0.0, composite_adj - cut_penalty)

        # ── Injury multiplier (applied last — scales the whole score) ──
        injury_status = get_injury_status(name)
        injury_mult = get_injury_multiplier(name)
        composite_adj = composite_adj * injury_mult

        # ── Hard-filter flags ──
        flags = compute_flags(pd.Series({**s, "name": name}))

        # ── Pre-Masters schedule label ──
        premasters_label = events_label(name)
        if schedule is not None and not (schedule["valero"] or schedule["houston"]):
            flags.append("No tune-up event")

        rows.append({
            "Player": name,
            "Tour": s.get("tour", "PGA"),
            "World_Rank": s.get("world_rank", 999),
            "Augusta_Score": round(composite_adj, 2),
            "Augusta_Cut_Rate": round(cut_rate, 2),
            "Form_Score": round(form_score_used, 2),
            "DNA_Score": round(dna_scores[i], 2),
            "Fit_Score": round(fit_scores[i], 2),
            "Vegas_Score": round(vegas_scores[i], 2),
            "Trajectory_Score": round(trajectory_scores[i], 2),
            "Chalk_Penalty": chalk_penalty_applied,
            "Vegas_Win_Prob": round(win_prob_pct, 2),
            "Odds_American": s.get("odds_american", 50000),
            "Injury_Status": injury_status,
            "Firm_Green_Bonus": round(firm_green_bonus, 2),
            "Pre_Masters_Events": premasters_label,
            "SG_Total": s.get("sg_total", 0),
            "SG_T2G": s.get("sg_t2g", 0),
            "SG_OTT": s.get("sg_ott", 0),
            "SG_App": s.get("sg_app", 0),
            "SG_ATG": s.get("sg_atg", 0),
            "SG_Putt": s.get("sg_putt", 0),
            "Drive_Dist": s.get("driving_distance", 295),
            "Drive_Acc": s.get("driving_accuracy", 60),
            "Par5_Avg": s.get("par5_scoring", 4.65),
            "Scoring_Avg": s.get("scoring_avg", 71.0),
            "Career_Wins": s.get("career_wins", 0),
            "Season_Wins": s.get("season_wins", 0),
            "Season_Top5": s.get("season_top5", 0),
            "Season_Top10": s.get("season_top10", 0),
            "Last_Start": s.get("last_start", ""),
            "Flags": "; ".join(flags) if flags else "",
            "Flag_Count": len(flags),
        })

    df = pd.DataFrame(rows)

    # Store chaos mode flag as DataFrame attribute (accessible via df.attrs)
    df.attrs["chaos_mode"] = chaos_active

    # ── Post-processing: DNA/Form Divergence flag ──
    # Flag players where Augusta DNA score is ≥25 pts higher than Form score.
    # Signals historical Augusta pedigree may be outrunning current competitive form.
    dna_form_gap = df["DNA_Score"] - df["Form_Score"]
    divergence_mask = dna_form_gap >= 25.0
    for idx in df[divergence_mask].index:
        existing = df.at[idx, "Flags"]
        new_flag = "DNA/Form Divergence"
        df.at[idx, "Flags"] = (existing + "; " + new_flag) if existing else new_flag
        df.at[idx, "Flag_Count"] = df.at[idx, "Flag_Count"] + 1

    # Add ownership and EV score (requires win probs)
    top20_df = df.nlargest(20, "Vegas_Win_Prob")
    top20_prob_sum = top20_df["Vegas_Win_Prob"].sum()
    if top20_prob_sum == 0:
        top20_prob_sum = 1.0

    df["Ownership_Pct"] = (df["Vegas_Win_Prob"] / top20_prob_sum * 100).clip(lower=0.5)
    # ownership_pct is 0-100 range; keep in % units so sqrt dampens correctly
    df["EV_Score"] = df["Augusta_Score"] * (1.0 / np.sqrt(df["Ownership_Pct"]))
    df["EV_Score"] = df["EV_Score"].round(2)

    return df.sort_values("Augusta_Score", ascending=False).reset_index(drop=True)


def get_player_detail(player_name: str, data: dict) -> dict:
    """Return detailed player info for the detail panel."""
    stats = data.get("stats", {}).get(player_name, FALLBACK_PLAYER_STATS.get(player_name, {}))
    history = data.get("player_masters_history", {}).get(player_name, {})

    # Enrich history with year labels
    hist_rows = []
    for year in sorted(history.keys(), reverse=True):
        finish = history[year].get("finish", "–")
        hist_rows.append({"Year": year, "Finish": finish})

    return {
        "stats": stats,
        "masters_history": hist_rows,
        "last_4_sg_t2g": stats.get("last_4_sg_t2g", []),
    }
