"""
tiebreaker.py — Tiebreaker regression model.
Predicts top-4 cumulative under-par based on weather condition.
"""

import numpy as np
from fetch_data import AUGUSTA_HISTORY


# ─────────────────────────────────────────────────────────────────
# HISTORICAL WEATHER → TOP-4 TOTALS MAPPING
# ─────────────────────────────────────────────────────────────────

# From AUGUSTA_HISTORY top4_total values, grouped by condition
# november_soft = 2020 anomaly (played in November, softer/longer course)
CONDITION_HISTORY = {
    "soft_wet":          [AUGUSTA_HISTORY[2025]["top4_total"]],   # -39
    "mild":              [AUGUSTA_HISTORY[2024]["top4_total"],     # -26
                          AUGUSTA_HISTORY[2023]["top4_total"],     # -35
                          AUGUSTA_HISTORY[2015]["top4_total"]],    # -58
    "warm_calm":         [AUGUSTA_HISTORY[2022]["top4_total"]],   # -27
    "fast_firm":         [AUGUSTA_HISTORY[2021]["top4_total"]],   # -33
    "november_soft":     [AUGUSTA_HISTORY[2020]["top4_total"]],   # -62 (outlier)
    "rain_thunderstorms":[AUGUSTA_HISTORY[2019]["top4_total"],     # -49
                          AUGUSTA_HISTORY[2018]["top4_total"]],    # -53
    "cold_wind_rain":    [AUGUSTA_HISTORY[2017]["top4_total"]],   # -29
    "cold_windy":        [AUGUSTA_HISTORY[2016]["top4_total"]],   # -10
}

# Weights for november_soft outlier (downweighted per spec)
CONDITION_WEIGHTS = {
    "soft_wet": 1.0, "mild": 1.0, "warm_calm": 1.0,
    "fast_firm": 1.0, "november_soft": 0.3,   # outlier
    "rain_thunderstorms": 1.0, "cold_wind_rain": 1.0, "cold_windy": 1.0,
}

# Friendly display names
CONDITION_LABELS = {
    "soft_wet": "Soft / Wet Course",
    "mild": "Mild & Calm",
    "warm_calm": "Warm & Calm",
    "fast_firm": "Fast / Firm Course",
    "november_soft": "Soft (November-style)",
    "rain_thunderstorms": "Rain / Thunderstorms",
    "cold_wind_rain": "Cold, Windy & Rainy",
    "cold_windy": "Cold & Windy",
}


# ─────────────────────────────────────────────────────────────────
# GLOBAL DISTRIBUTION (all years, weighted)
# ─────────────────────────────────────────────────────────────────

def _build_global_distribution() -> tuple[np.ndarray, np.ndarray]:
    """Return (values, weights) array from all historical data."""
    vals, wts = [], []
    for cond, top4_list in CONDITION_HISTORY.items():
        w = CONDITION_WEIGHTS.get(cond, 1.0)
        for v in top4_list:
            vals.append(v)
            wts.append(w)
    return np.array(vals, dtype=float), np.array(wts, dtype=float)


GLOBAL_VALS, GLOBAL_WEIGHTS = _build_global_distribution()
GLOBAL_MEAN = float(np.average(GLOBAL_VALS, weights=GLOBAL_WEIGHTS))
GLOBAL_STD = float(
    np.sqrt(np.average((GLOBAL_VALS - GLOBAL_MEAN) ** 2, weights=GLOBAL_WEIGHTS))
)


# ─────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────

def predict_tiebreaker(weather_data: dict) -> dict:
    """
    Given weather data dict (from fetch_data.fetch_weather),
    return a tiebreaker prediction dict.

    Returns:
        condition: classified condition string
        condition_label: friendly name
        predicted_median: recommended tiebreaker number
        predicted_range: (low, high) 80% interval
        confidence: "high" | "medium" | "low"
        historical_data: list of {year, condition, top4_total}
        interpretation: text description
    """
    condition = weather_data.get("condition", "mild")

    # Get condition-specific historical values
    cond_vals = CONDITION_HISTORY.get(condition, [])
    w_cond = CONDITION_WEIGHTS.get(condition, 1.0)

    # Blend condition-specific data with global (Bayesian shrinkage)
    # More data = less shrinkage toward global mean
    n_specific = len(cond_vals)
    shrinkage = 1.0 / (1.0 + n_specific)  # 0.5 for 1 obs, 0.33 for 2, etc.

    if cond_vals:
        specific_mean = float(np.average(cond_vals, weights=[w_cond] * n_specific))
        specific_std = float(np.std(cond_vals)) if n_specific > 1 else GLOBAL_STD * 0.8
    else:
        specific_mean = GLOBAL_MEAN
        specific_std = GLOBAL_STD
        shrinkage = 0.7

    blended_mean = specific_mean * (1 - shrinkage) + GLOBAL_MEAN * shrinkage

    # Uncertainty: blend specific std with global
    blended_std = specific_std * (1 - shrinkage) + GLOBAL_STD * shrinkage

    # 80% confidence interval (z ≈ ±1.28)
    lo = round(blended_mean - 1.28 * blended_std)
    hi = round(blended_mean + 1.28 * blended_std)
    median = round(blended_mean)

    # Confidence tier
    if n_specific >= 3:
        confidence = "high"
    elif n_specific >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    # Nearest integer below median (tiebreaker should be conservative)
    recommended = median - 1 if abs(median - blended_mean) < 0.5 else median

    # Historical data for display
    historical = []
    for yr in sorted(AUGUSTA_HISTORY.keys(), reverse=True):
        yr_data = AUGUSTA_HISTORY[yr]
        historical.append({
            "year": yr,
            "winner": yr_data["winner"],
            "condition": yr_data["weather"],
            "top4_total": yr_data["top4_total"],
        })

    # Weather-player impact analysis
    tournament_days = weather_data.get("tournament_days", {})
    avg_wind = _avg_stat(tournament_days, "wind_max")
    total_precip = _sum_stat(tournament_days, "precip")

    impacts = _build_player_impacts(condition, avg_wind, total_precip)

    interp = _interpretation(condition, blended_mean, avg_wind, total_precip)

    return {
        "condition": condition,
        "condition_label": CONDITION_LABELS.get(condition, condition),
        "predicted_median": median,
        "predicted_range": (lo, hi),
        "recommended_tiebreaker": recommended,
        "confidence": confidence,
        "blended_mean": round(blended_mean, 1),
        "blended_std": round(blended_std, 1),
        "historical_data": historical,
        "condition_history": cond_vals,
        "avg_wind_mph": round(avg_wind, 1),
        "total_precip_mm": round(total_precip, 1),
        "player_weather_impacts": impacts,
        "interpretation": interp,
    }


def _avg_stat(days: dict, key: str) -> float:
    vals = [d.get(key, 0) or 0 for d in days.values()]
    return float(np.mean(vals)) if vals else 0.0


def _sum_stat(days: dict, key: str) -> float:
    return sum(d.get(key, 0) or 0 for d in days.values())


def _build_player_impacts(condition: str, avg_wind: float, total_precip: float) -> list[dict]:
    """Return player weather impact notes for key players."""
    from fetch_data import FALLBACK_PLAYER_STATS

    impacts = []
    for name, stats in FALLBACK_PLAYER_STATS.items():
        dd = stats.get("driving_distance", 295)
        sg_t2g = stats.get("sg_t2g", 0.8)
        par5 = stats.get("par5_scoring", 4.65)

        if condition in ("soft_wet", "november_soft"):
            if dd > 310:
                effect = "BENEFITS — long hitter, gains extra on par-5s"
                direction = "positive"
            elif par5 < 4.58:
                effect = "BENEFITS — elite par-5 scorer in soft conditions"
                direction = "positive"
            else:
                effect = "Neutral to slightly negative — shorter hitter"
                direction = "neutral"
        elif condition in ("cold_windy", "cold_wind_rain", "rain_thunderstorms"):
            if sg_t2g > 1.5:
                effect = "BENEFITS — elite ball-striker handles wind"
                direction = "positive"
            elif dd < 290:
                effect = "HURT — short hitter penalized in wind"
                direction = "negative"
            else:
                effect = "Neutral — average wind performance"
                direction = "neutral"
        elif condition == "fast_firm":
            sg_app = stats.get("sg_app", 0.5)
            if sg_app > 0.9:
                effect = "BENEFITS — elite iron player on firm greens"
                direction = "positive"
            else:
                effect = "Neutral"
                direction = "neutral"
        else:
            effect = "Neutral — mild conditions suit all players"
            direction = "neutral"

        impacts.append({
            "player": name,
            "effect": effect,
            "direction": direction,
            "drive_dist": dd,
            "sg_t2g": sg_t2g,
        })

    return sorted(impacts, key=lambda x: {"positive": 0, "neutral": 1, "negative": 2}[x["direction"]])


def _interpretation(condition: str, blended_mean: float, avg_wind: float, total_precip: float) -> str:
    cond_label = CONDITION_LABELS.get(condition, condition)
    direction = "lower" if blended_mean < -35 else "higher"
    driver = ""
    if condition in ("soft_wet",):
        driver = "Soft fairways reduce roll and lower scores — expect low numbers."
    elif condition in ("cold_windy", "cold_wind_rain"):
        driver = "Wind and cold suppress scoring — the winner will likely be higher than average."
    elif condition == "fast_firm":
        driver = "Firm, fast greens put a premium on precision — mid-range aggregate expected."
    elif condition == "november_soft":
        driver = "November-like softness (see 2020) — scores can run very low. Treat as outlier."
    else:
        driver = "Mild conditions — historical average scoring expected."

    return (
        f"Forecast condition: **{cond_label}** "
        f"(avg wind {avg_wind:.0f} mph, {total_precip:.0f}mm total precip). "
        f"{driver} "
        f"Predicted top-4 aggregate: **{blended_mean:.0f}** under par. "
        f"Scores tend to run {direction} under these conditions."
    )
