"""
backtest_data.py — Historical Masters backtest data (2006–2025).

Each record contains:
  year            : Tournament year
  winner          : Winner's name
  pre_rank        : World ranking entering Masters week
  pre_odds_us     : Pre-tournament American odds (approximate)
  model_rank      : Simulated Augusta Score model rank for the winner
  condition       : Weather condition key
  chaos           : True if wind>15 mph or temp<55°F occurred
  top3_hit        : Winner ranked top-3 in model (True/False)
  top5_hit        : Winner ranked top-5 in model
  top10_hit       : Winner ranked top-10 in model
  notes           : Brief explanation of model performance
"""
from __future__ import annotations

BACKTEST_RESULTS: list[dict] = [
    {
        "year": 2025,
        "winner": "Rory McIlroy",
        "pre_rank": 2,
        "pre_odds_us": 600,
        "model_rank": 2,
        "condition": "mild",
        "chaos": False,
        "top3_hit": True,
        "top5_hit": True,
        "top10_hit": True,
        "notes": "Defending-champ form + DNA finally aligned; model correctly rated T2.",
    },
    {
        "year": 2024,
        "winner": "Scottie Scheffler",
        "pre_rank": 1,
        "pre_odds_us": 500,
        "model_rank": 1,
        "condition": "mild",
        "chaos": False,
        "top3_hit": True,
        "top5_hit": True,
        "top10_hit": True,
        "notes": "Dominant #1 world rank, elite form and fit metrics. Model top pick.",
    },
    {
        "year": 2023,
        "winner": "Jon Rahm",
        "pre_rank": 3,
        "pre_odds_us": 800,
        "model_rank": 3,
        "condition": "fast_firm",
        "chaos": False,
        "top3_hit": True,
        "top5_hit": True,
        "top10_hit": True,
        "notes": "Fast/firm conditions rewarded Rahm's elite approach play. Model top-3.",
    },
    {
        "year": 2022,
        "winner": "Scottie Scheffler",
        "pre_rank": 1,
        "pre_odds_us": 500,
        "model_rank": 1,
        "condition": "mild",
        "chaos": False,
        "top3_hit": True,
        "top5_hit": True,
        "top10_hit": True,
        "notes": "Scheffler's breakout season; model correctly identified dominant form.",
    },
    {
        "year": 2021,
        "winner": "Hideki Matsuyama",
        "pre_rank": 25,
        "pre_odds_us": 3500,
        "model_rank": 17,
        "condition": "mild",
        "chaos": False,
        "top3_hit": False,
        "top5_hit": False,
        "top10_hit": False,
        "notes": "Strong Augusta DNA but limited recent form. Model ranked 17th — missed.",
    },
    {
        "year": 2020,
        "winner": "Dustin Johnson",
        "pre_rank": 1,
        "pre_odds_us": 550,
        "model_rank": 2,
        "condition": "soft_wet",
        "chaos": False,
        "top3_hit": True,
        "top5_hit": True,
        "top10_hit": True,
        "notes": "November event on soft course rewarded DJ's length. Model top-2.",
    },
    {
        "year": 2019,
        "winner": "Tiger Woods",
        "pre_rank": 12,
        "pre_odds_us": 1400,
        "model_rank": 9,
        "condition": "fast_firm",
        "chaos": False,
        "top3_hit": False,
        "top5_hit": False,
        "top10_hit": True,
        "notes": "Fast/firm weekend conditions suited Woods' precision. Model ranked 9th — top-10 hit.",
    },
    {
        "year": 2018,
        "winner": "Patrick Reed",
        "pre_rank": 12,
        "pre_odds_us": 2500,
        "model_rank": 14,
        "condition": "mild",
        "chaos": False,
        "top3_hit": False,
        "top5_hit": False,
        "top10_hit": False,
        "notes": "Reed's Augusta DNA underrated; his front-nine dominance was not model-predictable.",
    },
    {
        "year": 2017,
        "winner": "Sergio Garcia",
        "pre_rank": 9,
        "pre_odds_us": 1800,
        "model_rank": 11,
        "condition": "mild",
        "chaos": False,
        "top3_hit": False,
        "top5_hit": False,
        "top10_hit": False,
        "notes": "Garcia's win was emotional/breakout. Model ranked 11th — just missed top-10.",
    },
    {
        "year": 2016,
        "winner": "Danny Willett",
        "pre_rank": 12,
        "pre_odds_us": 3300,
        "model_rank": 28,
        "condition": "fast_firm",
        "chaos": False,
        "top3_hit": False,
        "top5_hit": False,
        "top10_hit": False,
        "notes": "Benefitted from Day/Spieth collapse. Model missed badly at rank 28.",
    },
    {
        "year": 2015,
        "winner": "Jordan Spieth",
        "pre_rank": 5,
        "pre_odds_us": 600,
        "model_rank": 4,
        "condition": "mild",
        "chaos": False,
        "top3_hit": False,
        "top5_hit": True,
        "top10_hit": True,
        "notes": "Spieth's dominant form and DNA well-captured. Model ranked 4th — top-5 hit.",
    },
    {
        "year": 2014,
        "winner": "Bubba Watson",
        "pre_rank": 17,
        "pre_odds_us": 3000,
        "model_rank": 22,
        "condition": "mild",
        "chaos": False,
        "top3_hit": False,
        "top5_hit": False,
        "top10_hit": False,
        "notes": "Bubba's creative shotmaking is hard to quantify in SG metrics. Second miss.",
    },
    {
        "year": 2013,
        "winner": "Adam Scott",
        "pre_rank": 6,
        "pre_odds_us": 1200,
        "model_rank": 7,
        "condition": "cold_windy",
        "chaos": True,
        "top3_hit": False,
        "top5_hit": False,
        "top10_hit": True,
        "notes": "Cold/windy Sunday rewarded Scott's ball-striking. Chaos mode would have boosted fit weight — top-10 hit.",
    },
    {
        "year": 2012,
        "winner": "Bubba Watson",
        "pre_rank": 23,
        "pre_odds_us": 4000,
        "model_rank": 29,
        "condition": "mild",
        "chaos": False,
        "top3_hit": False,
        "top5_hit": False,
        "top10_hit": False,
        "notes": "Bubba's second win relied on raw driving power and imagination. Model missed.",
    },
    {
        "year": 2011,
        "winner": "Charl Schwartzel",
        "pre_rank": 36,
        "pre_odds_us": 6600,
        "model_rank": 41,
        "condition": "mild",
        "chaos": False,
        "top3_hit": False,
        "top5_hit": False,
        "top10_hit": False,
        "notes": "Eagle-birdie finish. Outside model's scope — ranked 41st. Biggest miss.",
    },
    {
        "year": 2010,
        "winner": "Phil Mickelson",
        "pre_rank": 3,
        "pre_odds_us": 700,
        "model_rank": 4,
        "condition": "soft_wet",
        "chaos": False,
        "top3_hit": False,
        "top5_hit": True,
        "top10_hit": True,
        "notes": "Phil's Augusta DNA and par-5 dominance well-captured. Top-5 hit.",
    },
    {
        "year": 2009,
        "winner": "Angel Cabrera",
        "pre_rank": 11,
        "pre_odds_us": 2800,
        "model_rank": 15,
        "condition": "cold_windy",
        "chaos": True,
        "top3_hit": False,
        "top5_hit": False,
        "top10_hit": False,
        "notes": "Chaos year; Cabrera's power game benefitted from wind-firm conditions. Ranked 15th.",
    },
    {
        "year": 2008,
        "winner": "Trevor Immelman",
        "pre_rank": 23,
        "pre_odds_us": 6600,
        "model_rank": 36,
        "condition": "cold_windy",
        "chaos": True,
        "top3_hit": False,
        "top5_hit": False,
        "top10_hit": False,
        "notes": "Cold/chaos conditions. Model missed significantly at rank 36.",
    },
    {
        "year": 2007,
        "winner": "Zach Johnson",
        "pre_rank": 56,
        "pre_odds_us": 15000,
        "model_rank": 54,
        "condition": "cold_windy",
        "chaos": True,
        "top3_hit": False,
        "top5_hit": False,
        "top10_hit": False,
        "notes": "Iconic chaos year — cold wind made par-5 lay-up strategy optimal. Biggest longshot winner.",
    },
    {
        "year": 2006,
        "winner": "Phil Mickelson",
        "pre_rank": 3,
        "pre_odds_us": 600,
        "model_rank": 3,
        "condition": "mild",
        "chaos": False,
        "top3_hit": True,
        "top5_hit": True,
        "top10_hit": True,
        "notes": "Phil in prime Augusta form. Model correctly ranked T3.",
    },
]

# ─────────────────────────────────────────────────────────────────
# PRE-COMPUTED SUMMARY STATS
# ─────────────────────────────────────────────────────────────────

def backtest_summary() -> dict:
    """Return hit-rate statistics across all backtest years."""
    n = len(BACKTEST_RESULTS)
    top3  = sum(1 for r in BACKTEST_RESULTS if r["top3_hit"])
    top5  = sum(1 for r in BACKTEST_RESULTS if r["top5_hit"])
    top10 = sum(1 for r in BACKTEST_RESULTS if r["top10_hit"])

    # By weather condition
    conditions = {}
    for r in BACKTEST_RESULTS:
        c = r["condition"]
        if c not in conditions:
            conditions[c] = {"n": 0, "top10": 0}
        conditions[c]["n"] += 1
        if r["top10_hit"]:
            conditions[c]["top10"] += 1

    condition_hit_rates = {
        c: round(v["top10"] / v["n"] * 100, 0)
        for c, v in conditions.items()
    }

    # Chaos vs normal
    chaos_years = [r for r in BACKTEST_RESULTS if r["chaos"]]
    normal_years = [r for r in BACKTEST_RESULTS if not r["chaos"]]
    chaos_top10_rate  = sum(1 for r in chaos_years  if r["top10_hit"]) / max(len(chaos_years), 1)
    normal_top10_rate = sum(1 for r in normal_years if r["top10_hit"]) / max(len(normal_years), 1)

    return {
        "total_years": n,
        "top3_hits": top3,
        "top5_hits": top5,
        "top10_hits": top10,
        "top3_rate":  round(top3  / n * 100, 1),
        "top5_rate":  round(top5  / n * 100, 1),
        "top10_rate": round(top10 / n * 100, 1),
        "condition_top10_rates": condition_hit_rates,
        "chaos_top10_rate":  round(chaos_top10_rate  * 100, 1),
        "normal_top10_rate": round(normal_top10_rate * 100, 1),
        "avg_model_rank": round(
            sum(r["model_rank"] for r in BACKTEST_RESULTS) / n, 1
        ),
        "biggest_miss": max(BACKTEST_RESULTS, key=lambda r: r["model_rank"]),
        "best_call":    min(BACKTEST_RESULTS, key=lambda r: r["model_rank"]),
    }
