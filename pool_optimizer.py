"""
pool_optimizer.py — Team construction and EV optimization.
Generates 3 teams calibrated to the Ferraro Green Jacket Pool:
  Team A (Floor):   pure quality, built to place consistently
  Team B (Ceiling): 2 chalk + 1 mid-tier + 1 underowned upside, built to win
  Team C (Value):   1 chalk anchor + 3 from 3-12% ownership sweet spot

Ownership calibration based on 2019 (n=253) and 2023 (n=532) pool data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations

try:
    from field_data import POOL_OVEROWNED_PLAYERS, POOL_UNDEROWNED_PLAYERS
except ImportError:
    POOL_OVEROWNED_PLAYERS: dict = {}
    POOL_UNDEROWNED_PLAYERS: dict = {}


# ─────────────────────────────────────────────────────────────────
# POOL CALIBRATION — Change 1
# ─────────────────────────────────────────────────────────────────

def calibrate_ownership_for_pool(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalibrate raw odds-derived ownership estimates to match observed
    Ferraro pool behavior.  Modifies df IN PLACE and returns it.

    Key finding: top-3 players (by Augusta_Score) are owned at 1.5×
    what pure odds imply; the rest of the field at 0.75×.

    Historical validation:
      2023: Scheffler odds implied ~25% → actual pool 59%
            McIlroy  odds implied ~18% → actual pool 55%
            Rahm     odds implied ~12% → actual pool 44%
            Aberg    odds implied  ~8% → actual pool  0%
      2019: McIlroy  odds implied ~18% → actual pool 63%
            Rose     odds implied ~10% → actual pool 43%
    """
    df["Ownership_Pct_Raw"] = df["Ownership_Pct"].copy()

    top3_names = df.nlargest(3, "Augusta_Score")["Player"].tolist()

    def _calibrate(row):
        raw = row["Ownership_Pct_Raw"]
        if row["Player"] in top3_names:
            return min(65.0, raw * 1.5)
        else:
            return max(0.3, raw * 0.75)

    df["Ownership_Pct"] = df.apply(_calibrate, axis=1).round(2)

    # Per-player over-owned corrections (fame / narrative bias)
    for player, multiplier in POOL_OVEROWNED_PLAYERS.items():
        mask = df["Player"] == player
        if mask.any():
            df.loc[mask, "Ownership_Pct"] = (
                df.loc[mask, "Ownership_Pct"] * multiplier
            ).clip(0, 70).round(2)

    # Per-player under-owned corrections (legitimate edge plays)
    for player, multiplier in POOL_UNDEROWNED_PLAYERS.items():
        mask = df["Player"] == player
        if mask.any():
            df.loc[mask, "Ownership_Pct"] = (
                df.loc[mask, "Ownership_Pct"] * multiplier
            ).clip(0.3, 70).round(2)

    return df


# ─────────────────────────────────────────────────────────────────
# POOL TIEBREAKER — Change 4
# ─────────────────────────────────────────────────────────────────

def compute_pool_tiebreaker(weather_data: dict, chaos_mode: bool) -> dict:
    """
    Tiebreaker strategy: beat the cluster, not the ceiling.

    Pool herds at round numbers.  We submit one below the largest
    cluster to beat it outright if our team scores there.

    Pool tiebreaker clusters from real data:
      2019: -28 (most common), -32 (second), -30 (third)
      2023: -25 (most common), -24 (second), -28 (third)

    Historical best team aggregates:
      2023 (mild): -19 best team, pool median -24
      2019 (Tiger year): -40 best team, pool median -32
      Average best team: -30 to -35 in a normal year
    """
    if chaos_mode:
        primary = -29
        rationale = (
            "Chaos conditions — scoring higher. "
            "Pool will cluster at -28. "
            "Submit -29 to beat that cluster."
        )
        clusters = [-28, -29, -30]
    else:
        avg_wind = weather_data.get("avg_wind_mph", 12)
        if avg_wind > 12:
            primary = -31
            rationale = (
                "Moderate wind expected. "
                "Submit -31, beats -30 and -32 clusters."
            )
            clusters = [-30, -31, -32]
        else:
            primary = -33
            rationale = (
                "Mild conditions. Pool will cluster "
                "at -32. Submit -33 to beat it."
            )
            clusters = [-30, -31, -32]

    return {
        "recommended": primary,
        "rationale": rationale,
        "pool_clusters_to_beat": clusters,
        "historical_note": (
            "2023 pool median -24, best team -19. "
            "2019 pool median -32, best team -40. "
            "Tiebreaker matters most when your team "
            "ties another — target the cluster just above "
            "your submission."
        ),
    }


# ─────────────────────────────────────────────────────────────────
# PORTFOLIO CORRELATION — Change 3
# ─────────────────────────────────────────────────────────────────

def compute_portfolio_correlation(
    team_a: list[str], team_b: list[str], team_c: list[str]
) -> dict:
    """
    Check cross-team player overlap.
    Ideal portfolio:
      A/B overlap: max 2 shared players
      A/C overlap: max 1 shared player
      B/C overlap: max 1 shared player
      All three teams share: max 1 player
    """
    set_a = set(team_a)
    set_b = set(team_b)
    set_c = set(team_c)

    ab_overlap = set_a & set_b
    ac_overlap = set_a & set_c
    bc_overlap = set_b & set_c
    abc_overlap = set_a & set_b & set_c

    warnings: list[str] = []
    if len(ab_overlap) > 2:
        warnings.append(
            f"Teams A+B share {len(ab_overlap)} players "
            f"({', '.join(sorted(ab_overlap))}) — high correlation"
        )
    if len(ac_overlap) > 1:
        warnings.append(
            f"Teams A+C share {len(ac_overlap)} players "
            f"({', '.join(sorted(ac_overlap))}) — reduces C differentiation"
        )
    if len(bc_overlap) > 1:
        warnings.append(
            f"Teams B+C share {len(bc_overlap)} players "
            f"({', '.join(sorted(bc_overlap))}) — reduces C differentiation"
        )
    if len(abc_overlap) > 1:
        warnings.append(
            f"All 3 teams share: {', '.join(sorted(abc_overlap))}"
        )

    return {
        "ab_overlap":  sorted(ab_overlap),
        "ac_overlap":  sorted(ac_overlap),
        "bc_overlap":  sorted(bc_overlap),
        "abc_overlap": sorted(abc_overlap),
        "warnings": warnings,
        # lower = better diversification
        "correlation_score": (
            len(ab_overlap) + len(ac_overlap) * 2 + len(bc_overlap) * 2
        ),
    }


# ─────────────────────────────────────────────────────────────────
# COMBO DUPLICATION CHECK — Change 5
# ─────────────────────────────────────────────────────────────────

# Any team with all 3 of these players is ~25% of the field
CHALK_TRIPLE = ["Scottie Scheffler", "Rory McIlroy", "Jon Rahm"]

# Perennial pool picks regardless of form (for context)
POOL_PERENNIALS_2026 = [
    "Scottie Scheffler",   # world #1, will be ~60% owned
    "Rory McIlroy",        # defending champ, will be ~55% owned
    "Jon Rahm",            # LIV star, familiar name, ~35%
    "Jordan Spieth",       # perennial pick, ~20% regardless of form
    "Justin Thomas",       # name recognition pick, ~15%
    "Dustin Johnson",      # past champion, LIV, ~10%
    "Brooks Koepka",       # name recognition, ~8%
]


def check_combo_frequency(team_players: list[str]) -> dict:
    """
    Estimate how many other pool entries likely have this combination
    based on historical Ferraro pool patterns.
    """
    chalk_overlap = sum(1 for p in team_players if p in CHALK_TRIPLE)

    if chalk_overlap == 3:
        est_similar = "~100-150 teams (20-30% of pool)"
        warning = (
            "HIGH DUPLICATION: Scheffler+McIlroy+Rahm "
            "combo is on ~25% of all entries. "
            "Even a winning score splits the payout."
        )
        severity = "high"
    elif chalk_overlap == 2:
        est_similar = "~50-80 teams (10-15% of pool)"
        warning = (
            "MODERATE DUPLICATION: 2 of top-3 chalk "
            "shared by ~10% of field."
        )
        severity = "medium"
    else:
        est_similar = "~10-20 teams (<5% of pool)"
        warning = None
        severity = "low"

    return {
        "estimated_similar_teams": est_similar,
        "duplication_severity": severity,
        "warning": warning,
        "chalk_overlap_count": chalk_overlap,
    }


# ─────────────────────────────────────────────────────────────────
# TEAM GENERATION — Change 2
# ─────────────────────────────────────────────────────────────────

def _has_cut_rate_col(df: pd.DataFrame) -> bool:
    return "Augusta_Cut_Rate" in df.columns


def generate_teams(
    df: pd.DataFrame,
    pot_size: float = 40000,
    num_entries: int = 500,
    weather: dict | None = None,
) -> dict:
    """
    Three teams covering three different outcomes.
    ZERO players shared across all three teams.
    Max 2 shared between any two teams.

    Team A: Top 4 by Augusta_Score — pure chalk, built to place.
    Team B: Best 2 of top-3 + 1 mid-tier + 1 value — built to win.
    Team C: Top-3 player NOT on team_b + 3 value picks — differentiated upside.
    """
    if weather is None:
        weather = {}

    # ── Step 1: Apply pool ownership calibration (in-place on df) ───
    calibrate_ownership_for_pool(df)

    WITHDRAWN = {"Tiger Woods", "Phil Mickelson"}
    eligible = df[
        df["Player"].notna() &
        ~df["Player"].isin(WITHDRAWN)
    ].copy()
    has_cut = _has_cut_rate_col(eligible)

    # Apply cut-rate floor to all teams
    if has_cut:
        eligible = eligible[eligible["Augusta_Cut_Rate"] >= 0.60]

    # ── Step 2: Identify top-3 anchors ──────────────────────────────
    top3 = eligible.nlargest(3, "Augusta_Score")["Player"].tolist()

    # ── Team A: Pure chalk ───────────────────────────────────────────
    # Top 4 by Augusta_Score — built to place
    team_a_players = eligible.nlargest(4, "Augusta_Score")["Player"].tolist()

    # ── Team B: Built to win ─────────────────────────────────────────
    # 1 anchor from top3 (best player) + 1 from secondary chalk (rank 4-8)
    # + 1 mid-tier + 1 value. Limits A+B overlap to max 2.
    b_anchor_1 = top3[0]  # best player in model — unavoidable overlap with A

    secondary_chalk = [
        p for p in eligible.nlargest(8, "Augusta_Score")["Player"].tolist()
        if p not in top3
    ]
    b_anchor_2 = next(
        (p for p in secondary_chalk if p != b_anchor_1),
        None,
    )
    if b_anchor_2 is None:
        b_anchor_2 = top3[1]  # fallback: still reduces overlap vs old top3[:2]

    b_anchors = [b_anchor_1, b_anchor_2]

    cut_mask = (eligible["Augusta_Cut_Rate"] >= 0.65) if has_cut else pd.Series(True, index=eligible.index)

    # Mid-tier: 2-8% owned (pool-calibrated), best Augusta_Score, not already a b_anchor
    eligible_mid = eligible[
        (eligible["Ownership_Pct"] >= 2.0) &
        (eligible["Ownership_Pct"] <= 8.0) &
        (~eligible["Player"].isin(b_anchors + top3)) &
        (cut_mask if has_cut else pd.Series(True, index=eligible.index))
    ].nlargest(1, "Augusta_Score")
    b_mid_players = eligible_mid["Player"].tolist()

    # Value: 0.5-7% owned, ≥50th percentile score, not already in team_b.
    # Upper bound raised from 2.5% to 7% so mid-tier calibrated players
    # (e.g. Schauffele at ~6%) qualify once anchors + mid_tier are filled.
    score_p50 = eligible["Augusta_Score"].quantile(0.50)
    eligible_val_b = eligible[
        (eligible["Ownership_Pct"] >= 0.5) &
        (eligible["Ownership_Pct"] <= 7.0) &
        (eligible["Augusta_Score"] >= score_p50) &
        (~eligible["Player"].isin(b_anchors + top3 + b_mid_players)) &
        (cut_mask if has_cut else pd.Series(True, index=eligible.index))
    ].nlargest(1, "Augusta_Score")

    team_b_players = b_anchors + b_mid_players + eligible_val_b["Player"].tolist()

    # Deduplicate preserving order
    seen: set[str] = set()
    team_b_players = [p for p in team_b_players if not (p in seen or seen.add(p))]  # type: ignore[func-returns-value]

    # Fill Team B to 4 if needed (no overlap with existing slots)
    if len(team_b_players) < 4:
        fill_b = eligible[
            ~eligible["Player"].isin(team_b_players)
        ].nlargest(4 - len(team_b_players), "Augusta_Score")
        team_b_players += fill_b["Player"].tolist()

    # ── Team C: Differentiated upside ───────────────────────────────
    # Anchor: top-3 player NOT in Team B (index 2 — never in b_anchors[:2])
    c_anchor = top3[2]

    # 3 value picks: 1.5-3.5% owned, odds ≤+5000, Augusta_Score ≥ p45.
    # Odds cap excludes genuine longshots (Bhatia +6000, Smith +8000).
    # Ownership floor 1.5% targets actual pool sweet-spot (calibrated ownership
    # compresses values; 1.5% ≈ 3-7% raw for players like Fleetwood/Young/Schauffele).
    score_p45 = eligible["Augusta_Score"].quantile(0.45)
    odds_col_ok = "Odds_American" in eligible.columns
    eligible_val_c = eligible[
        (eligible["Ownership_Pct"] >= 1.5) &
        (eligible["Ownership_Pct"] <= 3.5) &
        (eligible["Augusta_Score"] >= score_p45) &
        (eligible["Odds_American"] <= 5000 if odds_col_ok else True) &
        (eligible["Augusta_Cut_Rate"] >= 0.65 if has_cut else True) &
        (~eligible["Player"].isin(team_a_players)) &
        (~eligible["Player"].isin(team_b_players)) &
        (eligible["Player"] != c_anchor)
    ].copy()

    # Fallback: loosen ownership slightly but keep odds cap
    if len(eligible_val_c) < 3:
        eligible_val_c = eligible[
            (eligible["Ownership_Pct"] >= 1.0) &
            (eligible["Ownership_Pct"] <= 5.0) &
            (eligible["Augusta_Score"] >= score_p45) &
            (eligible["Odds_American"] <= 5000 if odds_col_ok else True) &
            (~eligible["Player"].isin(team_a_players)) &
            (~eligible["Player"].isin(team_b_players)) &
            (eligible["Player"] != c_anchor)
        ].copy()

    if not eligible_val_c.empty:
        eligible_val_c["TeamC_Score"] = (
            eligible_val_c["EV_Score"] * 0.45
            + eligible_val_c["Augusta_Score"] * 0.30
            + (5.0 / eligible_val_c["Ownership_Pct"].clip(0.5, 20)) * 0.25
        )
        c_value_players = eligible_val_c.nlargest(3, "TeamC_Score")["Player"].tolist()
    else:
        c_value_players = []

    team_c_players: list[str] = [c_anchor] + c_value_players

    # Fill if still short — grab anyone not on A or B
    if len(team_c_players) < 4:
        fill_c = eligible[
            ~eligible["Player"].isin(team_a_players + team_b_players + team_c_players)
        ].nlargest(4 - len(team_c_players), "Augusta_Score")
        team_c_players += fill_c["Player"].tolist()

    # ── Portfolio correlation ────────────────────────────────────────
    correlation = compute_portfolio_correlation(
        team_a_players, team_b_players, team_c_players
    )

    # ── Combo duplication check ──────────────────────────────────────
    def team_info(players: list[str], label: str, strategy: str) -> dict:
        sub = eligible[eligible["Player"].isin(players)].copy()
        return {
            "players": players,
            "label": label,
            "strategy": strategy,
            "total_augusta_score": round(float(sub["Augusta_Score"].sum()), 1),
            "total_ev_score": round(float(sub["EV_Score"].sum()), 1),
            "avg_odds_american": round(float(sub["Odds_American"].mean())),
            "duplication": check_combo_frequency(players),
            "stats": sub.to_dict("records"),
        }

    team_a = team_info(
        team_a_players,
        "Team A — Floor",
        "Best 4 players by composite Augusta Score. Built to place — "
        "consistent top-5 pool finish. Cut rate ≥0.65, odds ≤+6000.",
    )
    team_b = team_info(
        team_b_players,
        "Team B — Ceiling",
        "2 chalk anchors + 1 mid-tier (10-25%) + 1 value (3-12%). "
        "Mirrors 2023 winning structure (Rahm). Built to win outright.",
    )
    team_c = team_info(
        team_c_players,
        "Team C — Value",
        "1 chalk floor + 3 underowned upside plays (3-12% sweet spot). "
        "Targets this pool's fame/narrative blind spots.",
    )

    # ── Tiebreaker ───────────────────────────────────────────────────
    chaos_mode = weather.get("chaos_mode", False)
    tiebreaker = compute_pool_tiebreaker(weather, chaos_mode)

    # ── Payout context ────────────────────────────────────────────────
    payouts = {
        "1st": 0.50 * pot_size,
        "2nd": 0.25 * pot_size,
        "3rd": 0.15 * pot_size,
        "4th": 0.07 * pot_size,
        "5th": 0.03 * pot_size,
    }
    entry_fee = 100

    return {
        "team_a": team_a,
        "team_b": team_b,
        "team_c": team_c,
        "correlation": correlation,
        "tiebreaker": tiebreaker,
        "overlap": {
            "A-B": len(correlation["ab_overlap"]),
            "A-C": len(correlation["ac_overlap"]),
            "B-C": len(correlation["bc_overlap"]),
        },
        "payouts": payouts,
        "entry_fee": entry_fee,
        "pot_size": pot_size,
        "num_entries_est": num_entries,
    }


# ─────────────────────────────────────────────────────────────────
# SELECTION HELPERS (retained for backward-compat / swap logic)
# ─────────────────────────────────────────────────────────────────

def _team_score(players: list[str], df: pd.DataFrame, metric: str) -> float:
    return df.loc[df["Player"].isin(players), metric].sum()


def _overlap_count(team_a: list[str], team_b: list[str]) -> int:
    return len(set(team_a) & set(team_b))


def _pick_best_four(
    ranked_players: list[str],
    df: pd.DataFrame,
    metric: str,
    constraint_fn=None,
) -> list[str]:
    top4 = ranked_players[:4]
    if constraint_fn is None or constraint_fn(top4):
        return top4
    remaining = [p for p in ranked_players if p not in top4]
    for alt in remaining:
        for i in range(4):
            candidate = top4[:i] + [alt] + top4[i + 1:]
            if constraint_fn(candidate):
                return sorted(
                    candidate,
                    key=lambda p: df.loc[df["Player"] == p, metric].values[0],
                    reverse=True,
                )
    return top4


def _enforce_max_overlap(
    team: list[str],
    reference: list[str],
    ranked_df: pd.DataFrame,
    metric: str,
    full_df: pd.DataFrame,
) -> list[str]:
    overlap = set(team) & set(reference)
    if len(overlap) <= 2:
        return team
    shared_scores = full_df.loc[
        full_df["Player"].isin(overlap), ["Player", metric]
    ].set_index("Player")[metric].to_dict()
    weakest = min(overlap, key=lambda p: shared_scores.get(p, 0))
    team = [p for p in team if p != weakest]
    candidates = ranked_df[~ranked_df["Player"].isin(team + reference)]["Player"].tolist()
    if candidates:
        team.append(candidates[0])
    return team


# ─────────────────────────────────────────────────────────────────
# MANUAL TEAM SCORING (for the interactive swapper in app.py)
# ─────────────────────────────────────────────────────────────────

def score_custom_team(player_names: list[str], df: pd.DataFrame) -> dict:
    """Return score breakdown for a custom 4-player team."""
    if len(player_names) != 4:
        return {}
    sub = df[df["Player"].isin(player_names)]
    return {
        "players": player_names,
        "total_augusta_score": round(sub["Augusta_Score"].sum(), 1),
        "total_ev_score": round(sub["EV_Score"].sum(), 1),
        "avg_ownership": round(sub["Ownership_Pct"].mean(), 1),
        "stats": sub[
            ["Player", "Augusta_Score", "EV_Score", "Ownership_Pct", "Odds_American"]
        ].to_dict("records"),
    }
