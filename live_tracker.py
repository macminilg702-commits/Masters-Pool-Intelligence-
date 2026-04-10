"""
live_tracker.py — Pool file parsing, live score fetching, and standings computation.
"""
from __future__ import annotations

import io
import logging
import re
from typing import Optional

import pandas as pd
import numpy as np
import requests

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# DEMO DATA (shown before pool file is uploaded)
# ─────────────────────────────────────────────────────────────────

DEMO_ENTRIES = pd.DataFrame([
    {"Entry_Name": "John Smith #1",    "P1": "Scottie Scheffler", "P2": "Rory McIlroy",       "P3": "Collin Morikawa",  "P4": "Tommy Fleetwood",   "Tiebreaker": -38},
    {"Entry_Name": "John Smith #2",    "P1": "Scottie Scheffler", "P2": "Xander Schauffele",  "P3": "Jon Rahm",         "P4": "Cameron Young",     "Tiebreaker": -35},
    {"Entry_Name": "John Smith #3",    "P1": "Rory McIlroy",      "P2": "Ludvig Aberg",       "P3": "Viktor Hovland",   "P4": "Joaquin Niemann",   "Tiebreaker": -32},
    {"Entry_Name": "Jane Doe #1",      "P1": "Scottie Scheffler", "P2": "Jon Rahm",            "P3": "Brooks Koepka",    "P4": "Jordan Spieth",     "Tiebreaker": -36},
    {"Entry_Name": "Jane Doe #2",      "P1": "Rory McIlroy",      "P2": "Bryson DeChambeau",  "P3": "Hideki Matsuyama", "P4": "Patrick Cantlay",   "Tiebreaker": -34},
    {"Entry_Name": "Mike Johnson",     "P1": "Scottie Scheffler", "P2": "Collin Morikawa",    "P3": "Viktor Hovland",   "P4": "Robert MacIntyre",  "Tiebreaker": -33},
    {"Entry_Name": "Sarah Williams",   "P1": "Xander Schauffele", "P2": "Jon Rahm",            "P3": "Tommy Fleetwood",  "P4": "Cameron Smith",     "Tiebreaker": -30},
    {"Entry_Name": "Bob Martinez",     "P1": "Ludvig Aberg",      "P2": "Collin Morikawa",    "P3": "Matt Fitzpatrick", "P4": "Will Zalatoris",    "Tiebreaker": -29},
    {"Entry_Name": "Alice Chen",       "P1": "Rory McIlroy",      "P2": "Bryson DeChambeau",  "P3": "Brooks Koepka",    "P4": "Cameron Young",     "Tiebreaker": -40},
    {"Entry_Name": "Tom Wilson",       "P1": "Scottie Scheffler", "P2": "Jordan Spieth",      "P3": "Patrick Cantlay",  "P4": "Tony Finau",        "Tiebreaker": -31},
])

# Placeholder live scores for demo mode
DEMO_LIVE_SCORES = {
    "Scottie Scheffler": -8,   "Rory McIlroy": -6,    "Ludvig Aberg": -5,
    "Jon Rahm": -4,            "Bryson DeChambeau": -3, "Xander Schauffele": -5,
    "Collin Morikawa": -7,     "Tommy Fleetwood": -3,  "Matt Fitzpatrick": -2,
    "Brooks Koepka": -2,       "Jordan Spieth": -3,    "Hideki Matsuyama": -2,
    "Cameron Smith": -1,       "Viktor Hovland": -4,   "Shane Lowry": -1,
    "Sepp Straka": 0,          "Will Zalatoris": -1,   "Min Woo Lee": 1,
    "Tony Finau": 0,           "Patrick Cantlay": -2,  "Joaquin Niemann": -2,
    "Cameron Young": -3,       "Keegan Bradley": 1,    "Corey Conners": 0,
    "Robert MacIntyre": -2,    "Jason Day": 2,         "Adam Scott": 3,
    "Justin Thomas": -1,       "Patrick Reed": 2,      "Tom Kim": 3,
}


# ─────────────────────────────────────────────────────────────────
# FILE PARSING
# ─────────────────────────────────────────────────────────────────

REQUIRED_COLS = {"Entry_Name", "P1", "P2", "P3", "P4"}
COLUMN_ALIASES = {
    "entry": "Entry_Name", "name": "Entry_Name", "entrant": "Entry_Name",
    "player1": "P1", "player 1": "P1", "pick1": "P1", "pick 1": "P1",
    "player2": "P2", "player 2": "P2", "pick2": "P2", "pick 2": "P2",
    "player3": "P3", "player 3": "P3", "pick3": "P3", "pick 3": "P3",
    "player4": "P4", "player 4": "P4", "pick4": "P4", "pick 4": "P4",
    "tiebreaker": "Tiebreaker", "tb": "Tiebreaker", "tie": "Tiebreaker",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names using aliases."""
    renamed = {}
    for col in df.columns:
        normalized = COLUMN_ALIASES.get(col.lower().strip(), col)
        renamed[col] = normalized
    return df.rename(columns=renamed)


def parse_pool_entries(uploaded_file) -> tuple[pd.DataFrame, str]:
    """
    Parse uploaded pool entries file (CSV, XLSX, or pasted text).
    Returns (DataFrame, error_message). error_message is "" on success.
    """
    if uploaded_file is None:
        return DEMO_ENTRIES.copy(), ""

    try:
        name = getattr(uploaded_file, "name", "upload")
        if name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(uploaded_file)
        else:
            # Try CSV / TSV / space-separated
            content = uploaded_file.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            # Detect delimiter
            delim = "," if content.count(",") > content.count("\t") else "\t"
            df = pd.read_csv(io.StringIO(content), sep=delim)
    except Exception as e:
        return DEMO_ENTRIES.copy(), f"Parse error: {e}"

    df = _normalize_columns(df)
    df.columns = df.columns.str.strip()

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        return DEMO_ENTRIES.copy(), (
            f"Missing columns: {missing}. "
            f"Expected: Entry_Name, P1, P2, P3, P4 (+ optional Tiebreaker). "
            f"Found: {list(df.columns)}"
        )

    if "Tiebreaker" not in df.columns:
        df["Tiebreaker"] = 0

    df["Tiebreaker"] = pd.to_numeric(df["Tiebreaker"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["Entry_Name", "P1"])
    return df.reset_index(drop=True), ""


# ─────────────────────────────────────────────────────────────────
# LIVE SCORE FETCHING
# ─────────────────────────────────────────────────────────────────

def fetch_live_scores() -> tuple[dict[str, int], str]:
    """
    Attempt to fetch live Masters leaderboard from ESPN API.
    Returns (scores_dict, source_label).
    scores_dict: {player_name: score_vs_par}
    """
    scores, _detail, source = fetch_live_data()
    return scores, source


def fetch_live_data() -> tuple[dict[str, int], dict[str, dict], str]:
    """
    Fetch live Masters data from ESPN API.
    Returns (scores, detail, source_label) where:
      scores  — {player_name: total_score_to_par (int)}
      detail  — {player_name: {score, today, thru, position, round, state}}
    """
    result = _try_espn_leaderboard_full()
    if result:
        scores, detail = result
        return scores, detail, "ESPN (live)"

    # Fallback — demo data, no detail
    demo = DEMO_LIVE_SCORES.copy()
    demo_detail = {
        name: {"score": s, "today": "–", "thru": "–", "position": "–", "round": 1, "state": "pre"}
        for name, s in demo.items()
    }
    return demo, demo_detail, "Demo data (tournament not active)"


def _try_espn_leaderboard_full() -> tuple[dict[str, int], dict[str, dict]] | None:
    """
    Fetch from ESPN's golf leaderboard API.
    Returns (scores_dict, detail_dict) or None on failure.

    ESPN returns competitor.score as a dict {"value": float, "displayValue": str},
    NOT a plain string — the old code broke on this.  Total-to-par lives in
    competitor.statistics[0] where name == "scoreToPar".
    """
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/golf/leaderboard"
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return None

        data = resp.json()
        events = data.get("events", [])

        # Find Masters tournament
        masters_event = None
        for evt in events:
            if "masters" in evt.get("name", "").lower():
                masters_event = evt
                break
        if not masters_event and events:
            masters_event = events[0]
        if not masters_event:
            return None

        scores: dict[str, int] = {}
        detail: dict[str, dict] = {}

        for competitor in masters_event.get("competitions", [{}])[0].get("competitors", []):
            name = competitor.get("athlete", {}).get("displayName", "")
            if not name:
                continue

            # ── Total score-to-par ─────────────────────────────────
            score = 0
            for stat in competitor.get("statistics", []):
                if stat.get("name") == "scoreToPar":
                    try:
                        score = int(stat.get("value", 0))
                    except (TypeError, ValueError):
                        score = 0
                    break

            # ── Status fields ──────────────────────────────────────
            status    = competitor.get("status", {})
            state     = status.get("type", {}).get("state", "")   # "pre"/"in"/"post"
            position  = status.get("position", {}).get("displayName", "–")
            period    = int(status.get("period", 1))
            today_raw = status.get("todayDetail", "")             # e.g. "-5(F)", "+2(11)", "E(F)"

            # Parse today / thru from todayDetail
            # e.g. "-5(F)" → today="-5", thru="F"
            #      "+2(11)"→ today="+2", thru="11"  (active round)
            #      "E(F)"  → today="E",  thru="F"
            # todayDetail is populated even when state="pre" (shows last completed round).
            today_str = "–"
            thru_str  = "–"
            if today_raw:
                m = re.match(r'^([+\-]?\d+|E)\((\w+)\)$', today_raw)
                if m:
                    td, th = m.group(1), m.group(2)
                    today_str = "E" if td == "E" else td
                    thru_str  = "F" if th == "F" else th
                else:
                    today_str = today_raw  # leave raw if format differs

            scores[name] = score
            detail[name] = {
                "score":    score,
                "today":    today_str,
                "thru":     thru_str,
                "position": position,
                "round":    period,
                "state":    state,
            }

        return (scores, detail) if scores else None

    except Exception as e:
        logger.debug(f"ESPN leaderboard fetch failed: {e}")
        return None


def _try_masters_scrape() -> dict[str, int] | None:
    """Fallback: scrape masters.com leaderboard."""
    try:
        url = "https://www.masters.com/en_US/scores/index.html"
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return None

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")

        scores = {}
        # Masters.com leaderboard structure (subject to change)
        rows = soup.select(".leaderboard-table tr, .player-row, [data-player-name]")
        for row in rows:
            name_el = row.select_one(".player-name, .name, [data-name]")
            score_el = row.select_one(".score, .total, [data-score]")
            if name_el and score_el:
                name = name_el.get_text(strip=True)
                score_text = score_el.get_text(strip=True).replace("E", "0")
                try:
                    score = int(score_text.replace("+", ""))
                    scores[name] = score
                except ValueError:
                    pass

        return scores if len(scores) > 5 else None
    except Exception as e:
        logger.debug(f"Masters.com scrape failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# STANDINGS COMPUTATION
# ─────────────────────────────────────────────────────────────────

def compute_standings(
    entries: pd.DataFrame,
    live_scores: dict[str, int],
    user_entry_names: list[str] | None = None,
    pot_size: float = 40000,
) -> pd.DataFrame:
    """
    Compute pool standings from entries and live scores.
    Returns sorted DataFrame with payout projections.
    """
    payouts_pct = {1: 0.50, 2: 0.25, 3: 0.15, 4: 0.07, 5: 0.03}

    rows = []
    for _, row in entries.iterrows():
        p1_score = live_scores.get(row["P1"], 0)
        p2_score = live_scores.get(row["P2"], 0)
        p3_score = live_scores.get(row["P3"], 0)
        p4_score = live_scores.get(row["P4"], 0)
        total = p1_score + p2_score + p3_score + p4_score

        rows.append({
            "Entry_Name": row["Entry_Name"],
            "P1": row["P1"],  "P1_Score": p1_score,
            "P2": row["P2"],  "P2_Score": p2_score,
            "P3": row["P3"],  "P3_Score": p3_score,
            "P4": row["P4"],  "P4_Score": p4_score,
            "Total": total,
            "Tiebreaker": row.get("Tiebreaker", 0),
            "Is_User": row["Entry_Name"] in (user_entry_names or []),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Sort: by Total ascending (lower = better in golf), then tiebreaker
    df = df.sort_values(["Total", "Tiebreaker"], ascending=[True, False])
    df["Rank"] = range(1, len(df) + 1)

    # Handle ties
    df["Rank"] = df.groupby("Total")["Rank"].transform("min")

    # Projected payout
    def _payout(rank: int) -> float:
        return payouts_pct.get(int(rank), 0) * pot_size

    df["Proj_Payout"] = df["Rank"].apply(_payout)

    # Reorder columns
    cols = [
        "Rank", "Entry_Name", "Is_User",
        "P1", "P1_Score", "P2", "P2_Score",
        "P3", "P3_Score", "P4", "P4_Score",
        "Total", "Tiebreaker", "Proj_Payout",
    ]
    return df[cols].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# SCENARIO ANALYSIS
# ─────────────────────────────────────────────────────────────────

def scenario_analysis(
    entry_name: str,
    standings: pd.DataFrame,
    live_scores: dict[str, int],
) -> dict:
    """
    For a given entry, determine what needs to happen for it to win.
    Returns a dict with analysis.
    """
    if standings.empty or entry_name not in standings["Entry_Name"].values:
        return {"error": "Entry not found"}

    entry_row = standings[standings["Entry_Name"] == entry_name].iloc[0]
    current_rank = entry_row["Rank"]
    current_total = entry_row["Total"]

    # Current leader
    leader = standings.iloc[0]
    deficit = current_total - leader["Total"]

    players = [entry_row["P1"], entry_row["P2"], entry_row["P3"], entry_row["P4"]]
    player_scores = {p: live_scores.get(p, 0) for p in players}

    # What score needed to win outright
    leader_total = leader["Total"]
    needed_improvement = leader_total - 1 - current_total  # need to beat leader by 1

    in_money = current_rank <= 5
    payout_positions_away = max(0, current_rank - 5)

    return {
        "entry_name": entry_name,
        "current_rank": int(current_rank),
        "current_total": int(current_total),
        "deficit_from_leader": int(deficit),
        "needed_to_win": int(needed_improvement),
        "in_money": in_money,
        "payout_positions_away": int(payout_positions_away),
        "player_scores": player_scores,
        "leader_name": leader["Entry_Name"],
        "leader_total": int(leader["Total"]),
        "analysis": _scenario_text(current_rank, deficit, players, player_scores, needed_improvement),
    }


def _scenario_text(rank: int, deficit: int, players: list, scores: dict, needed: int) -> str:
    best_player = min(players, key=lambda p: scores.get(p, 0))
    worst_player = max(players, key=lambda p: scores.get(p, 0))

    if rank == 1:
        return "Currently leading. Protect the lead — no change needed."
    elif rank <= 3:
        return (
            f"Currently {rank}{_ordinal_suffix(rank)} place, {abs(deficit)} back. "
            f"{best_player} is your best scorer. Needs {abs(needed)} total improvement to take the lead."
        )
    elif rank <= 5:
        return (
            f"In the money at {rank}{_ordinal_suffix(rank)} place. "
            f"{abs(deficit)} back of the lead. "
            f"Focus on {best_player} continuing strong play."
        )
    else:
        return (
            f"{rank}{_ordinal_suffix(rank)} place — outside payout positions. "
            f"Need {abs(needed)} strokes of improvement to lead. "
            f"{worst_player} needs to improve the most."
        )


def _ordinal_suffix(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
