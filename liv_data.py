"""
liv_data.py — 2026 LIV Golf season reference data.

LIV_PLAYER_BASELINE captures confirmed 2026 season results
for players on the LIV schedule whose competitive form is
not fully captured by PGA Tour SG metrics.

recent_results: list[int], most-recent-first, 1 = win.
driving_distance_rank: rank within LIV field (lower = longer).
"""
from __future__ import annotations

LIV_PLAYER_BASELINE: dict[str, dict] = {
    "Bryson DeChambeau": {
        "recent_results": [1, 1, 2, 6, 12],
        # W South Africa (most recent), W Singapore,
        # ~T2 Adelaide, ~T6 Riyadh, ~T12 Hong Kong
        "driving_distance_rank": 1,
        "driving_distance_avg": 323.4,
        "notes": (
            "2 consecutive LIV wins March 2026. "
            "Hottest player in world golf. "
            "Back-to-back titles Singapore + SA."
        ),
    },
    "Jon Rahm": {
        "recent_results": [1, 2, 2, 2, 5],
        # W Hong Kong (most recent in sequence), T2 South Africa,
        # T2 Adelaide, T2 Riyadh, T5 Singapore
        "driving_distance_rank": 8,
        "driving_distance_avg": 307.2,
        "notes": (
            "Won Hong Kong. 4 top-2 finishes "
            "in 5 LIV events 2026. Dominant form."
        ),
    },
    "Cameron Smith": {
        "recent_results": [6, 8, 15, 20, 25],
        "driving_distance_rank": 22,
        "driving_distance_avg": 295.1,
        "notes": "Steady top-10 LIV form 2026.",
    },
    "Tyrrell Hatton": {
        "recent_results": [11, 15, 20, 18, 22],
        "driving_distance_rank": 28,
        "driving_distance_avg": 291.3,
        "notes": "T11 Singapore. Moderate LIV form.",
    },
    "Dustin Johnson": {
        "recent_results": [11, 11, 20, 28, 30],
        "driving_distance_rank": 15,
        "driving_distance_avg": 301.8,
        "notes": "T11 South Africa and Singapore.",
    },
    "Sergio Garcia": {
        "recent_results": [35, 35, 40, 30, 28],
        "driving_distance_rank": 25,
        "driving_distance_avg": 293.7,
        "notes": "Poor LIV 2026 form. T35 multiple.",
    },
    "Charl Schwartzel": {
        "recent_results": [24, 30, 35, 40, 42],
        "driving_distance_rank": 35,
        "driving_distance_avg": 285.0,
        "notes": "Back injury. Nearly WD South Africa.",
    },
    "Bubba Watson": {
        "recent_results": [28, 35, 40, 45, 48],
        "driving_distance_rank": 10,
        "driving_distance_avg": 309.5,
        "notes": "Poor form. Cut rate concern.",
    },
    "Carlos Ortiz": {
        "recent_results": [21, 30, 35, 42, 45],
        "driving_distance_rank": 32,
        "driving_distance_avg": 287.4,
        "notes": "T21 Singapore.",
    },
    "Tom McKibbin": {
        "recent_results": [21, 25, 20, 30, 35],
        "driving_distance_rank": 20,
        "driving_distance_avg": 297.2,
        "notes": "Solid young LIV player.",
    },
    "Phil Mickelson": {
        "recent_results": [99, 99, 99, 99, 99],
        "driving_distance_rank": 50,
        "driving_distance_avg": 270.0,
        "notes": "NOT PLAYING — family health matter. WD 2026 Masters.",
    },
}
