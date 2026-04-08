"""
player_context.py — Current-season context for 2026 Masters.
Injury status, pre-tournament schedule, and player notes.
Imported by score_engine.py and app.py.
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────────
# INJURY / HEALTH STATUS
# Options: Healthy | Minor Concern | Significant Concern | Unknown
# ─────────────────────────────────────────────────────────────────

INJURY_STATUS: dict[str, str] = {
    # Confirmed status
    "Collin Morikawa":      "Back Concern",          # Back spasms; WD Players + WD Valero; "day by day" at Masters; no competitive reps since March 12
    "Tiger Woods":          "Significant Concern",  # Back surgery Oct 2025; reportedly out
    "Scottie Scheffler":    "Healthy",               # Won AmEx Jan 2026; 3 top-5s; Houston WD = rest, not injury
    "Will Zalatoris":       "Minor Concern",         # Managing ongoing lumbar issues
    "Brooks Koepka":        "Minor Concern",         # Persistent knee/hip history; limited LIV rounds
    "Rory McIlroy":         "Healthy",               # Defending champion; no reported issues
    "Cameron Young":        "Healthy",               # Won The Players; healthy and in form
    "Jon Rahm":             "Healthy",               # No reported issues
    "Ludvig Aberg":         "Healthy",               # No reported issues
    "Xander Schauffele":    "Healthy",               # No reported issues
    "Tommy Fleetwood":      "Healthy",               # No reported issues
    "Bryson DeChambeau":    "Healthy",               # No reported issues
    "Jordan Spieth":        "Healthy",               # No reported issues
    "Hideki Matsuyama":     "Healthy",               # No reported issues
    "Justin Rose":          "Healthy",               # No reported issues
    "Viktor Hovland":       "Healthy",               # No reported issues
    "Patrick Cantlay":      "Healthy",               # No reported issues
    "Justin Thomas":        "Healthy",               # No reported issues
    "Matt Fitzpatrick":     "Healthy",               # No reported issues
    "Dustin Johnson":       "Healthy",               # No reported issues
    "Sungjae Im":           "Healthy",               # No reported issues
    "Cameron Smith":        "Healthy",               # No reported issues
    "Shane Lowry":          "Healthy",               # No reported issues
    "Patrick Reed":         "Unknown",               # Limited public info
    "Keegan Bradley":       "Unknown",               # Ryder Cup captain; limited starts
    "Adam Scott":           "Unknown",               # Limited schedule info
    "Anthony Kim":          "Unknown",               # Recent return; limited data
    "Phil Mickelson":       "Withdrawn",             # WD 2026 Masters — family health matter
    "Charl Schwartzel":     "Back Concern",          # Back issue limiting LIV schedule
}

INJURY_MULTIPLIERS: dict[str, float] = {
    "Healthy":              1.00,
    "Minor Concern":        0.92,
    "Back Concern":         0.88,
    "Significant Concern":  0.80,
    "Unknown":              0.95,
    "Withdrawn":            0.00,
}

# Player-specific multiplier overrides (checked before status-based lookup).
# Use when a player's situation is more nuanced than a single status tier.
PLAYER_INJURY_OVERRIDES: dict[str, float] = {
    "Collin Morikawa": 0.84,
    # Back spasms; WD Players after 1 hole, WD Valero entirely.
    # "Some shots I can't hit right now." No play since March 12.
    # Worse than generic "Back Concern" (0.88) — downgraded to 0.84.
    "Patrick Reed": 0.88,
    # LIV 2026 poor form; missed cuts, no top-5s. Status "Unknown" maps to 0.95
    # but form context warrants downgrade to match "Back Concern" tier.
    "Charl Schwartzel": 0.88,
    # Back issues — nearly WD South Africa Open. INJURY_STATUS = "Back Concern"
    # already maps to 0.88; explicit override ensures correct value regardless.
}


def get_injury_multiplier(player_name: str) -> float:
    if player_name in PLAYER_INJURY_OVERRIDES:
        return PLAYER_INJURY_OVERRIDES[player_name]
    status = INJURY_STATUS.get(player_name, "Healthy")
    return INJURY_MULTIPLIERS.get(status, 1.00)


def get_injury_status(player_name: str) -> str:
    return INJURY_STATUS.get(player_name, "Healthy")


# ─────────────────────────────────────────────────────────────────
# PRE-MASTERS SCHEDULE TRACKER
# Valero Texas Open: March 26–29, 2026  (TPC San Antonio)
# Houston Open:      March 30–Apr 2, 2026 (Memorial Park)
# Players who played NEITHER event get a form penalty
# ─────────────────────────────────────────────────────────────────

PRETOURNAMENT_SCHEDULE: dict[str, dict[str, bool]] = {
    "Scottie Scheffler":    {"valero": False, "houston": True},   # Played Houston as tune-up
    "Rory McIlroy":         {"valero": False, "houston": False},  # Skipped; resting as defending champ
    "Jon Rahm":             {"valero": False, "houston": False},  # LIV schedule; no PGA tune-ups
    "Collin Morikawa":      {"valero": False, "houston": False},  # WD The Players; rest/recovery
    "Xander Schauffele":    {"valero": False, "houston": True},
    "Ludvig Aberg":         {"valero": True,  "houston": False},
    "Tommy Fleetwood":      {"valero": False, "houston": False},  # DP World schedule
    "Bryson DeChambeau":    {"valero": False, "houston": False},  # LIV schedule
    "Jordan Spieth":        {"valero": True,  "houston": False},
    "Hideki Matsuyama":     {"valero": False, "houston": True},
    "Cameron Smith":        {"valero": False, "houston": False},  # LIV schedule
    "Brooks Koepka":        {"valero": False, "houston": False},  # LIV schedule
    "Justin Rose":          {"valero": False, "houston": False},  # DP World schedule
    "Viktor Hovland":       {"valero": False, "houston": True},
    "Patrick Cantlay":      {"valero": True,  "houston": False},
    "Justin Thomas":        {"valero": True,  "houston": False},
    "Patrick Reed":         {"valero": False, "houston": False},  # LIV schedule
    "Matt Fitzpatrick":     {"valero": False, "houston": True},
    "Cameron Young":        {"valero": False, "houston": False},  # Won The Players; resting
    "Dustin Johnson":       {"valero": False, "houston": False},  # LIV schedule
    "Sungjae Im":           {"valero": True,  "houston": False},
    "Tyrrell Hatton":       {"valero": False, "houston": False},  # LIV schedule
    "Will Zalatoris":       {"valero": False, "houston": False},  # Managing back; limited schedule
    "Shane Lowry":          {"valero": False, "houston": False},  # DP World schedule
    "Corey Conners":        {"valero": True,  "houston": False},
    "Brian Harman":         {"valero": False, "houston": True},
    "Max Homa":             {"valero": True,  "houston": False},
    "Sahith Theegala":      {"valero": True,  "houston": False},
    "Robert MacIntyre":     {"valero": False, "houston": True},
    "Tom Kim":              {"valero": True,  "houston": False},
    "Sepp Straka":          {"valero": False, "houston": True},
    "Jason Day":            {"valero": False, "houston": True},
    "Adam Scott":           {"valero": False, "houston": False},  # Limited PGA schedule
    "Akshay Bhatia":        {"valero": True,  "houston": False},
    "Sam Burns":            {"valero": True,  "houston": False},
    "Keegan Bradley":       {"valero": False, "houston": False},
    "Min Woo Lee":          {"valero": False, "houston": True},
    "Tony Finau":           {"valero": True,  "houston": False},
    "Wyndham Clark":        {"valero": False, "houston": True},
    "Harris English":       {"valero": True,  "houston": False},
    "Si Woo Kim":           {"valero": True,  "houston": False},
    "Nick Taylor":          {"valero": False, "houston": True},
    "Emiliano Grillo":      {"valero": True,  "houston": False},
    "J.J. Spaun":           {"valero": True,  "houston": False},
    "Chris Kirk":           {"valero": True,  "houston": False},
    "Taylor Moore":         {"valero": True,  "houston": False},
    "Davis Riley":          {"valero": True,  "houston": False},
    "Byeong Hun An":        {"valero": False, "houston": True},
    "Austin Eckroat":       {"valero": True,  "houston": False},
    "Mackenzie Hughes":     {"valero": True,  "houston": False},
    "Denny McCarthy":       {"valero": False, "houston": True},
    "Gary Woodland":        {"valero": False, "houston": True},
    "Maverick McNealy":     {"valero": True,  "houston": False},
    "Alex Noren":           {"valero": False, "houston": False},  # DP World schedule
    "Rasmus Hojgaard":      {"valero": False, "houston": False},  # DP World schedule
    "Nicolai Hojgaard":     {"valero": False, "houston": False},  # DP World schedule
    "Thomas Detry":         {"valero": False, "houston": False},  # DP World schedule
    "Aaron Rai":            {"valero": False, "houston": False},  # DP World schedule
    "Min Woo Lee":          {"valero": False, "houston": True},
    "Joaquin Niemann":      {"valero": False, "houston": False},  # LIV schedule
}

# Form penalty (points, applied to normalized Form_Score 0-100)
# Only applied for known schedules where NEITHER event was played.
# Player-specific overrides take precedence over the default.
# Use "_default" key for the baseline penalty applied to all no-tune-up players.
PRE_MASTERS_PENALTY: dict[str, float] = {
    "_default":       -3.0,   # anyone with no Valero + no Houston
    "Rory McIlroy":   -2.0,   # Withdrew Arnold Palmer (back), T46 Players. Defending champ = partial penalty not full -3
    "Collin Morikawa": -3.0,  # No competitive play since March 12; back spasms; explicit max
    "Patrick Reed":   -3.0,   # LIV schedule; no PGA tune-ups; poor 2026 form warrants full penalty
}


def get_pretournament_events(player_name: str) -> dict[str, bool] | None:
    """Returns {valero: bool, houston: bool} or None if unknown."""
    return PRETOURNAMENT_SCHEDULE.get(player_name, None)


def played_tune_up(player_name: str) -> bool | None:
    """True if played ≥1 tune-up, False if neither, None if unknown."""
    sched = PRETOURNAMENT_SCHEDULE.get(player_name, None)
    if sched is None:
        return None
    return sched["valero"] or sched["houston"]


def events_label(player_name: str) -> str:
    """Human-readable label for pre-Masters schedule."""
    sched = PRETOURNAMENT_SCHEDULE.get(player_name, None)
    if sched is None:
        return "Unknown"
    parts = []
    if sched["valero"]:
        parts.append("Valero")
    if sched["houston"]:
        parts.append("Houston")
    return ", ".join(parts) if parts else "Neither (-3 Form)"


# ─────────────────────────────────────────────────────────────────
# 2026 COURSE CHANGE NOTES
# Hurricane Helene (Sept 2024) impact + course renovations
# ─────────────────────────────────────────────────────────────────

COURSE_CHANGE_NOTES = """
### 2026 Augusta National — Course Changes & Conditions

**🌀 Hurricane Helene Impact (September 2024)**
Augusta National suffered significant tree loss during Hurricane Helene, affecting five holes.
This has meaningfully changed the character of several key scoring holes:

| Hole | Change | Expected Effect |
|------|--------|-----------------|
| **Hole 3** (Par 4, 350 yds) | Large oak trees removed on right side | More exposed; crosswind plays bigger; wider bailout right but more firm run-outs |
| **Hole 10** (Par 4, 495 yds) | Pines removed on left side | Tee shot plays more exposed; downhill fade line now more rewarding |
| **Hole 11** (Par 4, 505 yds) | Tree coverage on left reduced | Wind exposure increases; classic "Amen Corner" entrance more severe |
| **Hole 15** (Par 5, 530 yds) | Multiple trees removed behind green | Second shot zone more open; 2nd-shot plays shorter with fewer obstacle concerns |
| **Hole 16** (Par 3, 170 yds) | Tree line along left thinned | Wind affect on tee shot now more unpredictable; CBS camera angles changed |

**🏗️ Green Renovations**
Augusta rebuilt or re-grassed multiple greens since 2023, with the most significant work on:
- **Hole 1**: New TifEagle ultra-dwarf bermuda; faster and firmer than previous years
- **Hole 8**: Re-graded approach and green surface; back pin positions now more accessible
- **Hole 15**: Green rebuilt alongside pond expansion (2024); approach angles changed
- **Hole 16**: Entirely new green complex post-Helene; iconic Sunday back-pin now plays differently

**🌿 2026 Expected Conditions**
Based on pre-tournament reports, the course is expected to play **firm and fast** in 2026:
- Greens running ~13.5 Stimp (up from ~12.5 in wet 2025)
- Increased premium on **precise approach play** to firm, fast greens
- Par-5 #15 slightly easier (second shot opens up without rear tree hazard)
- Par-3 #16 more unpredictable (wind exposure increased)
- Amen Corner (11-12-13) plays harder in wind with tree coverage removed on 10/11

**📐 Model Adjustment: Firm Green Bonus**
The 2026 firm green conditions are modeled via a **Firm Green Bonus** (+0 to +4 pts) added to
each player's Augusta Score. The bonus rewards elite SG Approach precision specifically suited
to firm, fast green surfaces. Players with SG App > +0.80 benefit most.
"""

HOLE_BY_HOLE_NOTES = {
    3:  "Tree removal right side → more wind exposure; run-outs reward precise driving",
    10: "Pine removal left → downhill tee shot more exposed to crosswind; fade line rewarded",
    11: "Reduced left tree coverage → Amen Corner entrance plays harder in any wind",
    15: "Rebuilt green + trees removed behind → second shot opens up; easier scoring hole in 2026",
    16: "New green complex + thinned tree line → wind effect more unpredictable on tee shot",
    1:  "New TifEagle surface → faster, firmer; premium on approach trajectory",
    8:  "Re-graded green → back pins now more accessible with proper approach shape",
}

# Maximum bonus from Firm Green adjustment (points added to Augusta_Score)
FIRM_GREEN_BONUS_MAX = 4.0


# ─────────────────────────────────────────────────────────────────
# PLAYER NOTES — 2026 current season context
# ─────────────────────────────────────────────────────────────────

PLAYER_NOTES: dict[str, str] = {
    "Scottie Scheffler": (
        "Won American Express in January 2026 (20th PGA Tour career win). Limited pre-Masters "
        "starts due to family situation (wife expecting around tournament week). Won Augusta in "
        "2022 and 2024 — dominant two-time champion with elite course feel. **Minor concern:** "
        "fewer competitive rounds than normal in the 8 weeks leading up to Masters week."
    ),
    "Rory McIlroy": (
        "🏆 **Defending champion (2025 Masters, -11, playoff victory over Justin Rose).** "
        "Completed career Grand Slam in 2025 — one of the most celebrated Augusta moments ever. "
        "Healthy and reportedly motivated to become the first repeat winner since Tiger (2001-02). "
        "Played WGC Mexico and other events but skipped both Houston and Valero as he rests for Augusta."
    ),
    "Collin Morikawa": (
        "Won 2026 Pebble Beach Pro-Am, then **withdrew from The Players Championship** "
        "with a back injury (lower lumbar strain). Status uncertain — reportedly practicing at Augusta "
        "in early April but not at full competitive intensity. Augusta suits his elite approach play "
        "profile perfectly; back health is the single key variable for his 2026 campaign."
    ),
    "Cameron Young": (
        "**Won The Players Championship 2026** (first major-equivalent victory). Coming into "
        "Augusta in the best form of his career with multiple top-5s in 2025-26. "
        "Elite driving distance profile perfectly suits Augusta's par-5 birdie opportunities on "
        "2, 8, 13, and 15. Pool ownership will be elevated after The Players win — price him accordingly."
    ),
    "Jon Rahm": (
        "2023 Masters champion (-12). **2026 LIV season has been dominant** — won LIV Hong Kong, "
        "plus runner-up finishes at South Africa, Adelaide, and Riyadh, T5 Singapore. "
        "Five top-5s in five events is the best LIV season form of any Augusta contender. "
        "Augusta DNA is elite (2023 win, multiple other top-5s). "
        "Form + DNA alignment makes him one of the highest-conviction picks in the field."
    ),
    "Ludvig Aberg": (
        "T2 at 2024 Masters in his debut appearance — one of the most impressive Augusta debuts "
        "in recent memory. Rapidly ascending world ranking (Top 5 in 2026). Elite approach play "
        "and driving stats fit the 2026 firm course perfectly. "
        "Could be the breakout winner of his generation at Augusta."
    ),
    "Xander Schauffele": (
        "2024 PGA Championship and The Open Championship winner. Strong recent major résumé. "
        "Augusta specifically has been elusive — T5 is best finish. Excellent 2026 course-fit "
        "profile with elite approach numbers. Expect high pool ownership given recent major wins."
    ),
    "Tommy Fleetwood": (
        "T4 at 2025 Masters. Multiple high finishes across majors in 2024-25. "
        "Exceptional approach play (SG App consistently elite) suits 2026 firm green conditions. "
        "Lower ownership than his score warrants — potential sleeper at +1800."
    ),
    "Hideki Matsuyama": (
        "2021 Masters champion — Augusta remains a career-defining venue. Strong 2026 season "
        "form, particularly in the winter swing. His powerful, precise ball-striking is well-suited "
        "to the 2026 firm conditions. Worth monitoring as an ownership differentiator at +2800."
    ),
    "Jordan Spieth": (
        "2015 Masters champion with multiple Augusta top-5s (2014, 2015, 2016, 2018). "
        "Elite Augusta pedigree — arguably the best Augusta specialist of his generation. "
        "2026 form has shown glimpses of return to near-peak level. "
        "At +4000, genuinely undervalued if his putting returns to 2015 levels."
    ),
    "Justin Rose": (
        "T2 at 2025 Masters (lost in playoff to McIlroy). Multiple Augusta top-5s since 2017. "
        "At 45, this is a closing window for the green jacket but Augusta clearly suits him. "
        "DP World Tour member — expect below-average pool ownership as a differentiator."
    ),
    "Bryson DeChambeau": (
        "Runner-up at 2023 Masters. **Back-to-back LIV wins in 2026** — won South Africa and "
        "Singapore. Two wins and two other top-10s in five LIV events is the hottest form "
        "of any player in the field. Par-5 birdie machine when driving is firing — Augusta's "
        "four reachable par-5s are where he separates. T7 in 2025. "
        "**Highest LIV form score in the model** given back-to-back victories."
    ),
    "Brooks Koepka": (
        "Five-time major champion — Augusta is the one major he hasn't won yet. "
        "LIV Golf schedule naturally limits public form data. Historically peaks at majors "
        "regardless of build-up. Persistent knee/hip history is the concern — at +3300, "
        "a reasonable value given his major-championship pedigree."
    ),
    "Dustin Johnson": (
        "2020 Masters champion (record -20). Augusta profile remains elite despite age (41). "
        "LIV Golf limits competitive form assessment but he returns to Augusta every year "
        "motivated. Multiple prior top-10s. Underowned relative to his Augusta DNA — "
        "a legitimate contrarian option at +10000."
    ),
    "Tiger Woods": (
        "⚠️ **Reportedly out for 2026 Masters** following back surgery in October 2025. "
        "Listed in betting markets but official status unconfirmed. Has played Augusta with "
        "injuries before, but this recovery timeline makes participation extremely unlikely. "
        "**Do not pick** unless you see confirmed tee time entry."
    ),
    "Anthony Kim": (
        "Returned to professional golf via LIV in 2024 after a 12-year absence. "
        "Has competed at Augusta before (T9 in 2010 was his best result). "
        "At +12500, pure speculation on a return to form — "
        "extremely limited recent competitive rounds to assess current game."
    ),
    "Viktor Hovland": (
        "Consistent Augusta performer with multiple top-10s. 2023 FedEx Cup champion. "
        "Strong 2026 form with wins and top-5s on PGA Tour. "
        "Driving distance and approach play profile suit Augusta well. At +3300, fair value."
    ),
    "Patrick Cantlay": (
        "T5 at 2025 Masters — Augusta consistently brings out his best. "
        "Elite iron play and bogey avoidance profile. Top-10 most years he competes. "
        "At +4000, reasonable value in a pool format given his consistency."
    ),
    "Tyrrell Hatton": (
        "LIV Golf member with Augusta experience. Passionate competitor who channels "
        "frustration productively at majors. T10 at 2024 Masters. "
        "At +4000, a legitimate contrarian pick with real Augusta upside."
    ),
    "Matt Fitzpatrick": (
        "2022 US Open champion with elite iron play and approach precision. "
        "The 2026 firm green conditions at Augusta specifically favor his game style. "
        "Multiple top-15s at Augusta. At +2000, on the expensive side but the fit is real."
    ),
}

# Defending champion historical performance note (shown on McIlroy's card)
DEFENDING_CHAMP_NOTE = (
    "📊 **Defending Champion Historical Performance at Augusta (2000–2025):**\n"
    "Of 25 defending champions, only 3 finished top-5 the following year (Scheffler '25→T3, "
    "Mickelson '05→T4, Tiger '02→Win). Average result for the defending champ: T18. "
    "The market tends to overweight recency — defending champions are typically "
    "**priced ~15% too short** relative to their actual performance distribution. "
    "In a pool format, consider pairing McIlroy at the margin rather than as an anchor."
)
