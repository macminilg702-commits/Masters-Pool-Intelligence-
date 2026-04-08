# Augusta Pool Intelligence 2026

A decision-support system for the Ferraro Green Jacket Pool — a 500-entry, $40,000 pot, four-player team format played annually during the Masters Tournament. The app scores all 110 confirmed field players using a five-component composite model, generates three optimized team lineups calibrated to this pool's specific ownership behavior, and provides pool intelligence analytics based on actual entry data from 2019 and 2024. Unlike manual picks, it accounts for LIV player form proxies, Augusta-specific course fit metrics, pool ownership concentration patterns, and a tiebreaker strategy built on historical pool submission distributions — not just who plays the best golf.

---

## Quick Start

```bash
cd ~/Desktop/Claude\ Masters
python3 -m streamlit run app.py          # launch dashboard
python3 fetch_data.py                    # manual data refresh
python3 backtest_data.py                 # view 20-year model backtest
```

The app runs at `http://localhost:8501`. Data auto-refreshes from cache on load; force a live refresh with `python3 fetch_data.py`.

---

## Project Structure

```
Claude Masters/
├── app.py                # Streamlit dashboard — 4-tab UI, all rendering logic
├── score_engine.py       # 5-component Augusta Score model + EV scoring
├── pool_optimizer.py     # Team generation (A/B/C), ownership calibration,
│                         # portfolio correlation, tiebreaker logic
├── fetch_data.py         # PGA Tour GraphQL, Odds API, Open-Meteo fetches;
│                         # 6-hour cache; LIV player fallback stats
├── field_data.py         # Static lookup tables: odds name overrides,
│                         # Augusta cut rates, pool ownership multipliers,
│                         # additional Masters history, 2026 field expansion
├── player_context.py     # Injury status/multipliers, pre-Masters schedule
│                         # flags (Valero/Houston), −3 Form penalty logic
├── player_form_data.py   # Manual overrides: 60-day OWGR rank changes,
│                         # 90-day SG Approach, Sunday scoring differentials
├── live_tracker.py       # Tournament-mode pool standings tracker;
│                         # parses uploaded entry CSV/XLSX and live scores
├── tiebreaker.py         # Model-based tiebreaker prediction (weather-adjusted);
│                         # superseded by pool_optimizer.compute_pool_tiebreaker
│                         # for cluster-beating strategy
├── backtest_data.py      # 20-year historical backtest (2006–2025);
│                         # per-year winner model rank, hit rates by condition
└── data/
    ├── pga_stats.json    # Cached PGA Tour GraphQL stats (6-hour TTL)
    ├── odds.json         # Cached Odds API outright prices (6-hour TTL)
    └── weather.json      # Cached Open-Meteo Augusta forecast (6-hour TTL)
```

---

## The Scoring Model

### Formula

```
Augusta_Score = (Form × 0.32) + (Fit × 0.30) + (DNA × 0.13)
              + (Vegas × 0.20) + (Trajectory × 0.05)
```

All five components are normalized to a 0–100 scale across the full field before weighting. Final score is adjusted by modifiers (cut rate, injury, chalk penalty, course adjustment).

---

### Component A: Recent Form (32%)

Measures current competitive sharpness entering Masters week.

**Sub-weights:**

| Signal | Weight |
|--------|--------|
| SG Tee-to-Green (last 4 events) | 50% |
| Last start finish quality | 18% |
| Top-8 recency (last 7 starts) | 12% |
| Season wins / top-5 count | 20% |

**Recent win bonus** applied on top of weighted composite:

| Condition | Bonus |
|-----------|-------|
| Last start = win | +15 pts |
| Won earlier this season + current top-10 form | +8 pts |
| 2+ wins this season | +5 pts |

Season wins are weighted at 20% of the Form component because **81% of Masters winners from 2006–2025 had won at least once earlier that season** — the single strongest pre-tournament predictor in the backtest data. This signal was historically underweighted at 15% before being raised in v2.0.

---

### Component B: Course Fit (30%)

Measures how well a player's skill profile matches Augusta National's specific demands. SG Putting is **explicitly excluded** — the average Masters winner ranks 98th in putting that week. Augusta's greens are so idiosyncratic that pre-tournament putting stats are not predictive.

**Sub-weights:**

| Signal | Weight |
|--------|--------|
| SG Approach to Green (season) | 18% |
| SG Approach to Green (90-day) | 12% |
| Par-5 scoring average | 28% |
| Bogey avoidance (scoring avg proxy) | 16% |
| SG Around the Green | 18% |
| Driving distance | 5% |
| Sunday scoring differential | 3% |

Par-5 scoring is weighted at 28% of this component because Augusta's four reachable par-5s (2, 8, 13, 15) create up to **16 strokes of potential separation over 72 holes** — the largest single Augusta-specific scoring lever available in public data. Elite par-5 players (those scoring −0.8 or better) gain a compounding advantage across all four rounds.

---

### Component C: Augusta DNA (13%)

Measures historical performance at Augusta National specifically.

**Sub-weights:**

| Signal | Weight |
|--------|--------|
| Weighted finish history (recency-decayed) | 50% |
| Best-ever Augusta finish | 25% |
| Prior starts count | 15% |
| Last appearance result | 10% |

Recency decay multipliers: 2025 = 0.35, 2024 = 0.25, 2023 = 0.20, 2022 = 0.10, older = 0.05.

DNA was reduced from 18% to 13% in v2.0 based on the backtest finding that **76% of Masters top-10 finishers from 2010–2025 had zero prior Augusta top-10 finishes**. Past history at Augusta is less predictive of future performance than skill profile fit — the course rewards consistent ball-striking that can be measured directly.

---

### Component D: Vegas Calibration (20%)

**Sub-weights:**

| Signal | Weight |
|--------|--------|
| Vig-removed implied win probability | 70% |
| Model-vs-market divergence bonus | 30% |

Vegas is weighted at 20% for two reasons: (1) it captures current LIV player form that SG data cannot measure, since LIV does not publish strokes-gained statistics; (2) the betting market is demonstrably efficient at pricing Augusta-specific fit — sharp money has historically identified course-fit players that form-only models miss. The divergence bonus rewards players where the model's three-factor score (Form + DNA + Fit) exceeds market expectation, and penalizes overvalued picks.

---

### Component E: Rank Trajectory (5%)

60-day OWGR ranking change, converted to a 0–100 directional signal (top-10 improvement = 92 pts, top-25 = 75, stable = 50, declining = 25–35). Low weight by design — trajectory is a confirming signal, not a leading one.

---

### Modifiers

Applied after component scoring, in this order:

| Modifier | Logic |
|----------|-------|
| **Chalk penalty** | −5 pts if odds < +600 (world #1 and #2 range) |
| **Chaos coefficient** | Wind avg >15 mph OR temp <55°F triggers weight rebalance: Form 35%, Fit 33%, DNA 10%, Vegas 17%, Traj 5% |
| **Augusta cut rate** | Rate < 60%: penalty = (0.60 − rate) × 40 pts; Rate 60–75%: penalty = (0.75 − rate) × 20 pts |
| **Injury multiplier** | Healthy = 1.00, Minor Concern = 0.92, Significant Concern = 0.80, Unknown = 0.95 |
| **Pre-Masters schedule** | −3 Form pts if player skipped both Valero Texas Open and Houston Open |
| **2026 course adjustment** | +3 pts for top-20 driving distance (tree removal opens holes 3/10/11/15/16); +2 pts for top-20 SG Approach (firmer greens reward precision) |

---

### EV Score

```
EV_Score = Augusta_Score / sqrt(Ownership_Pct)
```

`Ownership_Pct` is the calibrated pool ownership estimate (see below), not raw odds-implied probability. EV Score is used for Team B differentiated pick selection and Team C value scoring.

---

## Pool Ownership Calibration

Raw odds-implied ownership estimates do not match how this pool actually behaves. Analysis of real entry data from 2019 (n=253) and 2024 (n=532) shows this pool **systematically over-concentrates on the top three players** relative to what betting markets imply:

| Player | Odds-Implied | Actual Pool |
|--------|-------------|-------------|
| Scheffler 2024 | ~25% | 59% |
| McIlroy 2024 | ~18% | 55% |
| Rahm 2024 | ~12% | 44% |
| Aberg 2024 | ~8% | 0% |
| McIlroy 2019 | ~18% | 63% |
| Rose 2019 | ~10% | 43% |

**Calibration applied in `calibrate_ownership_for_pool()`:**

1. Top 3 players by Augusta Score: `× 1.5` (over-concentration multiplier, capped at 65%)
2. All other players: `× 0.75`
3. Per-player corrections applied on top (from `field_data.py`):

**Pool overowned** (fame / narrative bias inflates ownership above calibrated estimate):

| Player | Multiplier |
|--------|-----------|
| Jordan Spieth | 1.8× |
| Justin Thomas | 1.6× |
| Dustin Johnson | 1.5× |
| Brooks Koepka | 1.4× |
| Jason Day | 1.4× |
| Tony Finau | 1.3× |
| Cameron Smith | 1.3× |

**Pool underowned** (legitimate contenders overlooked by fame/narrative bias):

| Player | Multiplier |
|--------|-----------|
| Ludvig Aberg | 0.4× |
| Sahith Theegala | 0.4× |
| Tommy Fleetwood | 0.5× |
| Wyndham Clark | 0.5× |
| Corey Conners | 0.5× |
| Viktor Hovland | 0.6× |
| Cameron Young | 0.6× |
| Shane Lowry | 0.6× |
| Matt Fitzpatrick | 0.6× |
| Collin Morikawa | 0.7× |
| Hideki Matsuyama | 0.7× |

The **3–12% ownership sweet spot** is where exploitable mispricings live. Players below 2% are underowned because they genuinely shouldn't be picked (the pool is correct about them). Players at 3–12% are underowned due to the pool's fame/narrative bias.

---

## Three-Team Strategy

### Team A — Floor

Built to **place consistently** (target: top-5 pool finish in any year).

- **Selection:** top 4 players by Augusta Score
- **Hard constraints:** cut rate ≥ 65%, odds ≤ +6,000
- **Backtest:** highest aggregate score in 10/20 years; avg aggregate −12.2 strokes

### Team B — Ceiling

Built to **win outright** when a mid-tier player emerges.

- **Target ownership profile** (derived from 2024 winning team analysis):
  - 2 chalk anchors from top-3 by Augusta Score (35–65% owned)
  - 1 mid-tier pick (10–25% pool ownership)
  - 1 differentiated upside pick (3–12% pool ownership)
- **Hard constraints:** cut rate ≥ 65%, Augusta Score ≥ 50th percentile for value/mid slots
- **Backtest:** highest aggregate score in 6/20 years

The 2024 winning team structure was Scheffler (59%) + Rahm (44%) + Morikawa (18%) + Schauffele (15%) — two chalk anchors plus two mid-tier picks. Team B targets this profile.

### Team C — Value

Built to **win outright via ownership edge** — the Aberg/Fleetwood play.

- **Target:** 1 chalk anchor (scoring floor) + 3 players from the 3–12% ownership sweet spot
- **Hard constraints:** cut rate ≥ 65%, Augusta Score ≥ 45th percentile, Ownership_Pct 2–20%
- **Selection score:** `EV × 0.40 + AugustaScore × 0.30 + OwnershipDiff × 0.30`
- The ownership differentiation score peaks at 7% ownership (the heart of the sweet spot)

In 2026, the 3–12% band likely contains: Tommy Fleetwood, Viktor Hovland, Cameron Young, Hideki Matsuyama, Wyndham Clark, Corey Conners, Shane Lowry, Sahith Theegala, Matt Fitzpatrick.

### Portfolio Correlation Check

After generating all three teams, `compute_portfolio_correlation()` flags:

| Overlap | Threshold | Flag |
|---------|-----------|------|
| Teams A+B | > 2 shared players | High correlation warning |
| Teams A+C | > 1 shared player | Differentiation warning |
| Teams B+C | > 1 shared player | Differentiation warning |
| All three | > 1 shared player | Concentration warning |

Lower correlation score = better portfolio diversification across tournament outcomes.

---

## Tiebreaker Strategy

**Do not submit the model's optimal aggregate estimate. Submit a score that beats the largest cluster in this specific pool.**

Historical Ferraro pool tiebreaker data:

| Year | n | Median | Most Common |
|------|---|--------|-------------|
| 2019 | 253 | −32 | −28, −30 |
| 2024 | 532 | −24 | −25 (41 entries), −24 |

Pool submissions herd around round numbers. The strategy targets the number just below the largest cluster:

| Condition | Submit | Rationale |
|-----------|--------|-----------|
| Normal (wind < 15 mph) | **−33** | Beats the entire −32 cluster |
| Chaos (wind ≥ 15 mph or temp < 55°F) | **−29** | Beats −28 and −30 clusters |

This is implemented in `compute_pool_tiebreaker()` in `pool_optimizer.py`. The tiebreaker only matters when your team's actual aggregate ties another entry — but at 500+ entries with a $20,000 first-place payout, the right cluster strategy converts a split into a win.

---

## LIV Player Handling

LIV Golf does not publish strokes-gained statistics. For LIV players, the model uses manually curated fallback stats in `FALLBACK_PLAYER_STATS` in `fetch_data.py`.

**Form proxy:** Recent LIV finish positions are converted to a 0–100 form score using the same finish-quality mapping as PGA Tour results, then discounted by the LIV field strength factor relative to PGA Tour fields.

**Driving distance:** LIV published stats, converted to PGA Tour equivalent percentile ranking using field size normalization.

LIV players are **not** flagged for SG threshold violations — the data doesn't exist, not that they failed the threshold.

**LIV players in the 2026 Masters field (11 confirmed):**
Bryson DeChambeau, Sergio Garcia, Tyrrell Hatton, Dustin Johnson, Tom McKibbin, Phil Mickelson, Carlos Ortiz, Jon Rahm, Charl Schwartzel, Cameron Smith, Bubba Watson.

---

## Data Sources

| Source | What it provides | Cache TTL |
|--------|-----------------|-----------|
| PGA Tour GraphQL | Season stats, recent results — 74 PGA Tour players, 14 stats each | 6 hours |
| The Odds API | Current outright win prices (Masters market) | 6 hours |
| Open-Meteo | Augusta National 10-day weather forecast (Apr 9–12 focus) | 6 hours |
| ESPN API | Fallback if PGA Tour GraphQL is unavailable | 6 hours |
| `fetch_data.py` fallback stats | Manual LIV player data, ~40 additional players | Static |

All cache files live in `data/`: `pga_stats.json`, `odds.json`, `weather.json`.

---

## 20-Year Backtest Summary

Backtest data in `backtest_data.py` covers the 2006–2025 Masters (20 tournaments). Each year records the winner's pre-tournament world ranking, approximate odds, and where the model would have ranked them.

Run `python3 -c "from backtest_data import backtest_summary; import pprint; pprint.pprint(backtest_summary())"` for full stats.

| Metric | Result |
|--------|--------|
| Winner ranked model top-3 | 6/20 years (30%) |
| Winner ranked model top-5 | 8/20 years (40%) |
| Winner ranked model top-10 | 10/20 years (50%) |
| Average winner model rank | 15.2 |
| Top-10 hit rate — mild conditions | 45% |
| Top-10 hit rate — fast/firm conditions | 67% |
| Top-10 hit rate — soft/wet conditions | 100% |
| Top-10 hit rate — chaos (cold/windy) | 25% |
| Top-10 hit rate — non-chaos years | 56% |

**Best call:** 2024 — Scheffler ranked #1 by model, won at +500.

**Biggest miss:** 2007 — Zach Johnson won at +15000 (world rank 56). Cold, windy conditions made par-5 lay-up strategy optimal. Model ranked him 54th. Largest longshot winner in backtest period.

**Weak years** for the model: chaos-condition years (2007, 2011) when cold wind neutralizes par-5 and driving distance advantages. Under Chaos Coefficient, DNA is reduced and Form is elevated to compensate.

### Top-10 Pattern Analysis (2006–2025)

- **81%** of Masters winners won at least once earlier that season
- **76%** of top-10 finishers had zero prior Augusta top-10s
- **100%** of winners ranked inside the top-25 world ranking
- **0** wins from players ranked outside the top-25 in 20 years
- Ball-flight weakness: draw players consistently outperform fade players in scoring models — explains misses in 2012 (Bubba Watson), 2016 (Danny Willett)

---

## Dashboard

Launch: `python3 -m streamlit run app.py`

### Tab 1 — MY PICKS

Three team cards (A/B/C) displaying each player's Augusta Score, component bars (Form/DNA/Fit), odds, and ownership band badge (CHALK/MID/VALUE/FIELD). Portfolio health row shows cross-team correlation score, avg duplication, and scenario coverage. Swap controls (inline dropdowns below cards) allow manual overrides. Tiebreaker section shows the pool-cluster-beating recommendation with rationale. Confirm button locks picks and switches to countdown view.

### Tab 2 — LIVE

Pre-tournament: demo pool standings and leaderboard layout for orientation. Tournament mode activates when a pool entries file is uploaded via the file uploader (CSV or XLSX, columns: `Team Owner`, `Golfer 1`, `Golfer 2`, `Golfer 3`, `Golfer 4`, `Tie`). Shows: pool rank (large display), per-player tournament scores, full pool standings table, Masters leaderboard with team players highlighted, and tiebreaker tracker vs live aggregate.

### Tab 3 — MODEL

Full 110-player intelligence list, sortable by Augusta Score or EV Score. My 12 team players pinned at top with expanded detail. Each row: rank, name, key driver sentence, Augusta Score, EV Score, component bars, odds, ownership %. Collapsed accordion sections for Scoring Architecture (weight diagram) and Model Validation (backtest results 2006–2025).

### Tab 4 — DATA

Analytics tab with four sections:

- **Field Overview:** top-40 Augusta Score horizontal bar chart (team players color-highlighted) + ownership vs score scatter with HIGH VALUE / CHALK / AVOID / OVEROWNED quadrants
- **Model Components:** radar charts per team (5-axis: Form/Fit/DNA/Vegas/Traj) + stacked component breakdown bar for top 20
- **Pool Intelligence:** ownership band donut (outer = full field, inner = my picks) + EV vs odds scatter with fair-value reference line + tiebreaker histogram overlaying real 2019 and 2024 pool submission distributions
- **My Teams Deep Dive:** player selector → component vs team average (grouped bar) → field percentile chart → 12-player correlation heatmap

---

## Key Files Reference

### score_engine.py

Entry point: `score_players(data, component_weights, fit_weights, apply_chalk_penalty) → DataFrame`

Calls: `compute_form_raw()`, `compute_fit_raw()`, `compute_dna_raw()`, `compute_vegas_score()`, `apply_modifiers()`, `compute_trajectory_score()`.

Returned DataFrame columns: `Player`, `Tour`, `World_Rank`, `Augusta_Score`, `Augusta_Cut_Rate`, `Form_Score`, `DNA_Score`, `Fit_Score`, `Vegas_Score`, `Trajectory_Score`, `Ownership_Pct`, `Ownership_Pct_Raw`, `EV_Score`, `Flags`, `Chalk_Penalty`.

### pool_optimizer.py

Entry point: `generate_teams(df, pot_size, num_entries, weather) → dict`

Steps: (1) `calibrate_ownership_for_pool(df)` modifies `df` in-place, adding `Ownership_Pct_Raw`; (2) builds Team A/B/C with ownership-band-based selection; (3) runs `compute_portfolio_correlation()`; (4) runs `check_combo_frequency()` per team; (5) calls `compute_pool_tiebreaker()`.

Return dict keys: `team_a`, `team_b`, `team_c`, `correlation`, `tiebreaker`, `overlap`, `payouts`.

### fetch_data.py

Entry point: `fetch_all_data(force=False) → dict`

Fetch chain: cache → PGA Tour GraphQL → ESPN fallback. Merges live stats with `FALLBACK_PLAYER_STATS` (priority: fallback < live PGA stats < recent results). Cache TTL: 6 hours for all sources. Returns dict with keys: `stats`, `odds`, `weather`.

### field_data.py

Static lookup tables used at import time:

- `ODDS_NAME_OVERRIDES`: maps Odds API name variants to canonical player names
- `UPDATED_ODDS_EXISTING`: current market prices for the base 30 players
- `AUGUSTA_CUT_RATES`: historical cut-made fraction per player (last 5 Augusta appearances)
- `ADDITIONAL_MASTERS_HISTORY`: Augusta results for players not in base history dict
- `POOL_OVEROWNED_PLAYERS`: ownership multipliers > 1.0 (fame bias)
- `POOL_UNDEROWNED_PLAYERS`: ownership multipliers < 1.0 (edge plays)

### player_context.py

- `INJURY_STATUS`: per-player status string (Healthy / Minor Concern / Significant Concern / Unknown)
- `INJURY_MULTIPLIERS`: `{Healthy: 1.00, Minor Concern: 0.92, Significant Concern: 0.80, Unknown: 0.95}`
- `PRETOURNAMENT_SCHEDULE`: per-player `{valero: bool, houston: bool}` flags
- `PRE_MASTERS_PENALTY = −3.0`: Form penalty for skipping both tune-up events
- `get_injury_multiplier(player_name) → float`

### player_form_data.py

Manual data not available via API:

- `RANKING_CHANGE_60D`: 60-day OWGR rank change per player
- `SG_APP_90D`: 90-day SG Approach figures (supplements live GraphQL data)
- `SUNDAY_SCORING_DIFF`: Sunday vs Thursday scoring differential per player

### live_tracker.py

- `parse_pool_entries(file_bytes, filename) → list[dict]`: parses uploaded CSV/XLSX
- `fetch_live_scores() → dict`: Masters leaderboard (live during tournament)
- `compute_standings(entries, live_scores) → DataFrame`: calculates pool positions

### tiebreaker.py

`predict_tiebreaker(weather_dict) → dict`: weather-adjusted model tiebreaker estimate. Used for the range/confidence display in the UI. The pool cluster strategy (`compute_pool_tiebreaker` in `pool_optimizer.py`) is the primary recommendation; `tiebreaker.py` provides the meteorological context panel.

### backtest_data.py

`BACKTEST_RESULTS`: list of 20 dicts (one per year, 2006–2025) with fields: `year`, `winner`, `pre_rank`, `pre_odds_us`, `model_rank`, `condition`, `chaos`, `top3_hit`, `top5_hit`, `top10_hit`, `notes`.

`backtest_summary() → dict`: hit rates by condition, avg model rank, best call, biggest miss.

---

## Updating Before April 9

### After Valero Texas Open (April 5–6)

1. Run `python3 fetch_data.py` to pull final Valero results and updated season stats
2. If the Valero winner is a new Masters exemption, add them to `field_data.py` field expansion
3. Confirm `recent_win_bonus` applies: last start = win → +15 Form pts
4. Update `PRETOURNAMENT_SCHEDULE` for any players who entered/withdrew late
5. Check LIV Hong Kong results (finished ~April 5) — update relevant `FALLBACK_PLAYER_STATS` manually
6. Weather forecast firms up significantly this week — check chaos mode threshold

### Morning of April 9 (First Tee ~10:45am ET)

1. Final `python3 fetch_data.py` — captures any late morning odds movement
2. Check for late withdrawals and update `FALLBACK_PLAYER_STATS` (set odds to 999999 and note WD)
3. Verify tiebreaker: −33 under normal conditions, −29 if chaos mode is active
4. Review all three teams, confirm no flagged cuts, submit via pool commissioner
5. Pool entries file distributed by commissioner — upload via Live tab when received

### During Tournament (April 10–13)

1. Commissioner distributes full entries file (typically Thursday morning)
2. Upload via the file uploader in the Live tab
3. Enter your entry names in "Your entry names" field — app highlights your position
4. App switches to tournament mode automatically on successful file parse
5. Live scores fetch from Masters leaderboard; pool standings update on each refresh

---

## Version History

### v1.0 — Initial Build

- 5-component Augusta Score model with PGA Tour GraphQL integration
- Streamlit dashboard (3 tabs: MY PICKS, LIVE, MODEL)
- Basic pool optimizer (Floor/Balanced/Contrarian teams)
- EV scoring (`AugustaScore / sqrt(Ownership)`)

### v2.0 — Current

**Model changes:**

- DNA weight: 18% → 13% (backtest: 76% of top-10 had zero prior Augusta top-10s)
- Fit weight: 27% → 30% (par-5 and scrambling are most Augusta-specific measurable signals)
- Vegas weight: 18% → 20% (captures LIV form; market efficiency at Augusta)
- Par-5 sub-weight raised to 28% of Fit component
- SG ATG sub-weight raised to 18% of Fit component
- Season wins sub-weight raised to 20% of Form component
- Recent win bonus added: +15/+8/+5 pts
- Augusta cut rate filter added (CUT_RATE_FLOOR = 0.60)
- 2026 course adjustment: +3 DD, +2 SG App (tree removals + firm greens)

**Pool intelligence changes:**

- Ownership calibration: 1.5× top-3 multiplier + per-player over/under bias corrections
- Team C redesigned: 3–12% ownership sweet spot targeting with ownership differentiation score
- Tiebreaker strategy: cluster-beating (−33 normal, −29 chaos) replaces ceiling-chasing
- Portfolio correlation check: flags high cross-team overlap
- Combo frequency estimator: warns when team matches ~25% of pool

**Dashboard changes:**

- Ownership band badges (CHALK/MID/VALUE/FIELD) on player slots
- Portfolio health row with correlation score and scenario coverage
- DATA tab: 4 sections, 10 Plotly charts, full field analytics

---

## Model Limitations

1. **No SG data for LIV players.** 11 of ~110 field players (including Rahm, DeChambeau, Hatton, Johnson, Smith) are scored using manually curated finish-position proxies. Directionally correct but imprecise relative to strokes-gained metrics.

2. **Historical odds not archived.** Backtest uses world ranking as a Vegas proxy for 2006–2018; approximate odds for 2019–2025. The Vegas component's backtest accuracy is lower than its live-year accuracy.

3. **Ownership calibration based on two years only** (2019 and 2024). Entries files from 2021–2023 would improve the calibration. The per-player bias multipliers are directionally supported but may be over-fit to two data points.

4. **Ball flight / shot shape not captured.** Augusta strongly favors draw players, particularly off the tee on holes 1, 9, 10, and 18. This explains some systematic misses: Bubba Watson (2012, 2014), Danny Willett (2016). No public dataset captures shot shape at the player level.

5. **Weather model uses forecast only.** Chaos mode triggers on the Open-Meteo forecast for April 10–13 — not real-time conditions. Forecast accuracy drops past 5 days. Check weather manually on April 8–9 and override chaos mode via the toggle if needed.

---

## Pool Details

| Detail | Value |
|--------|-------|
| Pool name | Ferraro Green Jacket Pool |
| Format | 4-player team, aggregate strokes-to-par (best 72-hole total) |
| Entry cost | $35/team or 3 teams for $100 |
| Estimated pot | ~$40,000 (500 entries) |
| Payouts | 1st 50% · 2nd 25% · 3rd 15% · 4th 7% · 5th 3% |
| Tiebreaker | Closest to your team's aggregate strokes-to-par prediction |
| Entry deadline | Before first tee Thursday April 9, 2026 |
| Full entries distributed | Thursday morning of tournament week |
