# NBA Edge Lab v2

**Live dashboard:** https://jason11santos.github.io/nba-edge-lab/ (updates every few minutes, runs in the cloud)

Can a model beat the NBA moneyline market? Version 2 adds live data, the data the October version was
missing (player availability, travel, a drifting home-court edge, spreads and totals, opening vs current
lines), and an ensemble of machine-learning models that re-weights itself every night.

## Result (walk-forward backtest, 15,967 games, Oct 2013 – Jan 2026)

| Forecast | Brier | Log loss | Gap to market |
|---|---|---|---|
| Market (vig removed) | 0.20570 | 0.5972 | — |
| Adaptive ensemble | 0.20572 | 0.5972 | +0.00002 (95% CI −0.00016 to +0.00019) |
| Market + boosted trees | 0.20574 | 0.5973 | +0.00004 |
| Stats only (no price), regression | 0.21154 | 0.6105 | +0.00583 |
| October model (Elo + rest + form) | 0.21479 | 0.6179 | +0.00909 |

* Without the price, v2 closes 36% of the October model's gap to the market.
* With the price, the ensemble ties the closing line. It does not beat it: the encompassing test gives p = 0.11,
  and ensemble bets returned −0.7% with a 95% interval of −10% to +9% (1,230 bets).
* The ensemble learned this by itself: by 2026 it puts ~68% weight on the market, ~32% on the two
  market-anchored learners and nothing on the stats-only learners.
* A first version counted any player who logged no minutes as "out". That leaked the game's flow
  (deep benches only play in blowouts) and produced a fake +8% return. Availability now follows the
  pre-game inactive list: inactive, or listed with an injury, rest or personal reason.

The historical odds are close to closing lines, which already price in injury news. If there is an edge,
it is earlier in the day, before the line moves. The live system stores every line it sees so that can be tested.

## Layout

```
nba_edge/
  teams.py        franchises, aliases, arena coordinates, time zones, altitude
  history.py      load the public odds + player box-score archives
  features.py     market, Elo/form, league home edge, schedule/travel, player ratings and tonight's lineup
  models.py       five learners (logistic + LightGBM, with and without the price) and the Hedge ensemble
  evaluate.py     Brier, log loss, calibration, bootstrap CIs by game night, encompassing test, bet simulation
  backtest.py     walk-forward backtest, refits every 14 days, writes web/data.json
  live.py         collect -> predict -> paper-bet -> settle -> learn loop, writes web/live.json
  store.py        SQLite: games, every odds snapshot, injury reports, player games, predictions, bets
  sources/espn.py      scoreboard, DraftKings open/current lines, closing lines, injuries, box scores
  sources/nbastats.py  season player logs via nba_api
  sources/oddsapi.py   optional: every US book's price (set ODDS_API_KEY)
web/index.html    dashboard (works from `python -m nba_edge serve` or as the published artifact)
tests/            parsers checked against recorded ESPN responses, plus no-look-ahead tests
```

## Runs in the cloud (no computer needed)

* `.github/workflows/live.yml` runs `python -m nba_edge cloud` on GitHub Actions every 5 minutes: ESPN lines and
  injuries, paper bets, settlement, a nightly retrain, and the opening-line study every Monday. The first run
  bootstraps everything (about 20–40 minutes).
* State (the SQLite database, the trained model, `live.json`) is saved to the `data` branch as a single commit
  that is overwritten each run, so the repository does not grow.
* `.github/workflows/pages.yml` publishes `web/` to GitHub Pages on every push to `main`. The page reads
  `live.json` from the `data` branch every 30 seconds, so it updates without a refresh.
* Optional: add an `ODDS_API_KEY` repository secret for every US book's price.
* GitHub's scheduler can run late, so expect fresh numbers every 5–15 minutes.

## Paper-betting rules

| Rule | Probability used | Bets when |
|---|---|---|
| Early line, EV > 3% (main) | model blended with the opening price | EV > 3% at the opening price and now; line moved ≤ 1.5 pts overall and ≤ 0.5 pts against our side |
| Early line, EV > 5% | same | same, with 5% |
| Line-move model | predicted closing price (ridge regression on what is known at the open) | predicted move toward one side beats half the bookmaker's margin by 0.5 pts; same open-window guards |
| Totals early, 3+ / 5+ pts | totals model (team scoring, market pace, league trend, rest, lineups) | model at least 3 (5) points from the opening total and from the best available total; total moved ≤ 2.5 pts, ≤ 1 against us |
| Ensemble at current line (old rule) | adaptive ensemble | EV > 2% at the current best price |

Stakes: quarter-Kelly, capped at 2%, on a shrunk edge. The opening-line study found the line moves only about a quarter of
the way toward the model, so the model's disagreement with the current fair price is multiplied by that slope before
sizing. A bet whose shrunk edge does not beat the margin is still logged (stake 0) so its closing-line value is measured.

Line shopping: with an `ODDS_API_KEY` secret, every US book's moneyline and total is pulled at most every 2 hours between
7 AM and 8 PM Pacific (about 420 of the free plan's 500 monthly credits), and each bet takes the best available price.

Each rule is tracked separately with closing-line value (CLV). Judge the rules by CLV, not profit: profit needs
thousands of bets to separate from luck; CLV settles within a few hundred.

## Opening-line study (`python -m nba_edge.openline`)

3,759 games, Dec 2023 – Nov 2026: the model does not beat the opening line on accuracy (+0.00048 Brier, CI includes 0),
but the line moves toward the model (57% of big disagreements, p < 0.001). Bets at one book's opening price gain
+2.0 points of CLV at EV > 2% and +3.7 at EV > 5%. The same model bet at the closing price loses ~9%.

## Run it on your own computer

```
pip install -r requirements.txt
python -m nba_edge setup        # download history from GitHub and build tables (~2 min)
python -m nba_edge backtest     # optional: rerun the backtest (~5 min)
python -m nba_edge live --once  # first run backfills Jan 2026 to now from ESPN, trains, predicts today
python -m nba_edge live         # keep running: a pass every 10 minutes, retrain nightly
python -m nba_edge serve        # dashboard at http://localhost:8000
python -m pytest -q             # tests
```

On Windows, use `py` in place of `python` if `python` is not on your PATH.

## What a live pass does

1. ESPN scoreboard for yesterday, today, tomorrow: scores, state, DraftKings opening and current lines (stored every pass).
2. Optional: every book's price from The Odds API, so a bet takes the best number.
3. ESPN injury report: each player's chance of playing (out 0, doubtful 0.15, questionable 0.55, probable 0.9).
4. Features for today, tonight's lineup weighted by those chances, every learner's forecast, ensemble blend.
5. Paper bet when expected value > 2% at the best price: quarter-Kelly, capped at 2% of bankroll. The price is locked at first signal.
6. Finished games: settle bets, compute closing-line value, update the ensemble's weights.
7. Nightly: pull the newest player logs and refit every learner on all games so far.

Research project, not betting advice. Sports betting is restricted in Washington.
