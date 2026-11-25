"""Walk-forward backtest and the dashboard's data file.

    python -m nba_edge backtest            # ~5 minutes; writes data/predictions.parquet and web/data.json
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from . import evaluate as E
from . import models as M
from . import teams
from .features import EloParams

DESIGN = {
    "burn_in": ["2007-08", "2008-09", "2009-10", "2010-11"],
    "tune": ["2011-12", "2012-13"],
    "test_from": "2013-10-01",
    "refit_days": 14,
    "main_edge": 0.02, "kelly_mult": 0.25, "max_stake": 0.02,
    "lineup_seasons": ["2011-12", "2023-24"],
}
LABELS = {"market": "Market (vig removed)", "ensemble": "Adaptive ensemble", "mkt_gbm": "Market + boosted trees",
          "mkt_lr": "Market + regression", "stats_gbm": "Stats only, boosted trees",
          "stats_lr": "Stats only, regression", "v1": "October model (Elo + rest + form)"}
ORDER = ["market", "ensemble", "mkt_gbm", "mkt_lr", "stats_lr", "stats_gbm", "v1"]


def run(X: pd.DataFrame, prm: M.Params | None = None, verbose=True):
    prm = prm or M.Params()
    X = X[X["p_market"].notna() | X["home_win"].isna()].reset_index(drop=True)
    P, whist, imp, fc = M.walk_forward(X, DESIGN["test_from"], prm, refit_days=DESIGN["refit_days"], verbose=verbose)
    G = X.set_index("game_id").loc[P["game_id"]].reset_index()
    for c in ORDER:
        G["p_" + c] = P[c].to_numpy()
    return G, whist, imp, fc


def metrics(G: pd.DataFrame, whist, imp, fc, elo: EloParams):
    y = G["home_win"].to_numpy()
    nights = G["date"].dt.strftime("%Y-%m-%d").to_numpy()
    P = {c: G["p_" + c].to_numpy() for c in ORDER}
    out = {"design": {**DESIGN, "elo": {"k": elo.k, "hca": elo.hca, "carry": elo.carry}}}
    out["summary"] = {c: E.summary(P[c], y) for c in ORDER}
    out["comparisons"] = {c: E.bootstrap_diff(P[c], P["market"], y, nights) for c in ORDER if c != "market"}
    out["calibration"] = {c: E.calibration(P[c], y) for c in ["market", "ensemble", "stats_lr", "v1"]}
    lin = G["has_lineup"].to_numpy() == 1
    out["split"] = {
        "with_lineups": {c: E.summary(P[c][lin], y[lin]) for c in ORDER},
        "without_lineups": {c: E.summary(P[c][~lin], y[~lin]) for c in ORDER} if (~lin).sum() else {},
    }
    bs = []
    for s, d in G.groupby("season"):
        yy = d["home_win"].to_numpy()
        row = {"season": s, "n": int(len(d)), "home_win_rate": float(yy.mean()),
               "market_home_prob": float(d["p_market"].mean()), "ensemble_home_prob": float(d["p_ensemble"].mean()),
               "v1_home_prob": float(d["p_v1"].mean()), "has_lineups": bool(d["has_lineup"].mean() > 0.5)}
        for c in ORDER:
            row[c + "_brier"] = E.brier(d["p_" + c].to_numpy(), yy)
        bs.append(row)
    out["by_season"] = bs
    out["encompassing"] = {c: E.encompassing(P[c], P["market"], y) for c in ["ensemble", "mkt_gbm", "stats_lr"]}
    bet = {}
    for c in ORDER:
        if c == "market":
            continue
        b = E.bets(P[c], G["ml_home"].to_numpy(), G["ml_away"].to_numpy(), y, DESIGN["main_edge"],
                   DESIGN["kelly_mult"], DESIGN["max_stake"])
        bet[c] = E.betting_summary(b, nights)
    out["betting"] = bet
    _, over = E.no_vig(G["ml_home"].to_numpy(), G["ml_away"].to_numpy())
    out["data"] = {"games_total": int(len(G)), "first_date": str(G["date"].min().date()),
                   "last_date": str(G["date"].max().date()), "avg_overround": float(np.nanmean(over)),
                   "blind_bet_roi": float(np.nanmean(1 / over - 1))}
    # importance: average over all refits
    imp_df = pd.DataFrame([d for _, d in imp]).fillna(0)
    out["importance"] = imp_df.mean().sort_values(ascending=False).round(4).to_dict()
    # weights over time, sampled weekly
    wh = pd.DataFrame([dict(date=d, **w) for d, w in whist])
    wh["date"] = pd.to_datetime(wh["date"])
    wk = wh.groupby(wh["date"].dt.to_period("W")).last().reset_index(drop=True)
    out["weights"] = {"dates": wk["date"].dt.strftime("%Y-%m-%d").tolist(),
                      **{k: wk[k].round(4).tolist() for k in M.Params().use}}
    out["coefs"] = fc.coefs()
    return out


def export(G, met, path, extra=None):
    T = sorted(teams.TEAMS)
    tix = {t: i for i, t in enumerate(T)}
    seasons = sorted(G["season"].unique())
    six = {s: i for i, s in enumerate(seasons)}
    dates = sorted(G["date"].dt.strftime("%Y-%m-%d").unique())
    dix = {d: i for i, d in enumerate(dates)}
    r3 = lambda a: [None if pd.isna(x) else round(float(x), 3) for x in a]  # noqa: E731
    r0 = lambda a: [None if pd.isna(x) else int(round(float(x))) for x in a]  # noqa: E731
    games = {
        "d": [dix[d] for d in G["date"].dt.strftime("%Y-%m-%d")], "s": [six[s] for s in G["season"]],
        "h": [tix[t] for t in G["home"]], "a": [tix[t] for t in G["away"]],
        "hp": r0(G["home_pts"]), "ap": r0(G["away_pts"]), "mlh": r0(G["ml_home"]), "mla": r0(G["ml_away"]),
        "eh": r0(G["elo_home"]), "ea": r0(G["elo_away"]),
        "lh": r0(G["miss_min_home"]), "la": r0(G["miss_min_away"]),
        **{k: r3(G["p_" + c]) for k, c in [("pk", "market"), ("pe", "ensemble"), ("pg", "mkt_gbm"), ("pr", "mkt_lr"),
                                           ("ps", "stats_lr"), ("pb", "stats_gbm"), ("p1", "v1")]},
    }
    data = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "teams": [{"abbr": t, "name": teams.name(t)} for t in T], "seasons": seasons, "dates": dates,
            "labels": LABELS, "games": games, "metrics": met, **(extra or {})}
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"), allow_nan=False, default=lambda o: None)
    return data
