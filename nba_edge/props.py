"""Player projections for prop markets (points, rebounds, assists, threes).

Structure the model can learn:  stat = minutes x per-minute rate, adjusted for
  - pace and blowout risk (the game's closing total and spread, known before tip-off),
  - the opponent (points / rebounds / assists it allows, relative to the league),
  - usage redistribution: rotation teammates who are OUT tonight, especially newly out (news the line may lag),
  - schedule (back-to-back, rest) and the player's own form at several horizons.
Everything is computed from games BEFORE the one being predicted. Availability of teammates is the pre-game
inactive list (the same definition used everywhere else: coach's-decision DNPs count as available).

Without historical prop prices this cannot measure profit. What it measures is how much better than the simple
projections a lazy line would be built from (season average, last 10 games) the model is, and WHERE it is better.
Forward paper trading against real prop lines is the real test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import teams

TARGETS = ["pts", "reb", "ast", "fg3m"]
ROT_MIN = 15.0     # players with prop lines are mostly rotation players


def _ewm(df, col, hl, by="player_id"):
    """EWMA of `col` over each player's previous PLAYED games (NaN rows are skipped), shifted: no same-game info."""
    return df.groupby(by, sort=False)[col].transform(lambda s: s.shift().ewm(halflife=hl, min_periods=1).mean())


def build(pg: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """One row per player-game (all box-score rows, including those who did not play). Features are pre-game."""
    g = g[["game_id", "date", "home", "away", "home_pts", "away_pts", "spread", "total", "ml_home", "ml_away"]].copy()
    # older spreads are unsigned: sign by the moneyline favourite (expected home margin, + = home favoured)
    fav_home = pd.to_numeric(g["ml_home"], errors="coerce") < pd.to_numeric(g["ml_away"], errors="coerce")
    g["spread"] = pd.to_numeric(g["spread"], errors="coerce").abs() * np.where(fav_home, 1.0, -1.0)
    df = pg.drop(columns=[c for c in ("home", "away") if c in pg]).merge(
        g[["game_id", "home", "away", "spread", "total", "home_pts", "away_pts"]], on="game_id", how="inner")
    df = df.sort_values(["date", "game_id", "team"]).reset_index(drop=True)
    df["player_id"] = df["player_id"].astype(str)
    df["home_side"] = (df["team"] == df["home"]).astype(float)
    df["opp"] = np.where(df["home_side"] == 1, df["away"], df["home"])
    df["season"] = df["date"].map(teams.season_of)
    played = df["played"].astype(bool)
    stats = ["min", "pts", "reb", "ast", "fg3m", "fga", "fta", "tov", "oreb"]
    for c in stats:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    P = df[stats].where(played)           # per-game values only on nights the player played
    P["usage"] = P["fga"] + 0.44 * P["fta"] + P["tov"]
    for c in ["pts", "reb", "ast", "fg3m", "usage", "fga"]:
        P[f"{c}_pm"] = P[c] / P["min"].where(P["min"] >= 4)
    P["player_id"] = df["player_id"]
    P["season"] = df["season"]
    for c in ["min", "pts", "reb", "ast", "fg3m", "usage"]:
        df[f"{c}_e5"] = _ewm(P, c, 5)
        df[f"{c}_e20"] = _ewm(P, c, 20)
    for c in ["pts_pm", "reb_pm", "ast_pm", "fg3m_pm", "usage_pm", "fga_pm"]:
        df[f"{c}_e30"] = _ewm(P, c, 30)
    # simple projections a lazy line could be built from: season-to-date average and last 10 games
    for c in TARGETS + ["min"]:
        df[f"{c}_season"] = P.groupby(["player_id", "season"], sort=False)[c].transform(lambda s: s.shift().expanding().mean())
        df[f"{c}_last10"] = P.groupby("player_id", sort=False)[c].transform(lambda s: s.shift().rolling(10, min_periods=3).mean())
    df["games_before"] = P.groupby("player_id", sort=False)["min"].transform(lambda s: s.shift().notna().cumsum())
    df["games_season"] = P.groupby(["player_id", "season"], sort=False)["min"].transform(lambda s: s.shift().notna().cumsum())
    # did he play / was he available in his team's previous game?
    df["avail_f"] = df["avail"].astype(float)
    df["played_last"] = df.groupby("player_id", sort=False)["played"].transform(lambda s: s.astype(float).shift()).fillna(0)
    df["days_rest"] = df.groupby("player_id", sort=False)["date"].transform(lambda s: s.diff().dt.days).clip(upper=10).fillna(10)

    # ---- teammates out tonight (rotation players by recent minutes), and how many are NEWLY out
    rot = df["min_e20"].fillna(0) >= ROT_MIN
    out = (df["avail_f"] == 0) & rot
    new_out = out & (df.groupby("player_id", sort=False)["avail_f"].transform(lambda s: s.shift()).fillna(0) == 1)
    df["_u_out"] = np.where(out, df["usage_e20"].fillna(0), 0.0)
    df["_u_new"] = np.where(new_out, df["usage_e20"].fillna(0), 0.0)
    df["_m_out"] = np.where(out, df["min_e20"].fillna(0), 0.0)
    df["_r_out"] = np.where(out, df["reb_e20"].fillna(0), 0.0)
    df["_a_out"] = np.where(out, df["ast_e20"].fillna(0), 0.0)
    tg = df.groupby(["game_id", "team"])[["_u_out", "_u_new", "_m_out", "_r_out", "_a_out"]].transform("sum")
    df["mates_usage_out"] = tg["_u_out"] - df["_u_out"]
    df["mates_usage_new_out"] = tg["_u_new"] - df["_u_new"]
    df["mates_min_out"] = tg["_m_out"] - df["_m_out"]
    df["mates_reb_out"] = tg["_r_out"] - df["_r_out"]
    df["mates_ast_out"] = tg["_a_out"] - df["_a_out"]
    df["usage_share"] = df["usage_e20"] / (df["usage_e20"] + 1)

    # ---- opponent: what it allows per game vs the league, EWMA over its previous games
    tm = df[played].groupby(["game_id", "team", "opp", "date"])[["pts", "reb", "ast", "fg3m"]].sum().reset_index()
    tm = tm.sort_values("date")
    for c in ["pts", "reb", "ast", "fg3m"]:
        daily = tm.groupby("date")[c].mean()
        lg = tm["date"].map(daily.shift().ewm(halflife=60, min_periods=1).mean())   # league level, earlier days only
        tm[f"opp_allow_{c}"] = tm.groupby("opp")[c].transform(lambda s: s.shift().ewm(halflife=15, min_periods=1).mean()) / lg
    df = df.merge(tm[["game_id", "opp"] + [f"opp_allow_{c}" for c in ["pts", "reb", "ast", "fg3m"]]],
                  on=["game_id", "opp"], how="left")
    # ---- market context: pace and blowout risk
    sgn = np.where(df["home_side"] == 1, 1.0, -1.0)
    df["team_spread"] = pd.to_numeric(df["spread"], errors="coerce") * sgn   # expected margin for this player's team
    df["abs_spread"] = df["team_spread"].abs()
    df["total"] = pd.to_numeric(df["total"], errors="coerce")
    df["b2b"] = (df["days_rest"] == 1).astype(float)
    return df.drop(columns=[c for c in df.columns if c.startswith("_")])


FEATS = (["min_e5", "min_e20", "pts_e5", "pts_e20", "reb_e5", "reb_e20", "ast_e5", "ast_e20", "fg3m_e5", "fg3m_e20",
          "usage_e5", "usage_e20", "pts_pm_e30", "reb_pm_e30", "ast_pm_e30", "fg3m_pm_e30", "usage_pm_e30", "fga_pm_e30",
          "pts_season", "min_season", "games_before", "games_season", "played_last", "days_rest", "b2b",
          "mates_usage_out", "mates_usage_new_out", "mates_min_out", "mates_reb_out", "mates_ast_out",
          "opp_allow_pts", "opp_allow_reb", "opp_allow_ast", "opp_allow_fg3m",
          "team_spread", "abs_spread", "total", "home_side"])


def _lgb():
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=63, min_child_samples=300,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)


def eligible(df: pd.DataFrame) -> np.ndarray:
    """Player-games that would plausibly have a prop line: rotation player, played, 5+ earlier games."""
    return (df["played"].astype(bool) & (df["min_e20"].fillna(0) >= ROT_MIN) & (df["games_before"] >= 5)
            & df["total"].notna()).to_numpy()


def walk_forward(df: pd.DataFrame, target="pts", first_test="2014-15"):
    """LightGBM refit each season on all earlier seasons. Returns predictions for eligible rows of test seasons."""
    el = eligible(df)
    pred = np.full(len(df), np.nan)
    for s in sorted(df["season"].unique()):
        if s < first_test:
            continue
        tr = el & (df["season"] < s).to_numpy()
        te = el & (df["season"] == s).to_numpy()
        if te.sum() == 0 or tr.sum() < 20000:
            continue
        m = _lgb().fit(df.loc[tr, FEATS], df.loc[tr, target])
        pred[te] = m.predict(df.loc[te, FEATS])
    return pred


def evaluate(df: pd.DataFrame, pred, target="pts"):
    """Model vs simple projections. Also: does the model's edge concentrate where teammates are newly out?"""
    m = ~np.isnan(pred) & df[f"{target}_season"].notna().to_numpy() & df[f"{target}_last10"].notna().to_numpy()
    y = df.loc[m, target].to_numpy(float)
    P = {"season average": df.loc[m, f"{target}_season"].to_numpy(float),
         "last 10 games": df.loc[m, f"{target}_last10"].to_numpy(float),
         "recent form (EWMA)": df.loc[m, f"{target}_e20"].to_numpy(float),
         "model": pred[m]}
    out = {"target": target, "n": int(m.sum()), "from": str(df.loc[m, "season"].min()), "to": str(df.loc[m, "season"].max()),
           "mae": {k: float(np.mean(np.abs(v - y))) for k, v in P.items()},
           "rmse": {k: float(np.sqrt(np.mean((v - y) ** 2))) for k, v in P.items()}}
    # where a line built from the season average would be most wrong: teammates newly out
    newo = df.loc[m, "mates_usage_new_out"].to_numpy(float) >= 8
    out["teammate_newly_out"] = {"n": int(newo.sum()),
                                 "mae": {k: float(np.mean(np.abs(v[newo] - y[newo]))) for k, v in P.items()},
                                 "bias": {k: float(np.mean(v[newo] - y[newo])) for k, v in P.items()}}
    # how often the model and the season average disagree by enough to matter, and who is closer then
    base = P["season average"]
    for thr in (2.0, 3.0, 4.0) if target == "pts" else (1.0, 1.5, 2.0):
        dis = np.abs(pred[m] - base) >= thr
        if dis.sum():
            closer = np.abs(pred[m][dis] - y[dis]) < np.abs(base[dis] - y[dis])
            right_side = np.sign(y[dis] - base[dis]) == np.sign(pred[m][dis] - base[dis])
            out.setdefault("disagree", {})[f"{thr:g}+"] = {
                "n": int(dis.sum()), "share_of_games": float(dis.mean()),
                "model_closer": float(closer.mean()),
                "right_side_of_season_avg": float(right_side[y[dis] != base[dis]].mean())}
    # spread of outcomes around the projection, for turning a projection into over/under probabilities later
    r = y - pred[m]
    out["resid_sd"] = float(np.std(r))
    return out


def study(g: pd.DataFrame, pg: pd.DataFrame):
    """All four targets, walk-forward since 2014-15, plus the ESPN-era slice (2024-25 on) on its own."""
    df = build(pg, g)
    res = {}
    for t in TARGETS:
        p = walk_forward(df, t)
        res[t] = evaluate(df, p, t)
        recent = (df["season"] >= "2024-25").to_numpy()
        p2 = np.where(recent, p, np.nan)
        if np.isfinite(p2).sum() > 1000:
            res[t]["recent"] = {k: v for k, v in evaluate(df, p2, t).items() if k in ("n", "mae", "teammate_newly_out", "disagree")}
    return res, df
