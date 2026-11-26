"""First-half and first-quarter totals: are they priced mechanically from the full-game total?

Books often derive half and quarter lines from the full-game line with a near-fixed ratio. If some teams reliably
start fast or slow (rotation habits, coaching, pace that fades), a fixed ratio misprices their games in a
predictable direction. This study asks, walk-forward:
  1. How much better than "league ratio x closing total" does a team-aware model predict the actual 1H / 1Q total?
  2. Do team start-speed tendencies persist (first half of a season vs the second half)?
  3. If a line were set by the fixed ratio, how often would the team-aware model land on the right side?
Without historical half-time prices (3) is an upper bound: real books adjust for some of this. Forward tracking
against real first-half lines is the real test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import teams


def frame(g: pd.DataFrame, ls: pd.DataFrame) -> pd.DataFrame:
    """One row per game with the closing total/spread and actual Q1, 1H and full-game totals per team."""
    ls = ls.copy()
    ls["period"] = ls["period"].astype(int)
    q = ls.pivot_table(index=["game_id", "team"], columns="period", values="pts", aggfunc="sum")
    q = q.reindex(columns=[1, 2, 3, 4], fill_value=np.nan)
    q = q.rename(columns={1: "q1", 2: "q2", 3: "q3", 4: "q4"}).reset_index()
    d = g[["game_id", "date", "home", "away", "home_pts", "away_pts", "spread", "total"]].dropna(subset=["home_pts", "total"])
    h = q.rename(columns={c: f"{c}_h" for c in ["q1", "q2", "q3", "q4"]}).rename(columns={"team": "home"})
    a = q.rename(columns={c: f"{c}_a" for c in ["q1", "q2", "q3", "q4"]}).rename(columns={"team": "away"})
    d = d.merge(h, on=["game_id", "home"]).merge(a, on=["game_id", "away"])
    d = d.dropna(subset=["q1_h", "q2_h", "q1_a", "q2_a"])
    d["q1"] = d["q1_h"] + d["q1_a"]
    d["h1"] = d["q1"] + d["q2_h"] + d["q2_a"]
    d["h1_home"], d["h1_away"] = d["q1_h"] + d["q2_h"], d["q1_a"] + d["q2_a"]
    # sanity: quarter sums should match the final score (allowing for overtime)
    reg = d["h1"] + d["q3_h"] + d["q3_a"] + d["q4_h"] + d["q4_a"]
    d = d[(reg <= d["home_pts"] + d["away_pts"] + 0.5) & (reg >= d["home_pts"] + d["away_pts"] - 40)]
    d["season"] = d["date"].map(teams.season_of)
    return d.sort_values("date").reset_index(drop=True)


def features(d: pd.DataFrame, hl=25):
    """League ratios from earlier days, and each team's start-speed tendency: EWMA of how far its games' 1H (and Q1)
    totals came in above the league ratio x closing total, split into offence (its own 1H points) and defence."""
    d = d.copy()
    for k in ("h1", "q1"):
        daily = (d[k] / d["total"]).groupby(d["date"]).mean()
        d[f"r_{k}"] = d["date"].map(daily.shift().ewm(halflife=60, min_periods=5).mean())
        d[f"fixed_{k}"] = d[f"r_{k}"] * d["total"]
        d[f"res_{k}"] = d[k] - d[f"fixed_{k}"]
    # per-team residual series: own 1H points vs half of the fixed 1H total
    rows = []
    for side, oth in (("home", "away"), ("away", "home")):
        rows.append(pd.DataFrame({"game_id": d["game_id"], "date": d["date"], "team": d[side], "side": side,
                                  "off": d[f"h1_{side}"] - d["fixed_h1"] / 2, "deff": d[f"h1_{oth}"] - d["fixed_h1"] / 2,
                                  "q1res": d["res_q1"] / 2}))
    T = pd.concat(rows).sort_values(["date", "game_id"])
    for c in ("off", "deff", "q1res"):
        T[f"{c}_e"] = T.groupby("team")[c].transform(lambda s: s.shift().ewm(halflife=hl, min_periods=5).mean())
    T["n_prev"] = T.groupby("team").cumcount()
    for side in ("home", "away"):
        t = T[T["side"] == side].set_index("game_id")
        for c in ("off_e", "deff_e", "q1res_e", "n_prev"):
            d[f"{c}_{side}"] = d["game_id"].map(t[c])
    d["team_adj_h1"] = d[["off_e_home", "deff_e_home", "off_e_away", "deff_e_away"]].sum(axis=1, min_count=4)
    d["team_adj_q1"] = d[["q1res_e_home", "q1res_e_away"]].sum(axis=1, min_count=2)
    d["abs_spread"] = pd.to_numeric(d["spread"], errors="coerce").abs()
    return d


def _ridge(X, y, lam=50.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    P = np.eye(Z.shape[1]) * lam
    P[0, 0] = 0
    b = np.linalg.solve(Z.T @ Z + P, Z.T @ y)
    return lambda Xn: np.column_stack([np.ones(len(Xn)), (Xn - mu) / sd]) @ b


def study(g: pd.DataFrame, ls: pd.DataFrame, first_test="2014-15"):
    from .openline import _boot
    d = features(frame(g, ls))
    out = {"n_games": int(len(d)), "from": str(d["date"].min().date()) if len(d) else None,
           "to": str(d["date"].max().date()) if len(d) else None}
    if len(d) < 3000:
        out["note"] = "not enough quarter scores yet"
        return out
    for k in ("h1", "q1"):
        cols = [f"team_adj_{k}", "abs_spread"]
        pred = np.full(len(d), np.nan)
        for s in sorted(d["season"].unique()):
            if s < first_test:
                continue
            ok = d[cols + [f"fixed_{k}"]].notna().all(axis=1).to_numpy()
            tr = ok & (d["season"] < s).to_numpy()
            te = ok & (d["season"] == s).to_numpy() & (d[["n_prev_home", "n_prev_away"]].min(axis=1) >= 10).to_numpy()
            if te.sum() == 0 or tr.sum() < 2000:
                continue
            f = _ridge(d.loc[tr, cols].to_numpy(float), d.loc[tr, f"res_{k}"].to_numpy(float))
            pred[te] = d.loc[te, f"fixed_{k}"].to_numpy(float) + f(d.loc[te, cols].to_numpy(float))
        m = ~np.isnan(pred)
        y, fx, pr = d.loc[m, k].to_numpy(float), d.loc[m, f"fixed_{k}"].to_numpy(float), pred[m]
        nights = d.loc[m, "date"].dt.strftime("%Y-%m-%d").to_numpy()
        r = {"n": int(m.sum()), "mae_fixed": float(np.mean(np.abs(y - fx))), "mae_team": float(np.mean(np.abs(y - pr))),
             "mae_gain_ci": _boot(np.abs(y - fx) - np.abs(y - pr), nights), "ratio_last": float(d[f"r_{k}"].iloc[-1])}
        # pseudo-line = fixed ratio rounded to the nearest half point; bet the team-aware side when they differ
        line = np.round(fx * 2) / 2
        for thr in ((1.0, 1.5, 2.0) if k == "h1" else (0.5, 1.0, 1.5)):
            side = np.where(pr - line >= thr, 1, np.where(line - pr >= thr, -1, 0))
            k_ = (side != 0) & (y != line)
            if k_.sum():
                won = np.sign(y[k_] - line[k_]) == side[k_]
                r.setdefault("vs_fixed_line", {})[f"{thr:g}+ pts"] = {
                    "n": int(k_.sum()), "win_rate": float(won.mean()),
                    "roi_at_110": float(np.mean(np.where(won, 100 / 110, -1.0))),
                    "roi_ci": _boot(np.where(won, 100 / 110, -1.0), nights[k_])}
        out[k] = r
    # persistence of start speed: team tendency in the first half of a season vs the second half
    T = []
    for s, q in d.groupby("season"):
        q = q.sort_values("date")
        mid = q["date"].iloc[len(q) // 2]
        for half, qq in (("a", q[q["date"] < mid]), ("b", q[q["date"] >= mid])):
            for side, oth in (("home", "away"), ("away", "home")):
                T.append(pd.DataFrame({"season": s, "half": half, "team": qq[side],
                                       "v": qq[f"h1_{side}"] - qq["fixed_h1"] / 2}))
    T = pd.concat(T).groupby(["season", "team", "half"])["v"].mean().unstack("half").dropna()
    out["persistence_corr"] = float(np.corrcoef(T["a"], T["b"])[0, 1]) if len(T) > 20 else None
    out["team_spread_pts"] = float(T["a"].std()) if len(T) else None
    return out
