"""Referee crews. Crews are announced around 9am Eastern on game day, after most opening lines and before the close.

For every official we keep a running, shrunk average of how far games he worked finished from the market:
    tot_resid  = actual total points - closing total     (does his crew make games run over/under the market?)
    home_resid = actual home margin  - closing spread    (spread is the expected home margin, positive = home favoured)
Only earlier games are used (as of the morning of each game). A crew's effect is the sum of its members' shrunk
averages divided by the crew size, so a game with three neutral refs scores 0.

Shrinkage: an official's average is sum(resid) / (n + K). With residual SD around 18 points on totals, 150 games
gives a standard error near 1.5 points, about the size of the real crew differences reported elsewhere, so K is
large and the study below decides whether anything survives.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

K_TOT = 150.0
K_HOME = 150.0


def crew_features(games: pd.DataFrame, officials: pd.DataFrame):
    """games: game_id, date, home_pts, away_pts, tot_close, spread_close (home spread, negative = home favoured).
    officials: game_id, ref. Returns game_id, crew_n, crew_tot, crew_home, crew_games (min games among crew)."""
    crews = officials.groupby("game_id")["ref"].apply(list).to_dict()
    s_tot, s_home, n = defaultdict(float), defaultdict(float), defaultdict(int)
    out = []
    for _, day in games.sort_values("date").groupby("date", sort=True):
        upd = []
        for r in day.itertuples(index=False):
            crew = crews.get(r.game_id, [])
            if crew:
                ct = sum(s_tot[x] / (n[x] + K_TOT) for x in crew) / len(crew)
                ch = sum(s_home[x] / (n[x] + K_HOME) for x in crew) / len(crew)
                out.append((r.game_id, len(crew), ct, ch, min(n[x] for x in crew)))
            else:
                out.append((r.game_id, 0, np.nan, np.nan, 0))
            if crew and pd.notna(r.home_pts) and pd.notna(r.tot_close):
                tr = (r.home_pts + r.away_pts) - r.tot_close
                hr = (r.home_pts - r.away_pts) - (r.spread_close if pd.notna(r.spread_close) else 0.0)
                upd.append((crew, tr, hr if pd.notna(r.spread_close) else 0.0))
        for crew, tr, hr in upd:
            for x in crew:
                s_tot[x] += tr
                s_home[x] += hr
                n[x] += 1
    f = pd.DataFrame(out, columns=["game_id", "crew_n", "crew_tot", "crew_home", "crew_games"])
    table = pd.DataFrame({"ref": list(n), "games": [n[x] for x in n],
                          "tot_effect": [s_tot[x] / (n[x] + K_TOT) for x in n],
                          "home_effect": [s_home[x] / (n[x] + K_HOME) for x in n]}).sort_values("games", ascending=False)
    return f, table


def study(d: pd.DataFrame, officials: pd.DataFrame, boot, ols):
    """d: the opening-line study frame (open/close totals, results, total_pred). Tests, in order:
       1. Do crew effects (from earlier games only) predict where games finish relative to the CLOSING total?
       2. Does the market already move the total toward the crew between open and close?
       3. Paper bets at the close: model total + crew effect vs the closing total."""
    from . import totals as TT
    from .evaluate import decimal_odds
    g = d[["game_id", "date", "home_pts", "away_pts", "tot_open", "tot_close", "spread_close"]].copy()
    f, table = crew_features(g, officials)
    t = d.merge(f, on="game_id", how="left")
    t = t[(t["crew_n"] >= 2) & t["tot_close"].notna()]
    t = t[t["tot_close"].between(180, 280) & t["tot_open"].between(180, 280) & (t["crew_games"] >= 20)]
    if len(t) < 300:
        return {"n_games": int(len(t)), "note": "not enough games with referee data yet"}
    act = (t["home_pts"] + t["away_pts"]).to_numpy(float)
    to, tc = t["tot_open"].to_numpy(float), t["tot_close"].to_numpy(float)
    ce = t["crew_tot"].to_numpy(float)
    nights = t["date"].dt.strftime("%Y-%m-%d").to_numpy()
    slope_close, p_close = ols(act - tc, ce, nights)
    slope_move, p_move = ols(tc - to, ce, nights)
    hm = (t["home_pts"] - t["away_pts"]).to_numpy(float) - t["spread_close"].fillna(0).to_numpy(float)
    slope_home, p_home = ols(hm, t["crew_home"].to_numpy(float), nights)
    q = np.quantile(ce, [0.2, 0.8])
    hi, lo = ce >= q[1], ce <= q[0]
    res = {
        "n_games": int(len(t)), "n_refs": int(len(table)),
        "crew_sd_pts": float(np.std(ce)),
        "vs_close": {"slope": slope_close, "p_value": p_close,
                     "note": "1.0 = crew effect fully shows up in results beyond the closing total; 0 = none"},
        "market_move": {"slope": slope_move, "p_value": p_move,
                        "note": "positive = the total moves toward the crew between open and close"},
        "home_margin": {"slope": slope_home, "p_value": p_home},
        "top_vs_bottom_fifth": {"over_close_rate_top": float(np.mean(act[hi] > tc[hi])),
                                "over_close_rate_bottom": float(np.mean(act[lo] > tc[lo])),
                                "resid_top": float(np.mean(act[hi] - tc[hi])),
                                "resid_bottom": float(np.mean(act[lo] - tc[lo]))},
        "bets": {},
    }
    if "total_pred" in t:
        pr = t["total_pred"].to_numpy(float) + ce
        dec = lambda a: np.where(np.isnan(a), 1 + 100 / 110, decimal_odds(np.nan_to_num(a, nan=-110)))  # noqa: E731
        po, pu = dec(t["over_close"].to_numpy(float)), dec(t["under_close"].to_numpy(float))
        for qq in (0.8, 0.9):                                  # crew alone: bet the most extreme crews at the close
            e = float(np.quantile(np.abs(ce), qq))
            side, pnl, _ = TT.ou_bets(tc + ce, tc, act, e, po, pu)
            res["bets"][f"close | crew alone, most extreme {int(round((1 - qq) * 100))}% (|effect| >= {e:.2f})"] = \
                TT.summary(side, pnl, nights)
        for e in (3, 4):
            side, pnl, _ = TT.ou_bets(pr, tc, act, e, po, pu)
            res["bets"][f"close | model + crew {e}+ pts off"] = TT.summary(side, pnl, nights)
            side0, pnl0, _ = TT.ou_bets(t["total_pred"].to_numpy(float), tc, act, e, po, pu)
            res["bets"][f"close | model alone {e}+ pts off"] = TT.summary(side0, pnl0, nights)
    res["effects"] = {x.ref: [round(float(x.tot_effect), 3), round(float(x.home_effect), 3), int(x.games)]
                      for x in table.itertuples(index=False)}
    res["table_top"] = table[table["games"] >= 60].assign(
        tot_effect=lambda x: x["tot_effect"].round(2), home_effect=lambda x: x["home_effect"].round(2)
    ).sort_values("tot_effect").iloc[np.r_[0:5, -5:0]].to_dict("records") if (table["games"] >= 60).sum() >= 10 else []
    return res
