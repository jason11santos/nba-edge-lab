"""Totals (over/under) model: predict combined points, compare with the posted total.

Features (all known before tip-off)
  * each team's recent points scored and allowed (exponentially weighted, regressed toward the league each summer)
  * "market pace": the average closing total in each team's recent games, the market's own view of how its games play
  * the league's recent average total (scoring has risen ~20 points since 2015)
  * rest, back-to-backs, altitude, travel, point in season, tonight's lineup strength and missing minutes

The model is a ridge regression on the points total itself (it never sees tonight's line), refit on a schedule.
Bets compare its number with the posted total: over when the model is at least `edge_pts` higher, under when lower.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

FEATS = ["exp_pts", "mkt_pace", "lg_total", "off_sum", "def_sum", "b2b_home", "b2b_away", "rest_home", "rest_away",
         "altitude", "travel_km_away", "season_frac", "str_sum", "miss_sum"]
BREAKEVEN_110 = 110 / 210   # win rate needed at -110 on both sides: 52.38%


def team_features(g: pd.DataFrame, halflife=10, carry=0.7):
    """Per game, before tip-off: each side's EWMA points for/against and the EWMA of closing totals in its games."""
    a = 1 - 0.5 ** (1 / halflife)
    pf, pa, mk = defaultdict(lambda: np.nan), defaultdict(lambda: np.nan), defaultdict(lambda: np.nan)
    last_season = {}
    lg_hist = []
    out = np.full((len(g), 7), np.nan)
    H, A, S = g["home"].to_numpy(), g["away"].to_numpy(), g["season"].to_numpy()
    hp, ap, tot = g["home_pts"].to_numpy(float), g["away_pts"].to_numpy(float), g["total"].to_numpy(float)
    dates = g["date"].to_numpy()
    lg_by_date, day_buf, cur_day = {}, [], None
    for i in range(len(g)):
        if dates[i] != cur_day:  # league average total as of the end of the previous day
            lg_hist.extend(day_buf)
            day_buf, cur_day = [], dates[i]
            lg_by_date[cur_day] = np.mean(lg_hist[-600:]) if len(lg_hist) >= 100 else np.nan
        lg = lg_by_date[cur_day]
        h, aw, s = H[i], A[i], S[i]
        for t in (h, aw):
            if last_season.get(t) not in (None, s) and not np.isnan(lg):
                half = lg / 2
                pf[t] = half + carry * (pf[t] - half)
                pa[t] = half + carry * (pa[t] - half)
                mk[t] = lg + carry * (mk[t] - lg)
            last_season[t] = s
        out[i] = (pf[h], pa[h], pf[aw], pa[aw], mk[h], mk[aw], lg)
        if np.isnan(hp[i]) or np.isnan(ap[i]):
            continue
        for t, f, ag in ((h, hp[i], ap[i]), (aw, ap[i], hp[i])):
            pf[t] = f if np.isnan(pf[t]) else (1 - a) * pf[t] + a * f
            pa[t] = ag if np.isnan(pa[t]) else (1 - a) * pa[t] + a * ag
            if not np.isnan(tot[i]):
                mk[t] = tot[i] if np.isnan(mk[t]) else (1 - a) * mk[t] + a * tot[i]
        day_buf.append(hp[i] + ap[i])
    f = pd.DataFrame(out, columns=["pf_h", "pa_h", "pf_a", "pa_a", "mk_h", "mk_a", "lg_total"], index=g.index)
    f["exp_pts"] = (f["pf_h"] + f["pa_a"]) / 2 + (f["pf_a"] + f["pa_h"]) / 2
    f["mkt_pace"] = (f["mk_h"] + f["mk_a"]) / 2
    f["off_sum"] = f["pf_h"] + f["pf_a"] - f["lg_total"]
    f["def_sum"] = f["pa_h"] + f["pa_a"] - f["lg_total"]
    return f


def design(X: pd.DataFrame) -> pd.DataFrame:
    T = team_features(X)
    D = pd.DataFrame(index=X.index)
    for c in ["exp_pts", "mkt_pace", "lg_total", "off_sum", "def_sum"]:
        D[c] = T[c]
    for c in ["b2b_home", "b2b_away", "rest_home", "rest_away", "altitude", "travel_km_away", "season_frac"]:
        D[c] = X[c] if c in X else np.nan
    D["rest_home"], D["rest_away"] = D["rest_home"].clip(upper=4), D["rest_away"].clip(upper=4)
    sh, sa = X.get("str_gmsc_home"), X.get("str_gmsc_away")
    D["str_sum"] = (sh + sa) if sh is not None and sa is not None else np.nan
    mh, ma = X.get("miss_min_home"), X.get("miss_min_away")
    D["miss_sum"] = (mh + ma) if mh is not None and ma is not None else np.nan
    return D


class TotalsModel:
    """Ridge regression for combined points. Missing lineup features are filled with training means."""

    def __init__(self, lam=10.0, halflife_days=900):
        self.lam, self.hl = lam, halflife_days

    def fit(self, D: pd.DataFrame, y: np.ndarray, dates: pd.Series, asof):
        ok = ~np.isnan(y) & D[["exp_pts", "mkt_pace", "lg_total"]].notna().all(1).to_numpy()
        A = D[ok]
        self.mu, self.sd = A.mean().fillna(0), A.std().replace(0, 1).fillna(1)
        Z = np.nan_to_num(((A.fillna(self.mu) - self.mu) / self.sd).to_numpy())
        w = 0.5 ** ((asof - dates[ok]).dt.days.to_numpy() / self.hl)
        Zc = np.column_stack([np.ones(len(Z)), Z])
        W = Zc * w[:, None]
        P = np.eye(Zc.shape[1]) * self.lam
        P[0, 0] = 0
        self.beta = np.linalg.solve(Zc.T @ W + P, W.T @ y[ok])
        self.resid_sd = float(np.sqrt(np.average((y[ok] - Zc @ self.beta) ** 2, weights=w)))
        return self

    def predict(self, D: pd.DataFrame) -> np.ndarray:
        Z = np.nan_to_num(((D.fillna(self.mu) - self.mu) / self.sd).to_numpy())
        return self.beta[0] + Z @ self.beta[1:]

    def weights(self):
        return dict(zip(["intercept"] + list(self.mu.index), np.round(self.beta, 3).tolist()))


def walk_forward(X: pd.DataFrame, test_from: str, refit_days=30, train_from="2008-08-01"):
    """Model totals for every finished game from test_from on, refitting monthly on earlier games only."""
    X = X.sort_values(["date", "game_id"]).reset_index(drop=True)
    D = design(X)
    y = (X["home_pts"] + X["away_pts"]).to_numpy(float)
    pred = np.full(len(X), np.nan)
    m, last = TotalsModel(), None
    idx = np.where((X["date"] >= pd.Timestamp(test_from)).to_numpy())[0]
    for d in sorted(X.loc[idx, "date"].unique()):
        d = pd.Timestamp(d)
        if last is None or (d - last).days >= refit_days:
            tr = ((X["date"] < d) & (X["date"] >= pd.Timestamp(train_from))).to_numpy()
            if (tr & ~np.isnan(y)).sum() < 1000:
                continue  # not enough history yet
            m.fit(D[tr], y[tr], X.loc[tr, "date"], d)
            last = d
        rows = (X["date"] == d).to_numpy()
        pred[rows] = m.predict(D[rows])
    return X.assign(total_pred=pred, total_actual=y), m


def ou_bets(pred, line, actual, edge_pts, price_over=None, price_under=None):
    """Over if pred >= line + edge, under if pred <= line - edge. Returns side (+1 over, -1 under, 0 none),
    profit per $1 (pushes return 0), and the decimal odds used (-110 when no price is given)."""
    side = np.where(pred - line >= edge_pts, 1, np.where(line - pred >= edge_pts, -1, 0))
    po = np.full(len(line), 1 + 100 / 110) if price_over is None else price_over
    pu = np.full(len(line), 1 + 100 / 110) if price_under is None else price_under
    dec = np.where(side == 1, po, pu)
    diff = actual - line
    win = np.where(side == 1, diff > 0, diff < 0)
    push = diff == 0
    pnl = np.where(side == 0, np.nan, np.where(push, 0.0, np.where(win, dec - 1, -1.0)))
    return side, pnl, dec


def summary(side, pnl, nights):
    from .openline import _boot
    m = (side != 0) & ~np.isnan(pnl)
    if m.sum() == 0:
        return {"n": 0}
    decided = pnl[m] != 0
    return {"n": int(m.sum()), "win_rate": float(np.mean(pnl[m][decided] > 0)) if decided.any() else None,
            "roi": float(np.mean(pnl[m])), "roi_ci": _boot(pnl[m], np.asarray(nights)[m])}


def backtest_vs_close(X: pd.DataFrame, test_from="2013-10-01"):
    """Model vs the closing total, 2013 on. History has no over/under prices, so -110 both sides is assumed."""
    P, m = walk_forward(X, test_from)
    T = P[(P["date"] >= test_from) & P["total"].notna() & P["total_actual"].notna() & P["total_pred"].notna()]
    y, line, pr = T["total_actual"].to_numpy(), T["total"].to_numpy(), T["total_pred"].to_numpy()
    nights = T["date"].dt.strftime("%Y-%m-%d").to_numpy()
    out = {"n_games": int(len(T)), "from": str(T["date"].min().date()), "to": str(T["date"].max().date()),
           "mae_line": float(np.mean(np.abs(y - line))), "mae_model": float(np.mean(np.abs(y - pr))),
           "mae_blend": float(np.mean(np.abs(y - (0.8 * line + 0.2 * pr)))),
           "corr_model_minus_line_vs_result_minus_line": float(np.corrcoef(pr - line, y - line)[0, 1]),
           "breakeven": BREAKEVEN_110, "bets": {}}
    for e in (2, 3, 4, 6):
        side, pnl, _ = ou_bets(pr, line, y, e)
        out["bets"][f"close | model {e}+ pts off"] = summary(side, pnl, nights)
    out["weights_last"] = m.weights()
    return out, P
