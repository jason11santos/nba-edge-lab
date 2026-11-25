"""Opening-line study: can the model beat the OPENING price, and does the line move toward the model?

    py -m nba_edge.openline           fetch opening/closing lines from ESPN (2023-24 on), then run the study
    py -m nba_edge.openline fetch     only fetch (safe to re-run; it resumes)
    py -m nba_edge.openline report    only run the study on what was fetched

Why: the main backtest used near-closing lines, which already contain the injury news. Lines open many
hours earlier. If the model has an edge, it should show up here, and the cleanest sign of it is that the
line moves TOWARD the model between open and close (closing-line value, CLV).

Rules fixed before looking at results (no tuning on this data):
  * The model may use only what is known when the line opens: results, Elo, schedule, and the lineup
    from each team's previous game. Tonight's injury report is NOT used, because at open it is unknown.
    (A second, clearly labelled "ceiling" run shows what perfect injury knowledge would be worth.)
  * The blend of model and opening price is refit walk-forward on earlier games only, monthly.
  * Main bet rule: expected value above 2% at the opening price; other thresholds shown for transparency.
  * Confidence intervals resample whole game nights.

Writes data/openline.sqlite (lines from every book ESPN lists) and web/openline.json, and prints a report.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import features as F
from . import models as M
from .evaluate import decimal_odds, implied
from .sources import espn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "openline.sqlite")
START = date(2023, 10, 24)
PREFERRED = ["DraftKings", "Draft Kings", "ESPN BET", "Caesars Sportsbook (New Jersey)", "MGM"]

SCHEMA = """
create table if not exists ol_games (game_id text primary key, espn_id text, date text, home text, away text,
  home_pts real, away_pts real, fetched int default 0);
create table if not exists ol_lines (game_id text, book text, ml_home_open real, ml_away_open real,
  ml_home_close real, ml_away_close real, spread_close real, total_close real, primary key (game_id, book));
"""


def _con():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    c = sqlite3.connect(DB, timeout=60)
    c.executescript(SCHEMA)
    return c


# --------------------------------------------------------------------------- fetch

def _all_books(espn_id):
    js = espn._get(f"{espn.CORE}/events/{espn_id}/competitions/{espn_id}/odds")
    out = []
    for it in js.get("items") or []:
        book = ((it.get("provider") or {}).get("name") or "").strip()
        if not book or "live" in book.lower() or "accuscore" in book.lower():
            continue  # in-game prices and model sites are not opening lines
        h, a = it.get("homeTeamOdds") or {}, it.get("awayTeamOdds") or {}
        am = lambda d, w: espn._num(((d.get(w) or {}).get("moneyLine") or {}).get("american"))  # noqa: E731
        sp = it.get("spread")
        out.append((book, am(h, "open"), am(a, "open"), am(h, "close") or espn._num(h.get("moneyLine")),
                    am(a, "close") or espn._num(a.get("moneyLine")), None if sp is None else -float(sp),
                    espn._num(it.get("overUnder"))))
    return out


def fetch(end: date | None = None):
    con = _con()
    end = end or date.today() - timedelta(days=1)
    have_days = {r[0] for r in con.execute("select distinct date from ol_games")}
    d, days = START, []
    while d <= end:
        if d.isoformat() not in have_days or (end - d).days < 3:
            days.append(d)
        d += timedelta(days=1)
    print(f"scoreboards to fetch: {len(days)}", flush=True)

    def day_games(dd):
        try:
            return [g for g in espn.scoreboard(dd) if g["state"] == "post" and g["season_type"] in (2, 3, 5)]
        except Exception as e:  # noqa: BLE001
            print("scoreboard failed", dd, e, flush=True)
            return []

    with ThreadPoolExecutor(6) as ex:
        for i, games in enumerate(ex.map(day_games, days)):
            con.executemany("insert or ignore into ol_games (game_id, espn_id, date, home, away, home_pts, away_pts)"
                            " values (?,?,?,?,?,?,?)",
                            [(g["game_id"], g["espn_id"], g["date"], g["home"], g["away"], g["home_pts"], g["away_pts"])
                             for g in games])
            if i % 60 == 0:
                con.commit()
                print(f"  scoreboards {i}/{len(days)}", flush=True)
    con.commit()
    todo = con.execute("select game_id, espn_id from ol_games where fetched=0").fetchall()
    print(f"games needing lines: {len(todo)}", flush=True)

    def lines(row):
        gid, eid = row
        try:
            return gid, _all_books(eid)
        except Exception as e:  # noqa: BLE001
            return gid, e

    with ThreadPoolExecutor(8) as ex:
        for i, (gid, res) in enumerate(ex.map(lines, todo)):
            if isinstance(res, Exception):
                continue
            con.executemany("insert or replace into ol_lines values (?,?,?,?,?,?,?,?)", [(gid, *r) for r in res])
            con.execute("update ol_games set fetched=1 where game_id=?", (gid,))
            if i % 200 == 0:
                con.commit()
                print(f"  lines {i}/{len(todo)}", flush=True)
    con.commit()
    n = con.execute("select count(*), count(distinct game_id) from ol_lines").fetchone()
    print(f"stored {n[0]:,} book lines for {n[1]:,} games", flush=True)


# --------------------------------------------------------------------------- data

def _fair(mh, ma):
    ih, ia = implied(mh), implied(ma)
    s = ih + ia
    ok = (np.abs(mh) >= 100) & (np.abs(ma) >= 100) & (s > 0.99) & (s < 1.15)
    return np.where(ok, ih / s, np.nan)


def load_lines():
    con = _con()
    g = pd.read_sql("select * from ol_games where fetched=1", con)
    ln = pd.read_sql("select * from ol_lines", con)
    con.close()
    ln = ln.dropna(subset=["ml_home_open", "ml_away_open", "ml_home_close", "ml_away_close"])
    ln["p_open"] = _fair(ln["ml_home_open"].to_numpy(), ln["ml_away_open"].to_numpy())
    ln["p_close"] = _fair(ln["ml_home_close"].to_numpy(), ln["ml_away_close"].to_numpy())
    # A symmetric -110/-110 style "open" followed by a big move is usually a placeholder, not a real price.
    placeholder = (ln["ml_home_open"] == ln["ml_away_open"]) & ((ln["p_open"] - ln["p_close"]).abs() > 0.15)
    ln = ln[~placeholder & ln["p_open"].notna() & ln["p_close"].notna()]
    rank = {b: i for i, b in enumerate(PREFERRED)}
    ln["rank"] = ln["book"].map(rank).fillna(99)
    ref = ln.sort_values(["game_id", "rank"]).drop_duplicates("game_id")  # one reference book per game
    best = ln.groupby("game_id").agg(
        best_h_open=("ml_home_open", lambda s: s.iloc[np.argmax(decimal_odds(s))]),
        best_a_open=("ml_away_open", lambda s: s.iloc[np.argmax(decimal_odds(s))]),
        best_h_close=("ml_home_close", lambda s: s.iloc[np.argmax(decimal_odds(s))]),
        best_a_close=("ml_away_close", lambda s: s.iloc[np.argmax(decimal_odds(s))]),
        n_books=("book", "nunique"))
    d = g.merge(ref[["game_id", "book", "ml_home_open", "ml_away_open", "ml_home_close", "ml_away_close",
                     "p_open", "p_close"]], on="game_id").merge(best, on="game_id")
    d["date"] = pd.to_datetime(d["date"])
    d["y"] = (d["home_pts"] > d["away_pts"]).astype(float)
    return d.sort_values(["date", "game_id"]).reset_index(drop=True), ln


def prenews_lineups(X: pd.DataFrame, pg: pd.DataFrame):
    """Lineup features using only each team's PREVIOUS game: who played last time is assumed to play tonight.
    This is what is knowable when the line opens, before the day's injury report."""
    st = F.PlayerState()
    by_tg = {k: v for k, v in pg.groupby(["game_id", "team"], sort=False)}
    last_played = {}
    cols = ["str_gmsc", "str_pm", "loss_gmsc", "delta_gmsc", "miss_min", "star_out"]
    out = np.full((len(X), 2, len(cols)), np.nan)
    for _, day in X.groupby("date", sort=True):
        pend = []
        for i, r in zip(day.index, day.itertuples(index=False)):
            for side, t in enumerate((r.home, r.away)):
                tr = by_tg.get((r.game_id, t))
                prev = last_played.get(t)
                if prev is not None:
                    f = st.lineup(t, {p: 1.0 for p in prev}, date=r.date)
                    if f is not None:
                        out[X.index.get_loc(i), side] = [f[c] for c in cols]
                if tr is not None:
                    pend.append((t, tr, r.date))
        for t, tr, d in pend:
            st.update_team_game(t, tr, None, d)
            last_played[t] = set(tr.loc[tr["played"].astype(bool), "player_id"])
    L = pd.DataFrame(index=X.index)
    for j, c in enumerate(cols):
        L[f"{c}_home"], L[f"{c}_away"] = out[:, 0, j], out[:, 1, j]
    L["d_str_gmsc"] = L["str_gmsc_home"] - L["str_gmsc_away"]
    L["d_str_pm"] = L["str_pm_home"] - L["str_pm_away"]
    L["d_loss_gmsc"] = L["loss_gmsc_home"] - L["loss_gmsc_away"]
    L["d_delta_gmsc"] = L["delta_gmsc_home"] - L["delta_gmsc_away"]
    L["d_miss_min"] = L["miss_min_home"] - L["miss_min_away"]
    L["has_lineup"] = L["d_str_gmsc"].notna().astype(float)
    return L


def stats_forecasts(X: pd.DataFrame, test_dates, refit_days=30):
    """Stats-only learners (never see a price), refit monthly on every earlier game since 2008."""
    prm = M.Params()
    fc = M.Forecasters(prm)
    out, last = [], None
    T_all = X[X["home_win"].notna()]
    for d, day in X[X["date"].isin(test_dates)].groupby("date", sort=True):
        if last is None or (d - last).days >= refit_days:
            T = T_all[(T_all["date"] < d)].copy()
            T["mkt_logit"] = T["mkt_logit"].fillna(0.0)
            T = T[T["date"] >= "2008-08-01"]
            fc.fit(T.assign(p_market=T["p_market"].fillna(0.5)), d)
            last = d
        P = fc.predict(day.assign(p_market=np.nan, mkt_logit=np.nan))
        out.append(pd.DataFrame({"game_id": day["game_id"].to_numpy(),
                                 "p_stats": 0.5 * (P["stats_lr"].to_numpy() + P["stats_gbm"].to_numpy())}))
    return pd.concat(out)


def blend_walk_forward(d: pd.DataFrame, warmup=400, refit_days=30):
    """logit p = a + b*logit(p_open) + c*(logit(p_stats) - logit(p_open)), fit on earlier games only."""
    lo, ls = F.logit(d["p_open"].to_numpy()), F.logit(d["p_stats"].to_numpy())
    p = np.full(len(d), np.nan)
    coef_hist, th, last = [], None, None
    for i, dt in enumerate(d["date"]):
        if i < warmup:
            continue
        if last is None or (dt - last).days >= refit_days:
            tr = d.index[:i][d["date"].iloc[:i] < dt]
            Xm = np.column_stack([np.ones(len(tr)), lo[tr], ls[tr] - lo[tr]])
            y = d["y"].to_numpy()[tr]
            th = np.zeros(3)
            th[1] = 1.0
            for _ in range(25):  # Newton steps with a light ridge on the disagreement term
                z = Xm @ th
                pp = F.sigmoid(z)
                W = pp * (1 - pp)
                g = Xm.T @ (pp - y) + np.array([0, 0, 2.0]) * th
                H = (Xm * W[:, None]).T @ Xm + np.diag([1e-6, 1e-6, 2.0])
                th = th - np.linalg.solve(H, g)
            last = dt
            coef_hist.append({"date": str(dt.date()), "a": th[0], "b_open": th[1], "c_model": th[2]})
        p[i] = F.sigmoid(th[0] + th[1] * lo[i] + th[2] * (ls[i] - lo[i]))
    return p, coef_hist


# --------------------------------------------------------------------------- study

def _boot(vals, nights, n=2000, seed=0, stat="mean"):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"v": vals, "n": nights}).groupby("n")["v"].agg(["sum", "count"])
    s, c = df["sum"].to_numpy(), df["count"].to_numpy()
    k = len(df)
    if k < 5:
        return [np.nan, np.nan]
    dr = rng.integers(0, k, size=(n, k))
    b = s[dr].sum(1) / c[dr].sum(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def _ols_cluster(y, x, groups):
    """Slope of y on x with standard errors clustered by game night (plain NumPy, no statsmodels)."""
    from math import erfc, sqrt
    X = np.column_stack([np.ones(len(x)), x])
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    e = y - X @ beta
    g = pd.factorize(groups)[0]
    S = np.zeros((len(np.unique(g)), 2))
    np.add.at(S, g, X * e[:, None])
    G, n = S.shape[0], len(y)
    V = bread @ (S.T @ S) @ bread * (G / (G - 1)) * ((n - 1) / (n - 2))
    z = beta[1] / sqrt(V[1, 1])
    return float(beta[1]), float(erfc(abs(z) / sqrt(2)))


def bets_at(d, p, mh, ma, edge):
    dh, da = decimal_odds(mh), decimal_odds(ma)
    evh, eva = p * dh - 1, (1 - p) * da - 1
    home = evh >= eva
    ev = np.where(home, evh, eva)
    take = (ev > edge) & ~np.isnan(p)
    dd = np.where(home, dh, da)
    won = np.where(home, d["y"] == 1, d["y"] == 0)
    pnl = np.where(won, dd - 1, -1.0)
    # CLV: fair closing probability of the side we bet minus the fair probability we paid for at open
    clv = np.where(home, d["p_close"] - d["p_open"], d["p_open"] - d["p_close"])
    return take, pnl, clv, home


def study():
    from . import live
    d, ln = load_lines()
    print(f"\nGames with opening + closing lines: {len(d):,} ({d['date'].min().date()} to {d['date'].max().date()})")
    print("Books per game by season:", d.groupby(d["date"].map(lambda x: F.teams.season_of(x)))["n_books"].median().to_dict())
    st = live._store()
    g, pg = live.assemble(st)
    num = ["home_pts", "away_pts", "ml_home", "ml_away", "spread", "total"]
    g[num] = g[num].apply(pd.to_numeric, errors="coerce").astype(float)
    pg["played"] = pg["played"].astype(bool)
    X = F.build(g, None)
    print("rebuilding pre-news lineups (about a minute)…", flush=True)
    L = prenews_lineups(X, pg)
    for c in L.columns:
        X[c] = L[c].to_numpy()
    X = X[X["game_id"].isin(set(d["game_id"])) | (X["date"] < d["date"].min())]
    test_dates = set(d["date"])
    print("refitting stats-only learners month by month…", flush=True)
    S = stats_forecasts(X, test_dates)
    d = d.merge(S, on="game_id", how="inner").sort_values(["date", "game_id"]).reset_index(drop=True)
    d["p_blend"], coefs = blend_walk_forward(d)
    T = d[d["p_blend"].notna()].copy()
    y, nights = T["y"].to_numpy(), T["date"].dt.strftime("%Y-%m-%d").to_numpy()
    br = lambda p: float(np.mean((p - y) ** 2))  # noqa: E731
    res = {"n_games": int(len(T)), "from": str(T["date"].min().date()), "to": str(T["date"].max().date())}
    res["brier"] = {k: br(T[c].to_numpy()) for k, c in
                    [("open", "p_open"), ("close", "p_close"), ("stats_only", "p_stats"), ("blend", "p_blend")]}
    res["brier_blend_minus_open"] = {"mean": res["brier"]["blend"] - res["brier"]["open"],
                                     "ci": _boot((T["p_blend"] - y) ** 2 - (T["p_open"] - y) ** 2, nights)}
    # Does the line move toward the model?
    mv = F.logit(T["p_close"].to_numpy()) - F.logit(T["p_open"].to_numpy())
    dis = F.logit(T["p_blend"].to_numpy()) - F.logit(T["p_open"].to_numpy())
    slope, pval = _ols_cluster(mv, dis, nights)
    big = np.abs(dis) > 0.10
    res["line_move"] = {"slope": slope, "p_value": pval,
                        "toward_model_share": float(np.mean(np.sign(mv[big]) == np.sign(dis[big]))) if big.any() else None,
                        "n_disagree": int(big.sum())}
    res["bets"] = {}
    for label, mh, ma in [("reference book, open", T["ml_home_open"], T["ml_away_open"]),
                          ("best book, open", T["best_h_open"], T["best_a_open"]),
                          ("reference book, close", T["ml_home_close"], T["ml_away_close"]),
                          ("best book, close", T["best_h_close"], T["best_a_close"])]:
        for edge in (0.01, 0.02, 0.03, 0.05):
            take, pnl, clv, _ = bets_at(T, T["p_blend"].to_numpy(), mh.to_numpy(), ma.to_numpy(), edge)
            if take.sum() == 0:
                continue
            at_open = label.endswith("open")
            res["bets"][f"{label} | EV>{int(edge*100)}%"] = {
                "n": int(take.sum()), "roi": float(pnl[take].mean()), "roi_ci": _boot(pnl[take], nights[take]),
                "avg_clv": float(np.nanmean(clv[take])) if at_open else None,
                "clv_ci": _boot(np.nan_to_num(clv[take]), nights[take]) if at_open else None,
                "clv_positive_share": float(np.mean(clv[take] > 0)) if at_open else None}
    res["blend_coefs_last"] = coefs[-1] if coefs else None
    by = []
    for s, q in T.groupby(T["date"].map(lambda x: F.teams.season_of(x))):
        yy = q["y"].to_numpy()
        by.append({"season": s, "n": int(len(q)), **{k: float(np.mean((q[c] - yy) ** 2)) for k, c in
                   [("open", "p_open"), ("close", "p_close"), ("blend", "p_blend"), ("stats_only", "p_stats")]}})
    res["by_season"] = by
    os.makedirs(os.path.join(ROOT, "web"), exist_ok=True)
    with open(os.path.join(ROOT, "web", "openline.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    _print(res)
    return res


def _print(r):
    b = r["brier"]
    print("\n================ OPENING-LINE STUDY ================")
    print(f"Test games: {r['n_games']:,}  ({r['from']} to {r['to']})   lower Brier = better")
    print(f"  Brier  opening line {b['open']:.5f} | closing line {b['close']:.5f} | "
          f"stats-only model {b['stats_only']:.5f} | model+open blend {b['blend']:.5f}")
    c = r["brier_blend_minus_open"]
    print(f"  Blend minus opening line: {c['mean']:+.5f}  (95% CI {c['ci'][0]:+.5f} to {c['ci'][1]:+.5f})"
          "   negative = model beats the open")
    m = r["line_move"]
    share = "n/a" if m["toward_model_share"] is None else f"{m['toward_model_share']:.0%}"
    print(f"\nDoes the line move toward the model? slope {m['slope']:+.3f} (p = {m['p_value']:.3f}); "
          f"when they disagree by >2.5 pts, line moves toward model {share} of the time (n = {m['n_disagree']:,})")
    print("\nPaper bets with the blend (CLV = closing fair prob minus fair prob at open, on our side)")
    print(f"  {'where / rule':38s} {'bets':>6s} {'return':>8s} {'95% CI':>18s} {'avg CLV':>8s} {'CLV>0':>6s}")
    for k, v in r["bets"].items():
        clv = "     n/a    n/a" if v["avg_clv"] is None else f" {v['avg_clv']:+8.2%} {v['clv_positive_share']:6.0%}"
        print(f"  {k:38s} {v['n']:6d} {v['roi']:+8.1%} {v['roi_ci'][0]:+8.1%} to {v['roi_ci'][1]:+6.1%}{clv}")
    print("\nBy season (Brier):")
    for s in r["by_season"]:
        print(f"  {s['season']}  n={s['n']:5d}  open {s['open']:.4f}  close {s['close']:.4f}  "
              f"blend {s['blend']:.4f}  stats {s['stats_only']:.4f}")
    print("\nSaved web/openline.json. Paste everything from 'OPENING-LINE STUDY' down back to Claude.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("fetch", "all"):
        fetch()
    if cmd in ("report", "all"):
        study()
