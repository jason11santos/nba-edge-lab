"""The live loop: collect -> predict -> settle -> learn.

    python -m nba_edge live --once        one pass: refresh scores, lines and injuries, predict today, write web/live.json
    python -m nba_edge live               keep running: a pass every 10 minutes, retrain once a night
    python -m nba_edge retrain            refit every model on all games played so far

What happens in a pass
  1. ESPN scoreboard for yesterday, today and tomorrow: scores, game state, DraftKings open and current lines.
     Every line seen is stored, so the database builds its own line-movement history.
  2. The Odds API (optional key): every book's price, so bets use the best available number.
  3. ESPN injury report: each listed player's chance of playing tonight.
  4. Features for today's games, with tonight's lineup weighted by those chances, then every model's forecast.
  5. Paper bets: the first time a game clears the betting rule the price is locked in. When the game ends
     the bet is settled and scored against the closing line (closing-line value, the best early sign of a
     real edge). The adaptive ensemble's weights are updated with each finished game.
"""
from __future__ import annotations

import json
import os
import pickle
import time
import traceback
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import features as F
from . import models as M
from . import teams
from .backtest import DESIGN
from .evaluate import decimal_odds, kelly_fraction, no_vig
from .sources import espn, oddsapi
from .store import Store, now

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
WEB = os.path.join(ROOT, "web")
MODEL_PATH = os.path.join(DATA, "live_model.pkl")
HISTORY_END = "2026-01-07"   # last date in the downloaded odds history; ESPN covers everything after


def _store():
    os.makedirs(DATA, exist_ok=True)
    return Store(os.path.join(DATA, "live.sqlite"))


def log(*a):
    print(datetime.now().strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------- collect

def collect_day(st: Store, day: date, closing=False):
    games = espn.scoreboard(day)
    st.upsert_games([{**{k: g[k] for k in ("game_id", "espn_id", "date", "season_type", "home", "away", "start_utc",
                                            "state", "status", "home_pts", "away_pts")}, "neutral": int(g["neutral"])}
                     for g in games])
    for g in games:
        if g["odds"] and g["state"] == "pre":
            st.add_odds(g["game_id"], "espn", "live", g["odds"])
        elif g["state"] == "post" and closing:
            have = st.q("select 1 from odds where game_id=? and kind='close'", (g["game_id"],))
            if have.empty:
                try:
                    o = espn.closing_odds(g["espn_id"])
                    if o:
                        st.add_odds(g["game_id"], "espn", "close", o)
                except Exception as e:  # noqa: BLE001
                    log("closing odds failed", g["game_id"], e)
    return games


def backfill(st: Store, start: str = "2026-01-08", end: str | None = None):
    """Fill every date from `start` to yesterday: results and closing lines. Safe to re-run."""
    d = pd.Timestamp(start).date()
    end_d = pd.Timestamp(end).date() if end else date.today() - timedelta(days=1)
    done = set(st.q("select distinct date from games where state='post'")["date"])
    while d <= end_d:
        if d.isoformat() not in done or d >= end_d - timedelta(days=2):
            collect_day(st, d, closing=True)
            log("backfilled", d)
        d += timedelta(days=1)


def sync_players(st: Store, seasons=("2024-25", "2025-26", "2026-27")):
    """Player lines for recent seasons. nba_api first (one call per season); ESPN box scores if that fails."""
    try:
        if os.environ.get("NBA_EDGE_SKIP_NBAAPI"):
            raise RuntimeError("skipped (stats.nba.com blocks cloud servers)")
        from .sources import nbastats
        for s in seasons:
            for stype in ("Regular Season", "Playoffs"):
                df = nbastats.season_logs(s, stype)
                if len(df):
                    st.add_player_games(df, "nbastats")
                    log("player logs", s, stype, len(df))
        return "nbastats"
    except Exception as e:  # noqa: BLE001
        log("nba_api unavailable, using ESPN box scores:", e)
    have = set(st.q("select distinct game_id from player_games")["game_id"])
    todo = st.q("select game_id, espn_id, date from games where state='post' and season_type in (2,3,5)")
    todo = todo[~todo["game_id"].isin(have)]
    for r in todo.itertuples(index=False):
        try:
            rows = pd.DataFrame(espn.box_score(r.espn_id))
        except Exception as e:  # noqa: BLE001
            log("box score failed", r.game_id, e)
            continue
        if rows.empty:
            continue
        rows["player_id"] = "espn:" + rows["espn_id"].astype(str)
        rows["game_id"], rows["date"] = r.game_id, r.date
        st.add_player_games(rows, "espn")
    return "espn"


# --------------------------------------------------------------------------- assemble

def load_history():
    g = pd.read_parquet(os.path.join(DATA, "games.parquet"))
    pg = pd.read_parquet(os.path.join(DATA, "player_games.parquet"))
    return g[g["date"] <= HISTORY_END], pg


def assemble(st: Store):
    """History + everything the store has collected, as one games table and one player table."""
    g_hist, pg_hist = load_history()
    sg = st.q("select * from games where season_type in (2,3,5)")
    sg["date"] = pd.to_datetime(sg["date"])
    sg = sg[sg["date"] > pd.Timestamp(HISTORY_END)]
    od = st.latest_odds()
    # closing line beats the last live snapshot, ESPN (one book) is the reference price
    od["rank"] = od["kind"].map({"close": 0, "live": 1}).fillna(2)
    od = od[od["source"] == "espn"].sort_values(["game_id", "rank"]).drop_duplicates("game_id")
    sg = sg.merge(od[["game_id", "ml_home", "ml_away", "spread", "total"]], on="game_id", how="left")
    sg.loc[sg["state"] != "post", ["home_pts", "away_pts"]] = np.nan
    g = pd.concat([g_hist, sg[g_hist.columns.intersection(sg.columns)]], ignore_index=True)
    num = ["home_pts", "away_pts", "ml_home", "ml_away", "spread", "total"]
    g[num] = g[num].apply(pd.to_numeric, errors="coerce").astype(float)
    g["season"] = g["date"].map(teams.season_of)
    g = g.sort_values(["date", "home"]).drop_duplicates("game_id", keep="last").reset_index(drop=True)
    pl = st.q("select * from player_games")
    pl["date"] = pd.to_datetime(pl["date"])
    pl["played"] = pl["played"].astype(bool)
    pl = pl[pl["date"] > pd.Timestamp("2024-07-01")]
    # map ESPN ids onto NBA ids by name where the same player exists in history
    names = pg_hist.drop_duplicates("player_id", keep="last").set_index("player")["player_id"]
    norm = lambda s: s.str.lower().str.replace(r"[^a-z ]", "", regex=True).str.replace(r"\s+(jr|sr|ii|iii|iv)$", "", regex=True).str.strip()  # noqa: E731
    nmap = dict(zip(norm(pd.Series(names.index)), names.values))
    esp = pl["player_id"].str.startswith("espn:")
    pl.loc[esp, "player_id"] = [str(nmap.get(n, p)) for n, p in zip(norm(pl.loc[esp, "player"]), pl.loc[esp, "player_id"])]
    pg = pd.concat([pg_hist.assign(source="history"), pl], ignore_index=True)
    pg["player_id"] = pg["player_id"].astype(str)
    return g, pg


def build_features(g, pg):
    """Features for every game. Lineup features are kept only where the source records who was available
    (historical box scores and ESPN box scores); nba_api rows list only who played, which would leak."""
    X = F.build(g, None)
    lf, state = F.lineup_features(X, pg)
    leaky = set(pg.loc[pg["source"] == "nbastats", "game_id"])
    lf.loc[X["game_id"].isin(leaky).to_numpy()] = np.nan
    for c in lf.columns:
        X[c] = lf[c].to_numpy()
    X["has_lineup"] = X["d_str_gmsc"].notna().astype(float)
    return X, state


# --------------------------------------------------------------------------- learn

def retrain(st: Store, prm: M.Params | None = None):
    g, pg = assemble(st)
    X, state = build_features(g, pg)
    T = X[X["home_win"].notna() & X["p_market"].notna()]
    fc = M.Forecasters(prm or M.Params()).fit(T, pd.Timestamp(date.today()))
    old = _load_model()
    hedge = old["hedge"] if old else M.Hedge(M.Params().use, M.Params().hedge_eta, M.Params().hedge_decay)
    if old is None:
        hedge = _warm_hedge(X, hedge)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"fc": fc, "hedge": hedge, "state": state, "trained": now(), "n_train": len(T)}, f)
    st.meta("last_retrain", now())
    log("retrained on", len(T), "games; weights", hedge.state())


def _warm_hedge(X, hedge):
    """Start the ensemble weights from the backtest, if its predictions are on disk."""
    p = os.path.join(DATA, "predictions.parquet")
    if os.path.exists(p):
        P = pd.read_parquet(p)
        P = P[P["date"] >= P["date"].max() - pd.Timedelta(days=400)]
        for _, day in P.groupby("date"):
            hedge.update(day.rename(columns={f"p_{k}": k for k in M.NAMES}), day["home_win"].to_numpy())
    else:  # weights the backtest ended with, shipped with the code
        init = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hedge_init.json")
        if os.path.exists(init):
            js = json.load(open(init))
            if js["names"] == hedge.names:
                hedge.loss = np.array(js["loss"])
    return hedge


def _load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- predict

def _norm(s):
    import re
    s = re.sub(r"[^a-z ]", "", (s or "").lower())
    return re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", s).strip()


def tonight_lineups(state: F.PlayerState, team: str, inj: pd.DataFrame, day=None):
    """Tonight's lineup features for one team, plus the listed players who matter most."""
    day = pd.Timestamp(day) if day is not None else None
    cands = state.candidates(team, day)
    by_name = {_norm(state.name.get(p, "")): p for p in cands}
    avail, listed = {p: 1.0 for p in cands}, []
    for r in inj[inj["team"] == team].itertuples(index=False):
        p = by_name.get(_norm(r.player))
        if p is None:
            continue  # not in the recent rotation
        avail[p] = r.play_prob
        q, _ = state.q(p)
        impact = state.base.get(p, 0) * (q - state.prior_gmsc)
        listed.append({"player": r.player, "status": r.status or r.fantasy, "play_prob": r.play_prob,
                       "minutes": round(state.base.get(p, 0), 1), "impact": round(impact, 2), "note": r.note})
    f = state.lineup(team, avail, date=day)
    listed.sort(key=lambda x: -x["impact"] * (1 - x["play_prob"]))
    return f, listed


def best_prices(st: Store, game_id):
    """Best moneyline on each side across every source and book seen in the last 2 hours."""
    cut = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    o = st.q("select * from odds where game_id=? and ts>=? and kind='live'", (game_id, cut))
    if o.empty:
        return None
    o = o.sort_values("ts").drop_duplicates("source", keep="last").dropna(subset=["ml_home", "ml_away"])
    if o.empty:
        return None
    dec_h, dec_a = decimal_odds(o["ml_home"]), decimal_odds(o["ml_away"])
    ih, ia = int(np.nanargmax(dec_h)), int(np.nanargmax(dec_a))
    return {"home": {"ml": float(o["ml_home"].iloc[ih]), "book": o["source"].iloc[ih]},
            "away": {"ml": float(o["ml_away"].iloc[ia]), "book": o["source"].iloc[ia]},
            "books": o[["source", "ml_home", "ml_away"]].to_dict("records")}


# Paper-betting rules. Each is tracked separately, one bet per game per rule, price locked when it first fires.
# "early" rules use the opening-line study: the model blended with the OPENING price, and they may only fire
# while the line is still close to where it opened (the study showed the edge disappears once the market moves).
RULES = {
    "open3": {"label": "Early line, EV > 3% (main)", "prob": "early", "edge": 0.03},
    "open5": {"label": "Early line, EV > 5%", "prob": "early", "edge": 0.05},
    "ens2": {"label": "Ensemble at current line, EV > 2% (old rule)", "prob": "ensemble", "edge": 0.02},
}
MAX_LINE_MOVE = 0.015   # early rules stop once the fair price has moved 1.5 points from the open
MAX_ADVERSE_MOVE = 0.005  # ...or 0.5 points AGAINST our side: the market disagreeing with us is information


def early_coefs():
    """Blend weights fitted by the opening-line study (py -m nba_edge.openline), if it has been run."""
    p = os.path.join(WEB, "openline.json")
    if not os.path.exists(p):
        return None
    try:
        c = json.load(open(p)).get("blend_coefs_last")
        return (float(c["a"]), float(c["b_open"]), float(c["c_model"])) if c else None
    except Exception:  # noqa: BLE001
        return None


def _fair_home(mh, ma):
    if mh is None or ma is None or np.isnan(mh) or np.isnan(ma):
        return None
    ph, over = no_vig(np.array([mh]), np.array([ma]))
    return float(ph[0]) if 0.99 < over[0] < 1.15 else None


def predict_day(st: Store, day: date, mdl, g):
    fc, hedge, state = mdl["fc"], mdl["hedge"], mdl["state"]
    if g[g["date"] == pd.Timestamp(day)].empty:
        return []
    # Elo, form, rest and market features for every game up to this day (lineups added below from the injury report)
    X = F.build(g[g["date"] <= pd.Timestamp(day)], None)
    Xt = X[X["date"] == pd.Timestamp(day)].copy()
    inj = st.q("select * from injuries where ts=(select max(ts) from injuries)")
    for k in ["str_gmsc", "str_pm", "loss_gmsc", "loss_pm", "delta_gmsc", "miss_min", "star_out"]:
        for sd in ("home", "away"):
            if f"{k}_{sd}" not in Xt:
                Xt[f"{k}_{sd}"] = np.nan
    sides = {}
    for i, r in Xt.iterrows():
        fh, lh = tonight_lineups(state, r["home"], inj, day)
        fa, la = tonight_lineups(state, r["away"], inj, day)
        sides[r["game_id"]] = (lh, la)
        if fh and fa:
            for k in ["str_gmsc", "str_pm", "loss_gmsc", "delta_gmsc", "miss_min", "star_out"]:
                Xt.loc[i, f"{k}_home"], Xt.loc[i, f"{k}_away"] = fh[k], fa[k]
            Xt.loc[i, "d_str_gmsc"] = fh["str_gmsc"] - fa["str_gmsc"]
            Xt.loc[i, "d_str_pm"] = fh["str_pm"] - fa["str_pm"]
            Xt.loc[i, "d_loss_gmsc"] = fh["loss_gmsc"] - fa["loss_gmsc"]
            Xt.loc[i, "d_delta_gmsc"] = fh["delta_gmsc"] - fa["delta_gmsc"]
            Xt.loc[i, "d_miss_min"] = fh["miss_min"] - fa["miss_min"]
            Xt.loc[i, "has_lineup"] = 1.0
    P = fc.predict(Xt)
    P["ensemble"] = hedge.combine(P)
    coefs = early_coefs()
    games, ts = [], now()
    sg = st.q("select * from games where date=?", (day.isoformat(),)).set_index("game_id")
    lo = st.q("select * from odds where game_id in (%s) and source='espn' order by ts" %
              ",".join("?" * len(Xt)), tuple(Xt["game_id"]))
    for (i, r), (_, p) in zip(Xt.iterrows(), P.iterrows()):
        gid = r["game_id"]
        meta = sg.loc[gid] if gid in sg.index else None
        hist = lo[lo["game_id"] == gid]
        last = hist.iloc[-1] if len(hist) else None
        bp = best_prices(st, gid)
        # early-line blend: model vs the OPENING price
        p_open = None if last is None else _fair_home(last["ml_home_open"], last["ml_away_open"])
        p_now = None if last is None else _fair_home(last["ml_home"], last["ml_away"])
        p_stats = 0.5 * (float(p["stats_lr"]) + float(p["stats_gbm"]))
        p_early, moved = None, None
        if coefs and p_open is not None:
            a, b, c = coefs
            lo_ = float(F.logit(p_open))
            p_early = float(F.sigmoid(a + b * lo_ + c * (float(F.logit(p_stats)) - lo_)))
            moved = None if p_now is None else abs(p_now - p_open)
        picks = []
        if bp and meta is not None and meta["state"] == "pre":
            for rule, R in RULES.items():
                prob = float(p["ensemble"]) if R["prob"] == "ensemble" else p_early
                if prob is None:
                    continue
                evh = prob * decimal_odds(bp["home"]["ml"]) - 1
                eva = (1 - prob) * decimal_odds(bp["away"]["ml"]) - 1
                side, ev = ("home", float(evh)) if evh >= eva else ("away", float(eva))
                px = bp[side]
                pp = prob if side == "home" else 1 - prob
                stake = float(min(DESIGN["kelly_mult"] * kelly_fraction(pp, decimal_odds(px["ml"])), DESIGN["max_stake"]))
                why = None
                if R["prob"] == "early":
                    # The study tested betting AT THE OPEN. So the edge must exist at the opening price on its own,
                    # not appear because the price drifted against the model after the open.
                    ml_o = None if last is None else last["ml_home_open" if side == "home" else "ml_away_open"]
                    adverse = None if (p_now is None or p_open is None) else ((p_open - p_now) if side == "home" else (p_now - p_open))
                    ev_open = None if ml_o is None or np.isnan(ml_o) else pp * float(decimal_odds(ml_o)) - 1
                    if moved is None or adverse is None or ev_open is None:
                        why = "no opening line"
                    elif adverse > MAX_ADVERSE_MOVE:
                        why = f"line moved {adverse * 100:.1f} pts against this side since the open"
                    elif moved > MAX_LINE_MOVE:
                        why = "line already moved from the open"
                    elif ev_open <= R["edge"]:
                        why = f"EV at the opening price {ev_open:+.1%}, under {R['edge']:.0%}"
                if why is None and ev <= R["edge"]:
                    why = f"EV under {R['edge']:.0%}"
                bet = why is None
                picks.append({"rule": rule, "side": side, "team": r[side], "ml": px["ml"], "book": px["book"],
                              "ev": ev, "stake": stake, "bet": bet, "why": why})
                if bet and p_now is not None:
                    fair = p_now if side == "home" else 1 - p_now
                    st.con.execute("insert or ignore into paper_bets (rule, game_id, ts, side, price, book, p_model,"
                                   " p_fair_at_bet, ev, stake) values (?,?,?,?,?,?,?,?,?,?)",
                                   (rule, gid, ts, side, px["ml"], px["book"], pp, fair, ev, stake))
            ens = next(x for x in picks if x["rule"] == "ens2")
            st.con.execute("insert into predictions values (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (ts, gid, p["market"], p["ensemble"], p["mkt_gbm"], p["mkt_lr"], p["stats_lr"], p["stats_gbm"],
                            ens["side"], ens["ml"], ens["book"], ens["ev"], ens["stake"]))
        locked = st.q("select rule, side, price, book, ev, stake, ts from paper_bets where game_id=?", (gid,))
        games.append({
            "game_id": gid, "date": day.isoformat(), "home": r["home"], "away": r["away"],
            "start_utc": None if meta is None else meta["start_utc"], "state": None if meta is None else meta["state"],
            "status": None if meta is None else meta["status"],
            "score": None if meta is None or meta["state"] == "pre" else [meta["home_pts"], meta["away_pts"]],
            "line": None if last is None else {k: _f(last[k]) for k in ["ml_home", "ml_away", "ml_home_open", "ml_away_open",
                                                                         "spread", "spread_open", "total", "total_open"]},
            "line_history": [{"ts": t, "ml_home": _f(a_), "ml_away": _f(b_)}
                             for t, a_, b_ in hist[["ts", "ml_home", "ml_away"]].itertuples(index=False)][-40:],
            "books": bp["books"] if bp else [],
            "p": {**{k: _f(p[k]) for k in M.NAMES + ["ensemble"]}, "early": _f(p_early), "stats": _f(p_stats)},
            "p_open": _f(p_open), "line_move": _f(moved),
            "elo": [_f(r["elo_home"]), _f(r["elo_away"])], "rest": [_f(r["rest_home"]), _f(r["rest_away"])],
            "b2b": [_f(r["b2b_home"]), _f(r["b2b_away"])], "travel_km_away": _f(r["travel_km_away"]),
            "lineup_loss": [_f(Xt.loc[i, "loss_gmsc_home"]), _f(Xt.loc[i, "loss_gmsc_away"])],
            "injuries": {"home": sides[gid][0][:6], "away": sides[gid][1][:6]},
            "picks": picks, "locked": locked.replace({np.nan: None}).to_dict("records"),
            "pick": next((x for x in picks if x["rule"] == "ens2"), None),
        })
    st.con.commit()
    return games


def predict_today(st: Store, day: date | None = None, days_ahead: int = 1):
    """Forecasts for today and tomorrow. Tomorrow matters: lines usually open the day before, and the early
    rules need to see a line while it is still at its opening price."""
    mdl = _load_model()
    if mdl is None:
        retrain(st)
        mdl = _load_model()
    day = day or datetime.now(espn.ET).date()
    g, _ = assemble(st)
    out = []
    for k in range(days_ahead + 1):
        out += predict_day(st, day + timedelta(days=k), mdl, g)
    return out


def _f(x):
    try:
        x = float(x)
        return None if np.isnan(x) else round(x, 4)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- settle

def settle(st: Store):
    """Score finished games: update ensemble weights from the last pre-game forecast, settle every rule's paper bets
    and measure closing-line value (fair closing probability on our side minus fair probability when we bet)."""
    mdl = _load_model()
    if mdl is None:
        return
    done = st.q("""select g.game_id, g.home_pts, g.away_pts from games g
                   where g.state='post' and g.game_id not in (select value from meta where key like 'settled:%')""")
    open_bets = st.q("""select b.*, g.home_pts, g.away_pts from paper_bets b join games g using (game_id)
                        where g.state='post' and b.won is null""")
    if done.empty and open_bets.empty:
        return
    od = st.latest_odds()
    od = od[od["source"] == "espn"]
    od["rank"] = od["kind"].map({"close": 0, "live": 1}).fillna(2)
    closes = od.sort_values(["game_id", "rank"]).drop_duplicates("game_id").set_index("game_id")
    for b in open_bets.itertuples(index=False):
        y = 1.0 if b.home_pts > b.away_pts else 0.0
        won = int((b.side == "home") == (y == 1))
        pnl = (float(decimal_odds(b.price)) - 1) * b.stake if won else -b.stake
        fc_, clv = None, None
        if b.game_id in closes.index:
            c = closes.loc[b.game_id]
            ph = _fair_home(c["ml_home"], c["ml_away"])
            if ph is not None:
                fc_ = ph if b.side == "home" else 1 - ph
                clv = fc_ - b.p_fair_at_bet
        st.con.execute("update paper_bets set won=?, pnl=?, p_fair_close=?, clv=? where rule=? and game_id=?",
                       (won, float(pnl), fc_, clv, b.rule, b.game_id))
    preds = st.q("select * from predictions").sort_values("ts").drop_duplicates("game_id", keep="last")
    upd = []
    for r in done.itertuples(index=False):
        y = 1.0 if r.home_pts > r.away_pts else 0.0
        pr = preds[preds["game_id"] == r.game_id]
        if len(pr):
            upd.append((pr.iloc[0], y))
        st.meta("settled:" + r.game_id, r.game_id)
    if upd:
        P = pd.DataFrame([{**{k: u[0][f"p_{k}"] for k in M.NAMES}} for u in upd])
        mdl["hedge"].update(P, np.array([u[1] for u in upd]))
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(mdl, f)
        log("settled", len(upd), "games; weights", mdl["hedge"].state())
    if len(open_bets):
        log("settled", len(open_bets), "paper bets")
    st.con.commit()


# --------------------------------------------------------------------------- one pass

def run_once(st: Store | None = None, write=True):
    st = st or _store()
    status = {}
    today = datetime.now(espn.ET).date()
    try:
        for d in (today - timedelta(days=1), today, today + timedelta(days=1)):
            collect_day(st, d, closing=(d < today))
        status["espn"] = "ok"
    except Exception as e:  # noqa: BLE001
        status["espn"] = f"error: {e}"
    if oddsapi.enabled():
        try:
            rows = oddsapi.moneylines()
            for r in rows:
                st.add_odds(r["game_id"], r["book"], "live", r)
            status["oddsapi"] = f"ok ({len(rows)} prices)"
        except Exception as e:  # noqa: BLE001
            status["oddsapi"] = f"error: {e}"
    else:
        status["oddsapi"] = "off (set ODDS_API_KEY for every book's price)"
    try:
        st.add_injuries(espn.injuries())
        status["injuries"] = "ok"
    except Exception as e:  # noqa: BLE001
        status["injuries"] = f"error: {e}"
    settle(st)
    status["early rules"] = "on" if early_coefs() else "off (run: py -m nba_edge.openline report)"
    games = []
    try:
        games = predict_today(st, today)
        status["model"] = "ok"
    except Exception as e:  # noqa: BLE001
        status["model"] = f"error: {e}"
        traceback.print_exc()
    if write:
        write_live_json(st, games, status, today)
    return games


def paper_summary(st: Store):
    b = st.q("select * from paper_bets")
    out = {}
    for rule, R in RULES.items():
        x = b[b["rule"] == rule]
        s = x[x["won"].notna()]
        c = s["clv"].dropna()
        out[rule] = {"label": R["label"], "open": int(x["won"].isna().sum()), "settled": int(len(s)),
                     "staked": float(s["stake"].sum()) if len(s) else 0.0, "pnl": float(s["pnl"].sum()) if len(s) else 0.0,
                     "roi": float(s["pnl"].sum() / s["stake"].sum()) if len(s) and s["stake"].sum() > 0 else None,
                     "flat_roi": float(np.mean(np.where(s["won"] == 1, decimal_odds(s["price"]) - 1, -1.0))) if len(s) else None,
                     "avg_clv": float(c.mean()) if len(c) else None,
                     "clv_se": float(c.std(ddof=1) / np.sqrt(len(c))) if len(c) > 1 else None,
                     "beat_close": float((c > 0).mean()) if len(c) else None}
    recent = b[b["won"].notna()].sort_values("ts").tail(30)
    return {"rules": out, "recent": recent.replace({np.nan: None}).to_dict("records")}


def write_live_json(st, games, status, day):
    mdl = _load_model()
    out = {"generated": now(), "day": day.isoformat(), "status": status, "games": games,
           "weights": mdl["hedge"].state() if mdl else None, "trained": mdl["trained"] if mdl else None,
           "paper": paper_summary(st), "rules": {k: {"label": v["label"], "edge": v["edge"]} for k, v in RULES.items()},
           "max_line_move": MAX_LINE_MOVE,
           "rule": {k: DESIGN[k] for k in ("main_edge", "kelly_mult", "max_stake")}}
    os.makedirs(WEB, exist_ok=True)
    tmp = os.path.join(WEB, "live.json.tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, default=str, allow_nan=False)
    os.replace(tmp, os.path.join(WEB, "live.json"))
    log("wrote web/live.json:", len(games), "games")


def maybe_retrain(st: Store, hour=4):
    """Refit once a day, after `hour` Pacific time, with the newest player data."""
    from zoneinfo import ZoneInfo
    now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
    last = st.meta("last_retrain_day") or ""
    if _load_model() is None or (now_pt.hour >= hour and last != now_pt.date().isoformat()):
        sync_players(st, seasons=(teams.season_of(now_pt),))
        retrain(st)
        st.meta("last_retrain_day", now_pt.date().isoformat())


def cloud_pass():
    """One scheduled run on a cloud server (GitHub Actions): bootstrap if needed, retrain daily,
    rerun the opening-line study weekly, then a normal live pass."""
    st = _store()
    if not st.meta("backfilled"):
        log("first run: backfilling results and closing lines since", HISTORY_END)
        backfill(st)
        sync_players(st)
        st.meta("backfilled", now())
    else:
        backfill(st, start=(date.today() - timedelta(days=3)).isoformat())
    maybe_retrain(st)
    wk = datetime.now().strftime("%G-W%V")
    if early_coefs() is None or (st.meta("study_week") != wk and datetime.now().weekday() == 0):
        from . import openline
        try:
            openline.fetch()
            openline.study()
            st.meta("study_week", wk)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
    run_once(st)
    b = st.q("select * from paper_bets")
    b.to_csv(os.path.join(WEB, "paper_bets.csv"), index=False)


def loop(minutes=10, retrain_hour=5):
    st = _store()
    if not st.meta("backfilled"):
        log("first run: backfilling results and closing lines since", HISTORY_END)
        backfill(st)
        sync_players(st)
        st.meta("backfilled", now())
        retrain(st)
    while True:
        try:
            last = st.meta("last_retrain") or ""
            if datetime.now().hour >= retrain_hour and last[:10] != datetime.now(timezone.utc).date().isoformat():
                sync_players(st, seasons=(teams.season_of(datetime.now()),))
                retrain(st)
            run_once(st)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        time.sleep(minutes * 60)
