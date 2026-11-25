"""Star-rest predictor: the chance each rotation player is available tonight, using only what is known before the
injury report (i.e. when the line opens).

The October model and the first opening-line study assumed "whoever played last game plays tonight". That misses
the absences the market learns about between open and close: stars resting on the second night of a back-to-back,
veterans managed late in the season, players drifting back from injury. Those are exactly the moves the line-move
model tries to anticipate, so a calibrated probability per player should sharpen it.

Pipeline
  replay(X, pg)   walk the games in date order; for every team-game list the candidates (recent rotation players),
                  their pre-game features, ratings and (if the box score exists) whether they were available.
  walk_forward()  LightGBM refit each season on all earlier seasons -> p_avail for every candidate.
  expected()      team strength with each player weighted by p_avail, plus "rest risk" features.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from . import features as F
from . import teams

FEATS = ["played_last", "avail_last", "out_streak", "miss10", "days_off", "base_min", "q_gmsc", "star",
         "b2b", "three_in_four", "team_rest", "away", "travel_km", "b2b_rate", "exp_seasons", "gp_season",
         "season_frac", "team_wp", "late_wp", "opp_elo"]
MIN_BASE = 10.0  # rotation players only: deep-bench absences barely move the line


def _d10():
    return deque(maxlen=10)


class _Tracker:
    """Per-player availability history while on a team, plus per-team record."""

    def __init__(self):
        self.hist = defaultdict(_d10)        # (player, team) -> last 10 team games: 1 available, 0 not
        self.streak = defaultdict(int)       # (player, team) -> consecutive team games unavailable
        self.last_avail = {}                 # (player, team) -> available in the team's previous game
        self.last_played = {}                # (player, team) -> played in the team's previous game
        self.b2b_n = defaultdict(int)        # player -> second nights of back-to-backs after being available
        self.b2b_miss = defaultdict(int)
        self.first_season = {}
        self.gp = defaultdict(int)           # (player, season) -> games played
        self.wl = defaultdict(lambda: [0, 0])  # (team, season) -> wins, games


def _features(st, tk, p, t, r, side, season):
    k = (p, t)
    h = tk.hist[k]
    ld = st.last_date.get(p)
    qg, _ = st.q(p)
    base = st.base.get(p, 10.0)
    b2b = float(getattr(r, f"b2b_{side}") or 0)
    w, n = tk.wl[(t, season)]
    wp = (w + 2) / (n + 4)
    fs = tk.first_season.get(p, season)
    elo = getattr(r, "elo_diff", 0.0) or 0.0
    return [
        float(tk.last_played.get(k, 0)), float(tk.last_avail.get(k, 0)), np.log1p(tk.streak[k]),
        (1 - np.mean(h)) if h else 0.0,
        np.log1p((r.date - ld).days) if ld is not None else 3.0,
        base, qg, base * (qg - st.prior_gmsc), b2b,
        float(getattr(r, f"three_in_four_{side}") or 0), min(float(getattr(r, f"rest_{side}") or 3), 5.0),
        float(side == "away"), float(getattr(r, f"travel_km_{side}") or 0) / 1000,
        (tk.b2b_miss[p] + 0.5) / (tk.b2b_n[p] + 5.0),
        float(int(season[:4]) - int(fs[:4])), float(tk.gp[(p, season)]),
        float(r.season_frac), wp, float(r.season_frac > 0.85) * (wp - 0.5),
        float(elo if side == "home" else -elo) / 100,
    ]


def replay(X: pd.DataFrame, pg: pd.DataFrame):
    """One pass over every game. Returns
         rows:  one per (game, team, rotation candidate): features, outcome (NaN if no box score), ratings
         teams: one per (game, team): index range into rows + the 'previous game' heuristic lineup."""
    st = F.PlayerState()
    tk = _Tracker()
    by_tg = {k: v for k, v in pg.groupby(["game_id", "team"], sort=False)}
    R, meta, T = [], [], []
    for _, day in X.groupby("date", sort=True):
        pend = []
        for r in day.itertuples(index=False):
            season = teams.season_of(r.date)
            for side, t in (("home", r.home), ("away", r.away)):
                tr = by_tg.get((r.game_id, t))
                avail_now = None
                if tr is not None:
                    avail_now = set(tr.loc[tr["avail"].astype(float) > 0, "player_id"])
                start = len(R)
                for p in st.candidates(t, r.date):
                    base = st.base.get(p, 10.0)
                    qg, qpm = st.q(p)
                    R.append(_features(st, tk, p, t, r, side, season))
                    y = np.nan if avail_now is None else float(p in avail_now)
                    meta.append((r.game_id, t, p, r.date, base, qg, qpm, y))
                T.append((r.game_id, t, side, start, len(R)))
                if tr is not None:
                    pend.append((t, tr, r, side, season))
        for t, tr, r, side, season in pend:
            avail = set(tr.loc[tr["avail"].astype(float) > 0, "player_id"])
            played = set(tr.loc[tr["played"].astype(bool), "player_id"])
            b2b = bool(getattr(r, f"b2b_{side}") or 0)
            for p in st.candidates(t, r.date) + [p for p in played if st.last_team.get(p) != t]:
                k = (p, t)
                a = p in avail
                if b2b and tk.last_avail.get(k):
                    tk.b2b_n[p] += 1
                    tk.b2b_miss[p] += int(not a)
                tk.hist[k].append(int(a))
                tk.streak[k] = 0 if a else tk.streak[k] + 1
                tk.last_avail[k] = int(a)
                tk.last_played[k] = int(p in played)
            for p in played:
                tk.first_season.setdefault(p, season)
                tk.gp[(p, season)] += 1
            won = (r.home_pts > r.away_pts) if side == "home" else (r.away_pts > r.home_pts)
            if pd.notna(r.home_pts):
                tk.wl[(t, season)][0] += int(won)
                tk.wl[(t, season)][1] += 1
            st.update_team_game(t, tr, None, r.date)
    rows = pd.DataFrame(R, columns=FEATS)
    m = pd.DataFrame(meta, columns=["game_id", "team", "player_id", "date", "base", "q_gmsc_r", "q_pm", "y"])
    rows = pd.concat([m, rows], axis=1)
    rows["season"] = rows["date"].map(teams.season_of)
    tg = pd.DataFrame(T, columns=["game_id", "team", "side", "start", "end"])
    return rows, tg, st


def _lgb():
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=31, min_child_samples=200,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)


def walk_forward(rows: pd.DataFrame, first_test_season="2012-13"):
    """p_avail for every row: LightGBM fit on all earlier seasons' rotation rows (refit once per season).
    Rows before `first_test_season` get the simple heuristic (played last game)."""
    p = rows["played_last"].astype(float).to_numpy() * 0.93 + 0.03
    p = np.asarray(p, float)
    rot = rows["base"].to_numpy() >= MIN_BASE
    seasons = sorted(rows["season"].unique())
    for s in seasons:
        if s < first_test_season:
            continue
        tr = (rows["season"] < s).to_numpy() & rows["y"].notna().to_numpy() & rot
        te = (rows["season"] == s).to_numpy()
        if tr.sum() < 5000:
            continue
        m = _lgb().fit(rows.loc[tr, FEATS], rows.loc[tr, "y"].astype(int), sample_weight=_recency(rows.loc[tr, "season"], s))
        p[te] = m.predict_proba(rows.loc[te, FEATS])[:, 1]
    return p


def _recency(season: pd.Series, current: str, halflife=3.0):
    """Load management has grown since the mid-2010s: weigh recent seasons more (half weight every 3 seasons)."""
    age = int(current[:4]) - season.str[:4].astype(int)
    return (0.5 ** ((age - 1).clip(lower=0) / halflife)).to_numpy()


def fit_final(rows: pd.DataFrame):
    ok = rows["y"].notna().to_numpy() & (rows["base"].to_numpy() >= MIN_BASE)
    nxt = teams.season_of(rows["date"].max() + pd.Timedelta(days=120))
    return _lgb().fit(rows.loc[ok, FEATS], rows.loc[ok, "y"].astype(int), sample_weight=_recency(rows.loc[ok, "season"], nxt))


def expected(rows: pd.DataFrame, tg: pd.DataFrame, p_avail):
    """Per team-game: strength with players weighted by p_avail, and how much strength is at risk.
    Also the old heuristic (previous game's players) for comparison."""
    base = rows["base"].to_numpy(float)
    qs = rows[["q_gmsc_r", "q_pm"]].to_numpy(float)
    prior = F.PlayerState().prior_gmsc
    star = (base * (qs[:, 0] - prior)).clip(min=0)
    last = rows["played_last"].to_numpy(float)
    out = np.full((len(tg), 6), np.nan)
    for i, (a, b) in enumerate(zip(tg["start"].to_numpy(), tg["end"].to_numpy())):
        if b <= a:
            continue
        pr = p_avail[a:b]
        full = F._strength(base[a:b], qs[a:b], np.ones(b - a))
        exp = F._strength(base[a:b], qs[a:b], pr)
        heur = F._strength(base[a:b], qs[a:b], last[a:b]) if last[a:b].sum() > 0 else full
        risk = ((1 - pr) * star[a:b] * last[a:b]).sum()      # stars who played last game but may sit tonight
        out[i] = [exp[0], exp[1], full[0] - exp[0], risk, heur[0], (1 - pr[last[a:b] > 0]).sum() if (last[a:b] > 0).any() else 0]
    E = tg[["game_id", "team", "side"]].copy()
    E[["exp_gmsc", "exp_pm", "exp_loss", "rest_risk", "heur_gmsc", "exp_absent_n"]] = out
    return E


def game_features(E: pd.DataFrame) -> pd.DataFrame:
    """Home-minus-away versions, one row per game."""
    h = E[E["side"] == "home"].set_index("game_id")
    a = E[E["side"] == "away"].set_index("game_id")
    G = pd.DataFrame(index=h.index.union(a.index))
    for c in ["exp_gmsc", "exp_pm", "exp_loss", "rest_risk", "heur_gmsc"]:
        G[f"{c}_home"], G[f"{c}_away"] = h[c], a[c]
    G["d_exp_gmsc"] = G["exp_gmsc_home"] - G["exp_gmsc_away"]
    G["d_exp_pm"] = G["exp_pm_home"] - G["exp_pm_away"]
    G["d_rest_risk"] = G["rest_risk_home"] - G["rest_risk_away"]
    G["d_exp_vs_heur"] = (G["exp_gmsc_home"] - G["heur_gmsc_home"]) - (G["exp_gmsc_away"] - G["heur_gmsc_away"])
    return G.reset_index().rename(columns={"index": "game_id"})
