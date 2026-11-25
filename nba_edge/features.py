"""Feature engine. Every feature for a game uses only information available before tip-off.

The same code runs on history (backtest) and on today's slate (live). The one difference is
player availability: in history we know who actually played; live, we use the injury report
and turn each status into a probability of playing.

Feature groups
  market     moneyline with the vig removed, point spread, game total
  ratings    margin-of-victory Elo, recent form vs Elo, league-wide home-court edge (drifts over time)
  schedule   rest days, back-to-backs, 3 games in 4 nights, travel distance, time zones, altitude, trip length
  lineup     who is available tonight, weighted by each player's minutes and box-score/plus-minus quality
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import teams

LOGIT_EPS = 1e-6


def logit(p):
    p = np.clip(p, LOGIT_EPS, 1 - LOGIT_EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def implied(ml):
    ml = np.asarray(ml, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))


def decimal_odds(ml):
    ml = np.asarray(ml, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ml < 0, 1 + 100 / -ml, 1 + ml / 100)


# --------------------------------------------------------------------------- market

def market_features(g: pd.DataFrame, spread_sd: float = 12.5) -> pd.DataFrame:
    ih, ia = implied(g["ml_home"]), implied(g["ml_away"])
    over = ih + ia
    ok = (g["ml_home"].abs() >= 100) & (g["ml_away"].abs() >= 100) & (over > 0.99) & (over < 1.15)
    p = np.where(ok, ih / over, np.nan)
    spread = g["spread"].where(g["spread"].abs() <= 25)
    total = g["total"].where(g["total"].between(150, 290))
    # Before 2022-23 the source stores the favorite's spread unsigned. Sign every spread by the
    # moneyline favorite (positive = home favored), which also matches the newer seasons 99% of the time.
    spread = (spread.abs() * np.sign(pd.Series(p, index=g.index) - 0.5)).where(pd.notna(p), spread)
    return pd.DataFrame({
        "p_market": p,
        "overround": np.where(ok, over, np.nan),
        "mkt_logit": logit(p),
        "spread": spread,
        "spread_logit": logit(norm.cdf(spread / spread_sd)),
        "total": total,
    }, index=g.index)


# --------------------------------------------------------------------------- Elo

@dataclass
class EloParams:
    k: float = 20.0
    hca: float = 90.0          # home-court edge in Elo points (tuned on 2008-09..2012-13)
    carry: float = 0.75        # share of last season's rating kept over the summer
    mean: float = 1505.0
    form_halflife: float = 8.0  # games


def elo_features(g: pd.DataFrame, prm: EloParams = EloParams(), return_state=False):
    """Pre-game Elo for both teams, plus 'form': how far a team has beaten its Elo expectation lately."""
    elo = defaultdict(lambda: prm.mean)
    form = defaultdict(float)
    a_form = 1 - 0.5 ** (1 / prm.form_halflife)
    last_season = {}
    out = np.full((len(g), 4), np.nan)
    H, A = g["home"].to_numpy(), g["away"].to_numpy()
    hp, ap = g["home_pts"].to_numpy(), g["away_pts"].to_numpy()
    S = g["season"].to_numpy()
    for i in range(len(g)):
        h, a, s = H[i], A[i], S[i]
        for t in (h, a):  # summer regression toward the mean
            if last_season.get(t) not in (None, s):
                elo[t] = prm.mean + prm.carry * (elo[t] - prm.mean)
                form[t] = 0.0
            last_season[t] = s
        eh, ea = elo[h], elo[a]
        out[i] = (eh, ea, form[h], form[a])
        if np.isnan(hp[i]) or np.isnan(ap[i]):
            continue  # game not played yet: no update
        diff = eh + prm.hca - ea
        exp_h = 1 / (1 + 10 ** (-diff / 400))
        margin = hp[i] - ap[i]
        win = 1.0 if margin > 0 else 0.0
        w_diff = diff if margin > 0 else -diff
        mult = (abs(margin) + 3) ** 0.8 / (7.5 + 0.006 * w_diff)
        delta = prm.k * mult * (win - exp_h)
        elo[h] += delta
        elo[a] -= delta
        resid = margin - diff / 28.0  # ~28 Elo points per point of margin
        form[h] = (1 - a_form) * form[h] + a_form * resid
        form[a] = (1 - a_form) * form[a] - a_form * resid
    df = pd.DataFrame(out, columns=["elo_home", "elo_away", "form_home", "form_away"], index=g.index)
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    df["elo_logit"] = (df["elo_diff"] + prm.hca) * np.log(10) / 400
    df["form_diff"] = df["form_home"] - df["form_away"]
    if return_state:
        return df, {"elo": dict(elo), "form": dict(form), "season": dict(last_season)}
    return df


def league_hca(g: pd.DataFrame, window: int = 800) -> pd.Series:
    """Average home margin over the previous `window` games. Home court has shrunk a lot since 2015;
    a fixed home edge learned from old seasons over-rates home teams, so the model sees this drift."""
    m = (g["home_pts"] - g["away_pts"])
    roll = m.rolling(window, min_periods=min(200, window)).mean().ffill()
    # value at the end of the previous day, so same-night results never leak in
    last_of_prev_day = roll.groupby(g["date"]).last().shift(1)
    return g["date"].map(last_of_prev_day).rename("league_hca")


# --------------------------------------------------------------------------- schedule

def schedule_features(g: pd.DataFrame) -> pd.DataFrame:
    """Rest, density, travel and time-zone features for both sides."""
    rows = []
    for i, r in enumerate(g.itertuples(index=False)):
        rows.append((i, r.date, r.home, r.home, True))
        rows.append((i, r.date, r.away, r.home, False))
    tg = pd.DataFrame(rows, columns=["i", "date", "team", "venue", "is_home"]).sort_values(["team", "date", "i"])
    grp = tg.groupby("team", sort=False)
    prev_date = grp["date"].shift(1)
    prev_venue = grp["venue"].shift(1)
    tg["rest"] = (tg["date"] - prev_date).dt.days.clip(upper=7).fillna(7)
    tg["b2b"] = (tg["rest"] == 1).astype(float)
    d3 = grp["date"].shift(2)
    tg["three_in_four"] = ((tg["date"] - d3).dt.days <= 3).astype(float)
    tg["travel_km"] = [teams.travel_km(p, v) if isinstance(p, str) else 0.0 for p, v in zip(prev_venue, tg["venue"])]
    tg["tz_shift"] = [teams.tz(v) - teams.tz(p) if isinstance(p, str) else 0 for p, v in zip(prev_venue, tg["venue"])]
    # length of the current road trip / home stand, counting this game
    streak = np.zeros(len(tg))
    last_team, last_home, run = None, None, 0
    for j, (t, h) in enumerate(zip(tg["team"].to_numpy(), tg["is_home"].to_numpy())):
        run = run + 1 if (t == last_team and h == last_home) else 1
        streak[j] = run
        last_team, last_home = t, h
    tg["streak"] = streak
    tg = tg.sort_values("i")
    hm, aw = tg[tg["is_home"]].set_index("i"), tg[~tg["is_home"]].set_index("i")
    f = pd.DataFrame(index=range(len(g)))
    for col in ["rest", "b2b", "three_in_four", "travel_km", "tz_shift"]:
        f[f"{col}_home"] = hm[col].to_numpy()
        f[f"{col}_away"] = aw[col].to_numpy()
    f["rest_diff"] = f["rest_home"].clip(upper=4) - f["rest_away"].clip(upper=4)
    f["road_trip_away"] = aw["streak"].to_numpy()
    f["home_stand"] = hm["streak"].to_numpy()
    f["altitude"] = [1.0 if teams.elevation(h) > 1000 and teams.elevation(a) < 1000 else 0.0
                     for h, a in zip(g["home"], g["away"])]
    f.index = g.index
    return f


# --------------------------------------------------------------------------- players and lineups

def _deque40():
    return deque(maxlen=40)


def _deque10():
    return deque(maxlen=10)


class PlayerState:
    """Running, as-of ratings for every player and recent rosters for every team.

    Quality is per-minute, shrunk toward replacement level until a player has enough minutes:
      q_gmsc  Game Score per minute       (box-score production)
      q_pm    plus-minus per minute       (on-court impact; noisy, so shrunk harder)
    base_min is the minutes a player gets when he plays (recent games weigh more).
    """

    def __init__(self, halflife_games=25, min_halflife=6, prior_gmsc=0.22, k_gmsc=150.0, k_pm=600.0,
                 roster_window=10, max_days_away=100):
        self.a = 1 - 0.5 ** (1 / halflife_games)
        self.am = 1 - 0.5 ** (1 / min_halflife)
        self.prior_gmsc, self.k_gmsc, self.k_pm = prior_gmsc, k_gmsc, k_pm
        self.window = roster_window
        self.max_days = max_days_away
        self.last_date = {}
        self.sg = defaultdict(float)
        self.spm = defaultdict(float)
        self.sm = defaultdict(float)
        self.base = {}
        self.name = {}
        self.last_team = {}
        self.last_idx = {}           # player -> team-game index of his last appearance for last_team
        self.team_n = defaultdict(int)
        self.team_recent = defaultdict(_deque40)
        self.team_strength_hist = defaultdict(_deque10)

    # ---- queries
    def q(self, p):
        m = self.sm.get(p, 0.0)
        qg = (self.sg.get(p, 0.0) + self.prior_gmsc * self.k_gmsc) / (m + self.k_gmsc)
        qpm = self.spm.get(p, 0.0) / (m + self.k_pm)
        return qg, qpm

    def candidates(self, team, date=None):
        """Players who have recently played for this team and have not since played for anyone else.
        Someone who has not played in `max_days_away` days (an offseason departure, a season-long injury)
        is no longer counted: the market and the ratings have long since absorbed his absence."""
        n = self.team_n[team]
        seen, out = set(), []
        for p in reversed(self.team_recent[team]):
            if p in seen:
                continue
            seen.add(p)
            if self.last_team.get(p) == team and n - self.last_idx.get(p, -99) <= self.window:
                ld = self.last_date.get(p)
                if date is None or ld is None or (date - ld).days <= self.max_days:
                    out.append(p)
        return out

    def lineup(self, team, avail: dict, newcomers=(), date=None):
        """Strength of tonight's group. avail maps player -> probability he plays (1 = certain, 0 = out).
        Returns a dict of features for one side."""
        cands = self.candidates(team, date)
        players = list(dict.fromkeys(list(cands) + list(newcomers)))
        if not players:
            return None
        base = np.array([self.base.get(p, 10.0) for p in players])
        qs = np.array([self.q(p) for p in players])
        pr = np.array([avail.get(p, 1.0 if p not in cands else 0.0) for p in players])
        full = _strength(base, qs, np.ones(len(players)) * np.array([p in cands for p in players]))
        tonight = _strength(base, qs, pr)
        miss_w = (1 - pr) * np.array([p in cands for p in players])
        star = (base * (qs[:, 0] - self.prior_gmsc)).clip(min=0)
        hist = self.team_strength_hist[team]
        recent = np.mean([h[0] for h in hist]) if hist else tonight[0]
        return {
            "str_gmsc": tonight[0], "str_pm": tonight[1],
            "loss_gmsc": full[0] - tonight[0], "loss_pm": full[1] - tonight[1],
            "delta_gmsc": tonight[0] - recent,
            "miss_min": float((base * miss_w).sum()),
            "star_out": float((star * miss_w).max(initial=0.0)),
        }

    # ---- updates (call after the games of a date are final)
    def update_team_game(self, team, rows: pd.DataFrame, tonight_strength=None, date=None):
        """rows: this team's box-score lines for one game (player_id, player, played, min, gmsc, pm)."""
        self.team_n[team] += 1
        n = self.team_n[team]
        if tonight_strength is not None:
            self.team_strength_hist[team].append(tonight_strength)
        for p, nm, played, mn, gs, pm in rows[["player_id", "player", "played", "min", "gmsc", "pm"]].itertuples(
                index=False, name=None):
            self.name[p] = nm
            if not played:
                continue
            # decayed sums: quality = decayed Game Score / decayed minutes
            self.sg[p] = (1 - self.a) * self.sg[p] + gs
            self.spm[p] = (1 - self.a) * self.spm[p] + pm
            self.sm[p] = (1 - self.a) * self.sm[p] + mn
            self.base[p] = mn if p not in self.base else (1 - self.am) * self.base[p] + self.am * mn
            self.last_team[p] = team
            self.last_idx[p] = n
            if date is not None:
                self.last_date[p] = date
            self.team_recent[team].append(p)


def _strength(base, qs, pr):
    """Minutes-weighted average quality of the players expected to play, scaled to 240 team minutes."""
    exp_min = base * pr
    tot = exp_min.sum()
    if tot <= 0:
        return np.array([np.nan, np.nan])
    alloc = np.minimum(exp_min * (240.0 / tot), 42.0)
    alloc = alloc * (240.0 / alloc.sum())
    return (alloc[:, None] * qs).sum(0) / 240.0


def lineup_features(g: pd.DataFrame, pg: pd.DataFrame, valid_from: str = "2011-08-01",
                    state: PlayerState | None = None):
    """Replay the player box scores in date order. For each game, rate tonight's available group using
    only earlier games, then update. Returns (features, final PlayerState)."""
    st = state or PlayerState()
    by_team_game = {k: v for k, v in pg.groupby(["game_id", "team"], sort=False)}
    cols = ["str_gmsc", "str_pm", "loss_gmsc", "loss_pm", "delta_gmsc", "miss_min", "star_out"]
    out = np.full((len(g), 2, len(cols)), np.nan)
    valid_from = pd.Timestamp(valid_from)
    for _, day in g.groupby("date", sort=True):
        pending = []
        for i, r in zip(day.index, day.itertuples(index=False)):
            for side, t in enumerate((r.home, r.away)):
                tr = by_team_game.get((r.game_id, t))
                if tr is None:
                    continue
                avail = dict(zip(tr["player_id"], tr["avail"].astype(float)))
                newcomers = [p for p, pl in avail.items() if pl > 0]
                f = st.lineup(t, avail, newcomers, date=r.date)
                if f is not None and r.date >= valid_from:
                    out[g.index.get_loc(i), side] = [f[c] for c in cols]
                pending.append((t, tr, None if f is None else (f["str_gmsc"], f["str_pm"]), r.date))
        for t, tr, s, d in pending:
            st.update_team_game(t, tr, s, d)
    df = pd.DataFrame(index=g.index)
    for j, c in enumerate(cols):
        df[f"{c}_home"] = out[:, 0, j]
        df[f"{c}_away"] = out[:, 1, j]
    df["d_str_gmsc"] = df["str_gmsc_home"] - df["str_gmsc_away"]
    df["d_str_pm"] = df["str_pm_home"] - df["str_pm_away"]
    df["d_loss_gmsc"] = df["loss_gmsc_home"] - df["loss_gmsc_away"]
    df["d_delta_gmsc"] = df["delta_gmsc_home"] - df["delta_gmsc_away"]
    df["d_miss_min"] = df["miss_min_home"] - df["miss_min_away"]
    df["has_lineup"] = df["d_str_gmsc"].notna().astype(float)
    return df, st


# --------------------------------------------------------------------------- assemble

MARKET = ["mkt_logit", "spread_logit", "total"]
RATINGS = ["elo_logit", "elo_diff", "form_diff", "league_hca", "season_frac"]
SCHEDULE = ["rest_diff", "b2b_home", "b2b_away", "three_in_four_home", "three_in_four_away", "travel_km_away",
            "tz_shift_away", "road_trip_away", "home_stand", "altitude"]
LINEUP = ["d_str_gmsc", "d_str_pm", "d_loss_gmsc", "d_delta_gmsc", "d_miss_min", "star_out_home", "star_out_away",
          "has_lineup"]
STATS = RATINGS + SCHEDULE + LINEUP


def build(g: pd.DataFrame, pg: pd.DataFrame | None, elo: EloParams = EloParams(), lineup_from="2011-08-01"):
    g = g.reset_index(drop=True)
    parts = [g.drop(columns=["spread", "total"]), market_features(g), elo_features(g, elo), league_hca(g), schedule_features(g)]
    if pg is not None and len(pg):
        lf, _ = lineup_features(g, pg, valid_from=lineup_from)
        parts.append(lf)
    X = pd.concat(parts, axis=1)
    for c in LINEUP:
        if c not in X:
            X[c] = np.nan
    X["has_lineup"] = X["d_str_gmsc"].notna().astype(float)
    n_season = X.groupby("season").cumcount()
    X["season_frac"] = (n_season / 1230).clip(upper=1.2)
    X["home_win"] = np.where(X["home_pts"].notna(), (X["home_pts"] > X["away_pts"]).astype(float), np.nan)
    return X
