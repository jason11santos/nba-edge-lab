"""SQLite store for everything the live pipeline sees. Snapshots are append-only, so line movement,
injury-report changes and what the model said before tip-off can all be replayed later."""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone

import pandas as pd

SCHEMA = """
create table if not exists games (game_id text primary key, espn_id text, date text, season_type int,
  home text, away text, start_utc text, state text, status text, home_pts real, away_pts real, neutral int);
create table if not exists odds (ts text, game_id text, source text, kind text, ml_home real, ml_away real,
  ml_home_open real, ml_away_open real, spread real, spread_open real, total real, total_open real);
create index if not exists odds_g on odds(game_id, ts);
create table if not exists injuries (ts text, team text, player text, status text, fantasy text, play_prob real,
  return_date text, note text);
create table if not exists player_games (game_id text, date text, team text, player_id text, player text,
  played int, avail real, min real, gmsc real, pm real, source text, primary key (game_id, team, player_id));
create table if not exists predictions (ts text, game_id text, p_market real, p_ensemble real, p_mkt_gbm real,
  p_mkt_lr real, p_stats_lr real, p_stats_gbm real, side text, price real, book text, ev real, stake real);
create index if not exists pred_g on predictions(game_id, ts);
create table if not exists bets (game_id text primary key, ts text, side text, price real, book text, p_model real,
  p_fair_at_bet real, ev real, stake real, p_fair_close real, clv real, won int, pnl real);
create table if not exists paper_bets (rule text, game_id text, ts text, side text, price real, book text,
  p_model real, p_fair_at_bet real, ev real, stake real, p_fair_close real, clv real, won int, pnl real,
  primary key (rule, game_id));
create table if not exists meta (key text primary key, value text);
create table if not exists officials (game_id text, ref text, ord int, primary key (game_id, ref));
create table if not exists summaries (game_id text primary key, ts text, n_refs int, n_players int);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


MIGRATIONS = {"odds": ["over_price real", "under_price real", "over_price_open real", "under_price_open real"],
              "paper_bets": ["line real"]}


class Store:
    def __init__(self, path):
        self.con = sqlite3.connect(path, timeout=60)
        self.con.executescript(SCHEMA)
        for table, cols in MIGRATIONS.items():
            have = {r[1] for r in self.con.execute(f"pragma table_info({table})")}
            for c in cols:
                if c.split()[0] not in have:
                    self.con.execute(f"alter table {table} add column {c}")
        self.con.commit()

    def upsert_games(self, rows):
        self.con.executemany(
            "insert into games values (:game_id,:espn_id,:date,:season_type,:home,:away,:start_utc,:state,:status,"
            ":home_pts,:away_pts,:neutral) on conflict(game_id) do update set state=excluded.state, "
            "status=excluded.status, home_pts=excluded.home_pts, away_pts=excluded.away_pts, "
            "start_utc=excluded.start_utc, espn_id=excluded.espn_id", rows)
        self.con.commit()

    def add_odds(self, game_id, source, kind, o, ts=None):
        keys = ("ml_home", "ml_away", "spread", "total", "over_price", "under_price")
        prev = self.con.execute("select ml_home, ml_away, spread, total, over_price, under_price from odds where game_id=?"
                                " and source=? and kind=? order by ts desc limit 1", (game_id, source, kind)).fetchone()
        if prev is not None and all((a is None and o.get(k) is None) or (a is not None and o.get(k) is not None and
                                    abs(a - o.get(k)) < 1e-9) for a, k in zip(prev, keys)):
            return  # unchanged since the last snapshot
        self.con.execute("insert into odds (ts, game_id, source, kind, ml_home, ml_away, ml_home_open, ml_away_open,"
                         " spread, spread_open, total, total_open, over_price, under_price, over_price_open,"
                         " under_price_open) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (ts or now(), game_id, source, kind, o.get("ml_home"), o.get("ml_away"),
                          o.get("ml_home_open"), o.get("ml_away_open"), o.get("spread"), o.get("spread_open"),
                          o.get("total"), o.get("total_open"), o.get("over_price"), o.get("under_price"),
                          o.get("over_price_open"), o.get("under_price_open")))
        self.con.commit()

    def add_injuries(self, rows):
        sig = repr(sorted((r["team"] or "", r["player"] or "", r["status"] or "", r["fantasy"] or "") for r in rows))
        if self.meta("injury_sig") == hashlib.md5(sig.encode()).hexdigest() and self.con.execute("select 1 from injuries limit 1").fetchone():
            return  # report unchanged
        self.meta("injury_sig", hashlib.md5(sig.encode()).hexdigest())
        ts = now()
        self.con.executemany("insert into injuries values (?,?,?,?,?,?,?,?)",
                             [(ts, r["team"], r["player"], r["status"], r["fantasy"], r["play_prob"],
                               r["return_date"], r["note"]) for r in rows])
        self.con.commit()

    def add_player_games(self, df: pd.DataFrame, source: str):
        rows = df.assign(source=source, date=df["date"].astype(str).str[:10], player_id=df["player_id"].astype(str))
        self.con.executemany(
            "insert or replace into player_games values (?,?,?,?,?,?,?,?,?,?,?)",
            rows[["game_id", "date", "team", "player_id", "player", "played", "avail", "min", "gmsc", "pm",
                  "source"]].itertuples(index=False, name=None))
        self.con.commit()

    def q(self, sql, params=()):
        return pd.read_sql(sql, self.con, params=params)

    def meta(self, key, value=None):
        if value is None:
            r = self.con.execute("select value from meta where key=?", (key,)).fetchone()
            return r[0] if r else None
        self.con.execute("insert or replace into meta values (?,?)", (key, str(value)))
        self.con.commit()

    def latest_odds(self):
        """Most recent line per game and source (closing line for finished games)."""
        return self.q("""select o.* from odds o join (select game_id, source, max(ts) ts from odds group by 1,2) m
                         using (game_id, source, ts)""")
