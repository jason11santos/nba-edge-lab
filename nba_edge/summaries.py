"""ESPN game summaries: one request per game gives the box score AND the referee crew.

Used to (1) fill the player data the cloud is missing between the end of the downloaded history (April 2024) and the
start of live collection (January 2026), and (2) record every game's officials since the 2023-24 season, for the
referee study. Going forward, each finished game's summary is fetched once by `fill`, and today's crews are polled
by `today` (the NBA posts assignments around 9am Eastern; ESPN usually copies them some time before tip-off).
"""
from __future__ import annotations

import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

from .sources import espn
from .store import Store, now

HIST_PLAYERS_END = "2024-04-28"   # last date in the downloaded player box scores
OFFICIALS_FROM = "2023-10-24"


def _log(*a):
    print(datetime.now(timezone.utc).strftime("%H:%M:%S"), *a, flush=True)


def _games(st: Store, ol_db: str | None):
    g = st.q("select game_id, espn_id, date, state from games where espn_id is not null")
    g = g[g["state"] == "post"][["game_id", "espn_id", "date"]]
    if ol_db and os.path.exists(ol_db):
        con = sqlite3.connect(ol_db)
        o = pd.read_sql("select game_id, espn_id, date from ol_games where home_pts is not null", con)
        con.close()
        g = pd.concat([o, g], ignore_index=True)
    g = g.dropna(subset=["espn_id"]).drop_duplicates("game_id", keep="last")
    return g[g["date"] >= OFFICIALS_FROM].sort_values("date")


def _fetch(espn_id):
    try:
        return espn.summary(str(espn_id))
    except Exception as e:  # noqa: BLE001
        return e


def fill(st: Store, ol_db: str | None = None, budget: int = 5000, threads: int = 8):
    """Fetch the summary of every finished game since 2023-24 not fetched yet. Officials for all of them;
    player lines for dates the downloaded history does not cover (and only if the store has none for that game)."""
    done = set(st.q("select game_id from summaries")["game_id"])
    todo = _games(st, ol_db)
    todo = todo[~todo["game_id"].isin(done)].head(budget)
    if todo.empty:
        return 0
    have_pl = set(st.q("select distinct game_id from player_games")["game_id"])
    _log(f"fetching {len(todo)} game summaries (box scores + referees)")
    n_ok = 0
    rows = list(todo.itertuples(index=False))
    with ThreadPoolExecutor(threads) as ex:
        for k, (r, js) in enumerate(zip(rows, ex.map(lambda r: _fetch(r.espn_id), rows))):
            if isinstance(js, Exception) or not isinstance(js, dict):
                continue
            refs = espn.officials(js)
            st.con.executemany("insert or replace into officials values (?,?,?)",
                               [(r.game_id, nm, i) for i, nm in enumerate(refs)])
            npl = 0
            if r.date > HIST_PLAYERS_END and r.game_id not in have_pl:
                pl = pd.DataFrame(espn.box_rows(js))
                if len(pl):
                    pl["player_id"] = "espn:" + pl["espn_id"].astype(str)
                    pl["game_id"], pl["date"] = r.game_id, r.date
                    st.add_player_games(pl, "espn")
                    npl = len(pl)
            st.con.execute("insert or replace into summaries values (?,?,?,?)", (r.game_id, now(), len(refs), npl))
            n_ok += 1
            if k % 250 == 249:
                st.con.commit()
                _log(f"  {k + 1}/{len(rows)}")
    st.con.commit()
    _log(f"summaries: {n_ok} fetched")
    return n_ok


def today(st: Store, games: pd.DataFrame):
    """Crews for upcoming games (state 'pre'), if ESPN has posted them. Stored like finished games' crews."""
    found = 0
    for r in games.itertuples(index=False):
        if getattr(r, "state", "pre") != "pre" or not getattr(r, "espn_id", None):
            continue
        have = st.q("select count(*) n from officials where game_id=?", (r.game_id,))["n"].iloc[0]
        if have >= 3:
            found += 1
            continue
        js = _fetch(r.espn_id)
        if isinstance(js, dict):
            refs = espn.officials(js)
            if refs:
                st.con.executemany("insert or replace into officials values (?,?,?)",
                                   [(r.game_id, nm, i) for i, nm in enumerate(refs)])
                found += 1
    st.con.commit()
    return found
