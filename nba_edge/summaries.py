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
VERSION = 2                       # 2: also keeps points/rebounds/assists/... and quarter scores


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
    done = set(st.q("select game_id from summaries where v >= ?", (VERSION,))["game_id"])
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
            try:
                st.add_linescores([(r.game_id, t, per, pts) for t, per, pts in espn.linescores(js.get("header") or {})])
            except Exception:  # noqa: BLE001
                pass
            if r.date > HIST_PLAYERS_END:
                try:
                    pl = pd.DataFrame(espn.box_rows(js))
                except (KeyError, TypeError, ValueError) as e:  # odd box score (All-Star game, missing athlete)
                    _log("box score skipped", r.game_id, repr(e))
                    pl = pd.DataFrame()
                if len(pl):
                    pl["player_id"] = "espn:" + pl["espn_id"].astype(str)
                    pl["game_id"], pl["date"] = r.game_id, r.date
                    st.add_player_games(pl, "espn")
                    npl = len(pl)
            st.con.execute("insert or replace into summaries (game_id, ts, n_refs, n_players, v) values (?,?,?,?,?)",
                           (r.game_id, now(), len(refs), npl, VERSION))
            n_ok += 1
            if k % 250 == 249:
                st.con.commit()
                _log(f"  {k + 1}/{len(rows)}")
    st.con.commit()
    _log(f"summaries: {n_ok} fetched")
    return n_ok


def _nba_page():
    """Today's crews from the NBA's referee-assignments page (posted ~9am Eastern). Returns a list of
    (set of team codes in the row, [referee names]). Best effort: the page layout is not an API."""
    import re

    import requests

    from . import teams
    html = requests.get("https://official.nba.com/referee-assignments/", timeout=20,
                        headers={"User-Agent": "Mozilla/5.0 (nba-edge-lab research project)"}).text
    keys = {}
    for code, (full, *_rest) in teams.TEAMS.items():
        nick = full.split(" ", 2)[-1] if full.startswith(("Los Angeles", "Golden State", "New York", "New Orleans",
                                                            "Oklahoma City", "San Antonio")) else full.split(" ", 1)[-1]
        keys[code] = {full.lower(), nick.lower(), full.lower().replace(nick.lower(), "").strip()}
    keys["LAL"] |= {"l.a. lakers", "la lakers"}
    keys["LAC"] |= {"l.a. clippers", "la clippers"}
    keys["LAL"].discard("los angeles")
    keys["LAC"].discard("la")
    out = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S | re.I):
        cells = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", c)).strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.S | re.I)]
        if len(cells) < 3 or "@" not in cells[0]:
            continue
        a, h = [x.strip().lower() for x in cells[0].split("@", 1)]
        codes = {c for c, ks in keys.items() for side in (a, h) if side in ks}
        refs = [re.sub(r"\s*\(#?\d+\)", "", c).strip() for c in cells[1:4]]
        refs = [r for r in refs if r and not r.isdigit()]
        if len(codes) == 2 and refs:
            out.append((codes, refs))
    return out


def linescore_history(st: Store, dates, threads: int = 8):
    """Quarter scores for older seasons from ESPN scoreboards (one request per date, every game that day).
    Used by the first-half / quarter study. Skips dates already stored."""
    from . import teams  # noqa: F401
    have = set(st.q("select distinct substr(game_id, 1, 8) d from linescores")["d"])
    todo = [d for d in sorted(set(dates)) if d.replace("-", "") not in have]
    if not todo:
        return 0
    _log(f"fetching quarter scores for {len(todo)} dates")

    def get(d):
        try:
            return espn._get(f"{espn.SITE}/scoreboard", {"dates": d.replace("-", ""), "limit": 50})
        except Exception as e:  # noqa: BLE001
            return e
    n = 0
    with ThreadPoolExecutor(threads) as ex:
        for k, (d, js) in enumerate(zip(todo, ex.map(get, todo))):
            if not isinstance(js, dict):
                continue
            rows = []
            for ev in js.get("events", []):
                comp = ev["competitions"][0]
                side = {c.get("homeAway"): espn._code((c.get("team") or {}).get("abbreviation", ""))
                        for c in comp.get("competitors", [])}
                if not side.get("home") or not side.get("away"):
                    continue
                gid = d.replace("-", "") + side["home"] + side["away"]
                rows += [(gid, t, per, pts) for t, per, pts in espn.linescores(comp)]
            st.add_linescores(rows)
            n += len(rows) > 0
            if k % 200 == 199:
                st.con.commit()
                _log(f"  {k + 1}/{len(todo)}")
    st.con.commit()
    _log(f"quarter scores: {n} dates")
    return n


def today(st: Store, games: pd.DataFrame):
    """Crews for upcoming games (state 'pre'), if ESPN has posted them. Stored like finished games' crews."""
    found = 0
    page = None
    try:
        page = _nba_page()
        _log("NBA referee page:", len(page), "games", page[:1])
    except Exception as e:  # noqa: BLE001
        _log("NBA referee page unavailable:", repr(e)[:120])
    if page:
        g2 = st.q("select game_id, home, away from games where game_id in (%s)" % ",".join("?" * len(games)),
                  tuple(games["game_id"])) if len(games) else pd.DataFrame()
        for r in g2.itertuples(index=False):
            for codes, refs in page:
                if codes == {r.home, r.away}:
                    st.con.executemany("insert or replace into officials values (?,?,?)",
                                       [(r.game_id, nm, i) for i, nm in enumerate(refs)])
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
