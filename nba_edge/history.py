"""Load the historical sources into two tidy tables.

games          one row per game: date, season, home, away, scores, moneylines, spread, total
player_games   one row per player per team-game: who was available, minutes and box-score stats

Sources (all public, see README):
  * kyleskom/NBA-Machine-Learning-Sports-Betting   Data/OddsData.sqlite   moneyline, spread, total, final score
                                                   Data/nba-YYYY-UTC.csv  current-season schedule and results
  * NocturneBear/NBA-Data-2010-2024                player box scores 2010-11 to 2023-24, with DNP reasons
"""
from __future__ import annotations

import glob
import os
import re
import sqlite3

import numpy as np
import pandas as pd

from . import teams

ODDS_COLS = ["Date", "Home", "Away", "OU", "Spread", "ML_Home", "ML_Away", "Points", "Win_Margin"]


def load_odds(sqlite_path: str) -> pd.DataFrame:
    """All seasons of the odds database, one clean row per game."""
    con = sqlite3.connect(sqlite_path)
    tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
    # Prefer the cleaned '*_new' / plain-season tables; the older un-suffixed tables have broken dates.
    keep = [t for t in tables if t.endswith("_new") or re.fullmatch(r"(odds_)?\d{4}-\d{2}", t)]
    frames = []
    for t in keep:
        df = pd.read_sql(f'select * from "{t}"', con)
        if not {"Date", "Home"}.issubset(df.columns):
            continue
        df = df[ODDS_COLS]
        df = df[df["Date"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")]
        frames.append(df)
    con.close()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["Date"])
    for c in ["OU", "Spread", "ML_Home", "ML_Away", "Points", "Win_Margin"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["home"] = df["Home"].map(teams.code)
    df["away"] = df["Away"].map(teams.code)
    # Points = total, Win_Margin = home minus away
    df["home_pts"] = (df["Points"] + df["Win_Margin"]) / 2
    df["away_pts"] = (df["Points"] - df["Win_Margin"]) / 2
    df = df.rename(columns={"ML_Home": "ml_home", "ML_Away": "ml_away", "Spread": "spread", "OU": "total"})
    out = df[["date", "home", "away", "home_pts", "away_pts", "ml_home", "ml_away", "spread", "total"]]
    out = out.drop_duplicates(subset=["date", "home", "away"], keep="last")
    return out.sort_values(["date", "home"]).reset_index(drop=True)


def _is_team(x) -> bool:
    try:
        teams.code(str(x))
        return True
    except KeyError:
        return False


def load_schedule_csv(path: str) -> pd.DataFrame:
    """fixturedownload-style season file: '20/10/2026 19:00' in UTC, Result '121 - 113'."""
    df = pd.read_csv(path)
    known = lambda s: s.map(lambda x: _is_team(x))  # noqa: E731  (drops 'To be announced' playoff slots)
    df = df[known(df["Home Team"]) & known(df["Away Team"])].copy()
    utc = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M", utc=True)
    # Tip-off times are UTC; a 7:30pm ET game is 00:30 UTC the next day, so convert back to US Eastern.
    df["date"] = utc.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    df["home"] = df["Home Team"].map(teams.code)
    df["away"] = df["Away Team"].map(teams.code)
    sc = df["Result"].astype(str).str.extract(r"(\d+)\s*-\s*(\d+)").astype(float)
    df["home_pts"], df["away_pts"] = sc[0], sc[1]
    for c in ["ml_home", "ml_away", "spread", "total"]:
        df[c] = np.nan
    return df[["date", "home", "away", "home_pts", "away_pts", "ml_home", "ml_away", "spread", "total"]]


def load_games(odds_sqlite: str, schedule_csvs: list[str] | None = None) -> pd.DataFrame:
    g = load_odds(odds_sqlite)
    for p in schedule_csvs or []:
        s = load_schedule_csv(p)
        s = s[s["date"] > g["date"].max()]  # only add games the odds file does not cover
        g = pd.concat([g, s], ignore_index=True)
    g["season"] = g["date"].map(teams.season_of)
    g = g.sort_values(["date", "home"]).reset_index(drop=True)
    g["game_id"] = g["date"].dt.strftime("%Y%m%d") + g["home"] + g["away"]
    return g


def _minutes(s: pd.Series) -> pd.Series:
    """'34:12' -> 34.2 ; blanks -> 0."""
    s = s.fillna("").astype(str)
    mm = s.str.extract(r"^(\d+):(\d+)")
    out = mm[0].astype(float) + mm[1].astype(float) / 60
    num = pd.to_numeric(s.where(mm[0].isna()), errors="coerce")
    return out.fillna(num).fillna(0.0)


# Raw box-score columns kept for player-level models (props). Same names in the live store.
STAT_SRC = {"pts": "points", "reb": "reboundsTotal", "ast": "assists", "fg3m": "threePointersMade",
            "fga": "fieldGoalsAttempted", "fta": "freeThrowsAttempted", "tov": "turnovers", "oreb": "reboundsOffensive",
            "stl": "steals", "blk": "blocks"}
STATS = list(STAT_SRC)


def load_player_games(folder: str, games: pd.DataFrame) -> pd.DataFrame:
    """Player box scores, attached to our game ids by (date, team): a team plays at most once a day."""
    files = sorted(glob.glob(os.path.join(folder, "regular_season_box_scores_*.csv")))
    files += sorted(glob.glob(os.path.join(folder, "play_off_box_scores_*.csv")))
    df = pd.concat([pd.read_csv(f, dtype={"comment": str, "position": str, "jerseyNum": str}) for f in files],
                   ignore_index=True)
    df["date"] = pd.to_datetime(df["game_date"])
    df["team"] = df["teamTricode"].map(teams.code)
    df["min"] = _minutes(df["minutes"])
    df["played"] = df["min"] > 0
    st = lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0)  # noqa: E731
    # John Hollinger's Game Score: a one-number box-score summary of a player's night.
    df["gmsc"] = (st("points") + 0.4 * st("fieldGoalsMade") - 0.7 * st("fieldGoalsAttempted")
                  - 0.4 * (st("freeThrowsAttempted") - st("freeThrowsMade")) + 0.7 * st("reboundsOffensive")
                  + 0.3 * st("reboundsDefensive") + st("steals") + 0.7 * st("assists") + 0.7 * st("blocks")
                  - 0.4 * st("foulsPersonal") - st("turnovers"))
    df["pm"] = st("plusMinusPoints")
    for c, src in STAT_SRC.items():
        df[c] = st(src)
    df["status"] = df["comment"].fillna("").str.split(" - ").str[0].str.strip()
    # Available = dressed and healthy, which is what the pre-game inactive list tells you.
    # "DNP - Coach's Decision" was available; DND (injury/rest), NWT and DNP-injury were not;
    # players missing from the box score entirely were inactive. Using "logged minutes" instead
    # would leak the game script (deep benches only get in during blowouts).
    cmt = df["comment"].fillna("")
    df["avail"] = (df["played"] | cmt.str.contains("Coach", case=False)).astype(float)
    key = pd.concat([games[["date", "home", "game_id"]].rename(columns={"home": "team"}),
                     games[["date", "away", "game_id"]].rename(columns={"away": "team"})])
    df = df.merge(key, on=["date", "team"], how="inner")
    out = df[["game_id", "date", "team", "personId", "personName", "played", "avail", "min", "gmsc", "pm", "status"]
             + list(STAT_SRC)]
    out = out.rename(columns={"personId": "player_id", "personName": "player"})
    out = out.drop_duplicates(subset=["game_id", "team", "player_id"])
    return out.sort_values(["date", "game_id", "team"]).reset_index(drop=True)
