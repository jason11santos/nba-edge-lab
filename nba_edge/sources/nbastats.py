"""Player game logs from stats.nba.com via the nba_api package (pip install nba_api).

One request per season returns every player's line for every game. Player ids are the NBA's own,
so they line up with the historical box scores. stats.nba.com sometimes throttles or blocks
requests; the live pipeline falls back to ESPN box scores when this fails.
"""
from __future__ import annotations

import time

import pandas as pd

from .. import teams


def season_logs(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamelog

    for k in range(3):
        try:
            df = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation="P",
                                             season_type_all_star=season_type, timeout=60).get_data_frames()[0]
            break
        except Exception:  # network errors, throttling
            if k == 2:
                raise
            time.sleep(5 * (k + 1))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["GAME_DATE"])
    df["team"] = df["TEAM_ABBREVIATION"].map(teams.code)
    m = df["MATCHUP"].str.extract(r"^(\w+)\s+(vs\.|@)\s+(\w+)$")
    opp = m[2].map(teams.code)
    home = df["team"].where(m[1].eq("vs."), opp)
    away = opp.where(m[1].eq("vs."), df["team"])
    df["game_id"] = df["date"].dt.strftime("%Y%m%d") + home + away
    df["min"] = pd.to_numeric(df["MIN"], errors="coerce").fillna(0)
    g = lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0)  # noqa: E731
    df["gmsc"] = (g("PTS") + 0.4 * g("FGM") - 0.7 * g("FGA") - 0.4 * (g("FTA") - g("FTM")) + 0.7 * g("OREB")
                  + 0.3 * g("DREB") + g("STL") + 0.7 * g("AST") + 0.7 * g("BLK") - 0.4 * g("PF") - g("TOV"))
    df["pm"] = g("PLUS_MINUS")
    df["played"] = df["min"] > 0
    # This feed lists only players who got in, so "available" cannot be told apart from "played".
    # Rows from it update player ratings but are never used as lineup training data (see live.py).
    df["avail"] = df["played"].astype(float)
    out = df.rename(columns={"PLAYER_ID": "player_id", "PLAYER_NAME": "player"})
    return out[["game_id", "date", "team", "player_id", "player", "played", "avail", "min", "gmsc", "pm"]]
