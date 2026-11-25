"""Optional: every US sportsbook's moneyline from The Odds API (https://the-odds-api.com).

The free plan allows 500 requests a month, enough for a poll every ~90 minutes on game days. Set the
ODDS_API_KEY environment variable to turn it on. With several books you can bet the best price
on each side ("line shopping"), which on its own is worth more than most model tweaks.
"""
from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

from .. import teams

URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"


def enabled() -> bool:
    return bool(os.environ.get("ODDS_API_KEY"))


def moneylines():
    """One row per game per book: game_id, book, ml_home, ml_away."""
    r = requests.get(URL, params={"apiKey": os.environ["ODDS_API_KEY"], "regions": "us", "markets": "h2h",
                                  "oddsFormat": "american"}, timeout=20)
    r.raise_for_status()
    rows = []
    for ev in r.json():
        h, a = teams.code(ev["home_team"]), teams.code(ev["away_team"])
        start = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        gid = start.astimezone(ZoneInfo("America/New_York")).strftime("%Y%m%d") + h + a
        for b in ev.get("bookmakers", []):
            for m in b.get("markets", []):
                if m["key"] != "h2h":
                    continue
                px = {teams.code(o["name"]): o["price"] for o in m["outcomes"]}
                if h in px and a in px:
                    rows.append({"game_id": gid, "book": b["title"], "ml_home": float(px[h]),
                                 "ml_away": float(px[a])})
    return rows
