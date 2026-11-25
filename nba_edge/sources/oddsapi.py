"""Optional: every US sportsbook's moneyline and total from The Odds API (https://the-odds-api.com).

Free plan: 500 credits a month. Each call asks for two markets (moneyline + totals) = 2 credits, so the live loop
calls it at most every 2 hours between 7 AM and 8 PM Pacific: about 7 calls, 14 credits a day, ~420 a month.
Set the ODDS_API_KEY environment variable (or GitHub secret) to turn it on.
"""
from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

from .. import teams

URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
EVERY_HOURS = 2
WINDOW_PT = (7, 20)


def enabled() -> bool:
    return bool(os.environ.get("ODDS_API_KEY"))


def due(last_iso: str | None) -> bool:
    """True when a call fits the free-plan budget: inside the daytime window and 2+ hours since the last one."""
    now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
    if not (WINDOW_PT[0] <= now_pt.hour < WINDOW_PT[1]):
        return False
    if not last_iso:
        return True
    last = datetime.fromisoformat(last_iso)
    return (datetime.now(last.tzinfo) - last).total_seconds() >= EVERY_HOURS * 3600 - 300


def lines():
    """One row per game per book: moneyline and total with over/under prices. Also returns credits left."""
    r = requests.get(URL, params={"apiKey": os.environ["ODDS_API_KEY"], "regions": "us", "markets": "h2h,totals",
                                  "oddsFormat": "american"}, timeout=20)
    r.raise_for_status()
    left = r.headers.get("x-requests-remaining")
    rows = []
    for ev in r.json():
        h, a = teams.code(ev["home_team"]), teams.code(ev["away_team"])
        start = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        gid = start.astimezone(ZoneInfo("America/New_York")).strftime("%Y%m%d") + h + a
        for b in ev.get("bookmakers", []):
            row = {"game_id": gid, "book": b["title"]}
            for m in b.get("markets", []):
                if m["key"] == "h2h":
                    px = {teams.code(o["name"]): o["price"] for o in m["outcomes"]}
                    if h in px and a in px:
                        row["ml_home"], row["ml_away"] = float(px[h]), float(px[a])
                elif m["key"] == "totals":
                    for o in m["outcomes"]:
                        if o["name"].lower() == "over":
                            row["total"], row["over_price"] = float(o.get("point")), float(o["price"])
                        elif o["name"].lower() == "under":
                            row["under_price"] = float(o["price"])
            if len(row) > 2:
                rows.append(row)
    return rows, left


def moneylines():  # backwards compatible name
    return lines()[0]
