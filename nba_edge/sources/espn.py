"""ESPN's public JSON feeds: schedule, scores, DraftKings lines (open and current), injuries, box scores.

No key is needed. Formats were checked against the live feeds on 2026-11-24. ESPN's spread is the
home team's line (negative = home favored); everything here converts to this project's convention
(positive = home favored).
"""
from __future__ import annotations

import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

import requests

from .. import teams

SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
ET = ZoneInfo("America/New_York")
HEADERS = {"User-Agent": "Mozilla/5.0 (nba-edge-lab research project)"}
_session = requests.Session()
_session.headers.update(HEADERS)


def _get(url, params=None, tries=3):
    for k in range(tries):
        try:
            r = _session.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if k == tries - 1:
                raise
            time.sleep(1.5 * (k + 1))


def _num(x):
    """'+110' -> 110.0, 'o214.5' -> 214.5, 'EVEN' -> 100.0, None -> None"""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().lstrip("ou")
    if s.upper() in ("EVEN", "EV", "PK"):
        return 100.0 if s.upper().startswith("EV") else 0.0
    try:
        return float(s.replace("+", ""))
    except ValueError:
        return None


def _code(abbr):
    try:
        return teams.code(abbr)
    except KeyError:
        return None  # All-Star teams and exhibition opponents


def _odds_from_scoreboard(c):
    o = (c.get("odds") or [None])[0]
    if not o:
        return None
    ml, ps, tot = o.get("moneyline") or {}, o.get("pointSpread") or {}, o.get("total") or {}
    g = lambda d, side, when, k: _num(((d.get(side) or {}).get(when) or {}).get(k))  # noqa: E731
    home_line = o.get("spread")
    return {
        "provider": (o.get("provider") or {}).get("name"),
        "ml_home": g(ml, "home", "close", "odds"), "ml_away": g(ml, "away", "close", "odds"),
        "ml_home_open": g(ml, "home", "open", "odds"), "ml_away_open": g(ml, "away", "open", "odds"),
        "spread": None if home_line is None else -float(home_line),
        "spread_open": None if g(ps, "home", "open", "line") is None else -g(ps, "home", "open", "line"),
        "total": _num(o.get("overUnder")), "total_open": g(tot, "over", "open", "line"),
        "over_price": g(tot, "over", "close", "odds"), "under_price": g(tot, "under", "close", "odds"),
        "over_price_open": g(tot, "over", "open", "odds"), "under_price_open": g(tot, "under", "open", "odds"),
    }


def scoreboard(day: date | str | None = None):
    """Games on one US date (ESPN groups by Eastern date). Returns a list of dicts."""
    params = {"limit": 50}
    if day is not None:
        params["dates"] = (day if isinstance(day, str) else day.strftime("%Y%m%d")).replace("-", "")
    js = _get(f"{SITE}/scoreboard", params)
    out = []
    for e in js.get("events", []):
        c = e["competitions"][0]
        comp = {t["homeAway"]: t for t in c["competitors"]}
        if "home" not in comp or "away" not in comp:
            continue
        h = _code((comp["home"].get("team") or {}).get("abbreviation"))
        a = _code((comp["away"].get("team") or {}).get("abbreviation"))
        if not h or not a:
            continue
        start = datetime.fromisoformat(e["date"].replace("Z", "+00:00"))
        local = start.astimezone(ET).date()
        st = c["status"]["type"]
        out.append({
            "espn_id": e["id"], "start_utc": start.isoformat(), "date": local.isoformat(),
            "season_type": (e.get("season") or {}).get("type"), "state": st.get("state"),
            "status": st.get("shortDetail"), "home": h, "away": a,
            "home_pts": _num(comp["home"].get("score")) if st.get("state") != "pre" else None,
            "away_pts": _num(comp["away"].get("score")) if st.get("state") != "pre" else None,
            "neutral": bool(c.get("neutralSite")),
            "odds": _odds_from_scoreboard(c),
            "game_id": local.strftime("%Y%m%d") + h + a,
        })
    return out


def closing_odds(espn_id: str):
    """Opening and closing DraftKings lines for a finished game (the scoreboard drops them)."""
    js = _get(f"{CORE}/events/{espn_id}/competitions/{espn_id}/odds")
    it = (js.get("items") or [None])[0]
    if not it:
        return None
    h, a = it.get("homeTeamOdds") or {}, it.get("awayTeamOdds") or {}
    am = lambda d, when: _num(((d.get(when) or {}).get("moneyLine") or {}).get("american"))  # noqa: E731
    sp = it.get("spread")
    return {"provider": (it.get("provider") or {}).get("name"),
            "ml_home": am(h, "close") or _num(h.get("moneyLine")), "ml_away": am(a, "close") or _num(a.get("moneyLine")),
            "ml_home_open": am(h, "open"), "ml_away_open": am(a, "open"),
            "spread": None if sp is None else -float(sp), "total": _num(it.get("overUnder")),
            "over_price": _num(it.get("overOdds")), "under_price": _num(it.get("underOdds")),
            "total_open": _num(((it.get("open") or {}).get("total") or {}).get("american"))}


# Probability a player suits up, by injury-report status. Rough league-wide base rates.
STATUS_PLAY_PROB = {"out": 0.0, "doubtful": 0.15, "questionable": 0.55, "day-to-day": 0.6, "gtd": 0.6,
                    "probable": 0.9, "active": 1.0}


def play_prob(status: str, fantasy: str, note: str) -> float:
    """ESPN files 'probable' and 'questionable' players under Day-To-Day; the note says which."""
    s, n = (status or "").lower(), (note or "").lower()
    if fantasy == "OUT" or s == "out":
        return 0.0
    for word in ("doubtful", "questionable", "probable"):
        if word in n or word in s:
            return STATUS_PLAY_PROB[word]
    if "won't play" in n or "ruled out" in n or "will not play" in n or "out for" in n:
        return 0.0
    return STATUS_PLAY_PROB.get(s, 0.5 if s else 1.0)


def injuries():
    """League injury report: one row per listed player."""
    js = _get(f"{SITE}/injuries")
    rows = []
    for t in js.get("injuries", []):
        for x in t.get("injuries", []):
            ath = x.get("athlete") or {}
            team = _code(((ath.get("team") or {}).get("abbreviation")) or t.get("displayName", ""))
            det = x.get("details") or {}
            status = (x.get("status") or "").strip()
            fant = ((det.get("fantasyStatus") or {}).get("abbreviation") or "").upper()
            rows.append({"team": team, "player": ath.get("displayName"), "status": status, "fantasy": fant,
                         "play_prob": play_prob(status, fant, x.get("shortComment") or ""),
                         "return_date": det.get("returnDate"), "updated": x.get("date"),
                         "note": x.get("shortComment")})
    return rows


def summary(espn_id: str):
    return _get(f"{SITE}/summary", {"event": espn_id})


def officials(js) -> list[str]:
    """Referee crew names from a game summary (empty until ESPN posts them)."""
    out = []
    for o in ((js.get("gameInfo") or {}).get("officials") or []):
        nm = o.get("displayName") or o.get("fullName")
        if nm:
            out.append(nm.strip())
    return out


def box_score(espn_id: str):
    """Player lines for one finished game, in the same shape as the historical box scores."""
    return box_rows(summary(espn_id))


def box_rows(js):
    rows = []
    for t in (js.get("boxscore") or {}).get("players", []):
        team = _code(t["team"]["abbreviation"])
        st = t["statistics"][0]
        keys = st["keys"]
        for a in st["athletes"]:
            if not (a.get("athlete") or {}).get("id"):
                continue
            s = dict(zip(keys, a.get("stats") or []))
            pair = lambda k: [(_num(v) or 0.0) for v in (s.get(k) or "0-0").split("-")[:2]]  # noqa: E731
            fgm, fga = pair("fieldGoalsMade-fieldGoalsAttempted")
            ftm, fta = pair("freeThrowsMade-freeThrowsAttempted")
            f = lambda k: _num(s.get(k)) or 0.0  # noqa: E731
            mins = f("minutes")
            gmsc = (f("points") + 0.4 * fgm - 0.7 * fga - 0.4 * (fta - ftm) + 0.7 * f("offensiveRebounds")
                    + 0.3 * f("defensiveRebounds") + f("steals") + 0.7 * f("assists") + 0.7 * f("blocks")
                    - 0.4 * f("fouls") - f("turnovers"))
            reason = (a.get("reason") or "").upper()
            played = mins > 0 and not a.get("didNotPlay")
            rows.append({"team": team, "espn_id": a["athlete"]["id"], "player": a["athlete"]["displayName"],
                         "played": played, "avail": 1.0 if played or "COACH" in reason else 0.0,
                         "min": mins, "gmsc": gmsc, "pm": f("plusMinus")})
    return rows
