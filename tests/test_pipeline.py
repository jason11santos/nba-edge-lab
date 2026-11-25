"""Tests. Fixtures are trimmed copies of real ESPN responses captured on 2026-11-24.

    python -m pytest -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nba_edge import features as F  # noqa: E402
from nba_edge.sources import espn  # noqa: E402

SCOREBOARD = {"events": [
    {"id": "401909314", "date": "2026-11-25T00:00Z", "season": {"type": 2},
     "competitions": [{"neutralSite": False, "status": {"type": {"state": "pre", "shortDetail": "11/24 - 7:00 PM EST"}},
                       "competitors": [{"homeAway": "home", "score": "0", "team": {"abbreviation": "ORL"}},
                                       {"homeAway": "away", "score": "0", "team": {"abbreviation": "DET"}}],
                       "odds": [{"provider": {"name": "Draft Kings"}, "details": "DET -1.5", "spread": 1.5,
                                 "overUnder": 212.5,
                                 "moneyline": {"home": {"open": {"odds": "+106"}, "close": {"odds": "+110"}},
                                               "away": {"open": {"odds": "-125"}, "close": {"odds": "-130"}}},
                                 "pointSpread": {"home": {"open": {"line": "+1.5"}, "close": {"line": "+1.5"}}},
                                 "total": {"over": {"open": {"line": "o214.5"}, "close": {"line": "o212.5"}}}}]}]},
    {"id": "401810831", "date": "2026-11-24T01:00Z", "season": {"type": 2},
     "competitions": [{"neutralSite": False, "status": {"type": {"state": "post", "shortDetail": "Final"}},
                       "competitors": [{"homeAway": "home", "score": "116", "team": {"abbreviation": "OKC"}},
                                       {"homeAway": "away", "score": "103", "team": {"abbreviation": "MIN"}}]}]},
    {"id": "1", "date": "2026-02-15T01:00Z", "season": {"type": 2},
     "competitions": [{"status": {"type": {"state": "post"}},
                       "competitors": [{"homeAway": "home", "score": "37", "team": {"abbreviation": "STARS"}},
                                       {"homeAway": "away", "score": "35", "team": {"abbreviation": "WORLD"}}]}]},
]}
CORE_ODDS = {"count": 1, "items": [{"provider": {"name": "Draft Kings"}, "spread": -8.5, "overUnder": 227.5,
                                    "homeTeamOdds": {"moneyLine": -380, "open": {"moneyLine": {"american": "-340"}},
                                                     "close": {"moneyLine": {"american": "-380"}}},
                                    "awayTeamOdds": {"moneyLine": 300, "open": {"moneyLine": {"american": "+270"}},
                                                     "close": {"moneyLine": {"american": "+300"}}}}]}
INJURIES = {"injuries": [{"displayName": "Atlanta Hawks", "injuries": [
    {"status": "Out", "date": "2026-11-21T22:12Z", "shortComment": "Landale (hip) won't play Sunday.",
     "athlete": {"displayName": "Jock Landale", "team": {"abbreviation": "ATL"}},
     "details": {"fantasyStatus": {"abbreviation": "GTD"}, "returnDate": "2026-11-27"}},
    {"status": "Day-To-Day", "date": "2026-11-22T22:12Z", "shortComment": "",
     "athlete": {"displayName": "Some Guard", "team": {"abbreviation": "ATL"}},
     "details": {"fantasyStatus": {"abbreviation": "GTD"}}}]}]}
SUMMARY = {"boxscore": {"players": [{"team": {"abbreviation": "MIN"}, "statistics": [{
    "keys": ["minutes", "points", "fieldGoalsMade-fieldGoalsAttempted", "threePointFieldGoalsMade-threePointFieldGoalsAttempted",
             "freeThrowsMade-freeThrowsAttempted", "rebounds", "assists", "turnovers", "steals", "blocks",
             "offensiveRebounds", "defensiveRebounds", "fouls", "plusMinus"],
    "athletes": [
        {"athlete": {"id": "3064514", "displayName": "Julius Randle"}, "didNotPlay": False, "reason": "COACH'S DECISION",
         "stats": ["35", "32", "11-18", "3-5", "7-8", "7", "6", "1", "0", "1", "2", "5", "4", "-5"]},
        {"athlete": {"id": "3195", "displayName": "Mike Conley"}, "didNotPlay": True, "reason": "COACH'S DECISION",
         "stats": []}]}]}]}}


@pytest.fixture
def fake_espn(monkeypatch):
    def get(url, params=None, tries=3):
        if url.endswith("/scoreboard"):
            return SCOREBOARD
        if "/odds" in url:
            return CORE_ODDS
        if url.endswith("/injuries"):
            return INJURIES
        if url.endswith("/summary"):
            return SUMMARY
        raise AssertionError(url)
    monkeypatch.setattr(espn, "_get", get)


def test_scoreboard_parsing(fake_espn):
    g = espn.scoreboard("20261124")
    assert len(g) == 2  # the All-Star exhibition is dropped
    orl = g[0]
    assert orl["date"] == "2026-11-24"           # 00:00 UTC is 7pm Eastern the day before
    assert orl["game_id"] == "20261124ORLDET"
    o = orl["odds"]
    assert (o["ml_home"], o["ml_away"], o["ml_home_open"], o["ml_away_open"]) == (110, -130, 106, -125)
    assert o["spread"] == -1.5                   # ESPN +1.5 home line = home is a 1.5-point underdog
    assert (o["total"], o["total_open"]) == (212.5, 214.5)
    assert g[1]["home_pts"] == 116 and g[1]["state"] == "post"


def test_closing_odds(fake_espn):
    o = espn.closing_odds("401810831")
    assert (o["ml_home"], o["ml_away"], o["ml_home_open"], o["ml_away_open"]) == (-380, 300, -340, 270)
    assert o["spread"] == 8.5


def test_injuries(fake_espn):
    r = espn.injuries()
    assert r[0]["play_prob"] == 0.0 and r[0]["team"] == "ATL"
    assert 0.4 < r[1]["play_prob"] < 0.8


def test_box_score_availability(fake_espn):
    rows = espn.box_score("x")
    randle, conley = rows
    assert randle["played"] and randle["avail"] == 1.0 and randle["min"] == 35
    assert round(randle["gmsc"], 1) == round(32 + 0.4 * 11 - 0.7 * 18 - 0.4 * 1 + 0.7 * 2 + 0.3 * 5 + 0 + 0.7 * 6 + 0.7 * 1 - 0.4 * 4 - 1, 1)
    assert not conley["played"] and conley["avail"] == 1.0   # healthy scratch: available, coach chose not to play


def _toy_games():
    d = pd.to_datetime(["2025-10-21", "2025-10-21", "2025-10-23", "2025-10-23"])
    return pd.DataFrame({"date": d, "home": ["BOS", "LAL", "BOS", "LAL"], "away": ["NYK", "GSW", "GSW", "NYK"],
                         "home_pts": [110, 100, 120, 99], "away_pts": [100, 105, 90, 101],
                         "ml_home": [-150, 120, -200, 110], "ml_away": [130, -140, 170, -130],
                         "spread": [3.5, 2.5, 5.5, 1.5], "total": [220, 225, 224, 221],
                         "season": "2025-26", "game_id": ["a", "b", "c", "d"]})


def test_pregame_features_do_not_see_the_result():
    g = _toy_games()
    e1 = F.elo_features(g)
    g2 = g.copy()
    g2.loc[2, ["home_pts", "away_pts"]] = [80, 130]   # flip game c's result
    e2 = F.elo_features(g2)
    assert e1.loc[2, "elo_home"] == e2.loc[2, "elo_home"]       # game c's own pre-game rating unchanged
    assert e1.loc[3].equals(e2.loc[3])                          # LAL/NYK never touched by game c
    h1, h2 = F.league_hca(g, window=2), F.league_hca(g2, window=2)
    assert h1.iloc[2] == h2.iloc[2]


def test_spread_sign_follows_moneyline_favorite():
    g = _toy_games()
    g.loc[1, "spread"] = 2.5   # unsigned in the old data; home is the underdog at +120
    m = F.market_features(g)
    assert m.loc[0, "spread"] > 0 and m.loc[1, "spread"] < 0
    assert abs(m.loc[0, "p_market"] - (0.6 / (0.6 + 100 / 230))) < 1e-9


def test_lineup_uses_only_earlier_games():
    g = _toy_games()
    rows = []
    for gid, t, opp in [("a", "BOS", "NYK"), ("c", "BOS", "GSW")]:
        for p, mn in [("star", 36), ("bench", 12)]:
            rows.append({"game_id": gid, "date": g.set_index("game_id").loc[gid, "date"], "team": t, "player_id": p,
                         "player": p, "played": True, "avail": 1.0, "min": mn,
                         "gmsc": 30.0 if p == "star" else 2.0, "pm": 5.0})
    pg = pd.DataFrame(rows)
    # star sits game c: availability 0, so BOS's lineup loss in game c must be positive
    pg.loc[(pg.game_id == "c") & (pg.player_id == "star"), ["played", "avail", "min", "gmsc", "pm"]] = [False, 0.0, 0, 0, 0]
    lf, st = F.lineup_features(g, pg, valid_from="2000-01-01")
    assert np.isnan(lf.loc[0, "loss_gmsc_home"]) or lf.loc[0, "loss_gmsc_home"] == 0
    assert lf.loc[2, "loss_gmsc_home"] > 0
    assert lf.loc[2, "miss_min_home"] == pytest.approx(36)


def test_totals_bets_and_push():
    from nba_edge import totals as TT
    pred = np.array([230.0, 210.0, 225.0, 220.0])
    line = np.array([224.0, 216.0, 224.0, 220.0])
    act = np.array([231.0, 205.0, 224.0, 240.0])
    side, pnl, _ = TT.ou_bets(pred, line, act, 3)
    assert list(side) == [1, -1, 0, 0]          # over, under, too close, no edge
    assert pnl[0] > 0 and pnl[1] > 0 and np.isnan(pnl[2])
    side, pnl, _ = TT.ou_bets(np.array([230.0]), np.array([224.0]), np.array([224.0]), 3)
    assert pnl[0] == 0.0                         # push returns the stake


def test_odds_api_budget():
    from nba_edge.sources import oddsapi
    from datetime import datetime, timedelta, timezone
    recent = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    assert oddsapi.due(recent) is False          # never twice inside 2 hours
    assert 7 * 2 * 31 <= 500                     # 7 calls a day x 2 credits fits the free plan


def test_referee_crew_features_use_only_earlier_games():
    import pandas as pd
    from nba_edge import refs
    g = pd.DataFrame({"game_id": ["a", "b", "c"], "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                      "home_pts": [130, 120, 100], "away_pts": [120, 110, 95], "tot_close": [220, 220, 220],
                      "spread_close": [0, 0, 0]})
    off = pd.DataFrame({"game_id": ["a", "b", "c"], "ref": ["X", "X", "Y"]})
    f, table = refs.crew_features(g, off)
    f = f.set_index("game_id")
    assert f.loc["a", "crew_tot"] == 0            # nothing known before his first game
    assert f.loc["b", "crew_tot"] == 30 / (1 + refs.K_TOT)   # one game, 30 over, shrunk hard
    assert f.loc["c", "crew_tot"] == 0            # a different official
    assert table.set_index("ref").loc["X", "games"] == 2


def test_nba_referee_page_parser(monkeypatch):
    import requests
    from nba_edge import summaries
    html = ('<table><tr><th>Game</th><th>Crew Chief</th></tr><tr><td>LA Clippers @ Memphis</td><td>Scott Foster (#48)'
            '</td><td>Ben Taylor (#46)</td><td>Jenna Reneau (#93)</td></tr></table>')
    monkeypatch.setattr(requests, "get", lambda *a, **k: type("R", (), {"text": html})())
    assert summaries._nba_page() == [({"LAC", "MEM"}, ["Scott Foster", "Ben Taylor", "Jenna Reneau"])]
