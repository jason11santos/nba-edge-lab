"""Team reference data: one canonical code per franchise, plus every alias the data sources use.

Canonical codes follow the NBA's current tricodes. Relocated or renamed franchises map to the
current code so Elo and player history carry across (Seattle -> OKC, New Jersey -> Brooklyn,
New Orleans Hornets -> Pelicans, Charlotte Bobcats -> Hornets).
"""
from __future__ import annotations

import math

# code: (current full name, arena lat, arena lon, standard UTC offset, arena elevation in meters)
TEAMS: dict[str, tuple[str, float, float, int, int]] = {
    "ATL": ("Atlanta Hawks", 33.757, -84.396, -5, 320),
    "BOS": ("Boston Celtics", 42.366, -71.062, -5, 5),
    "BKN": ("Brooklyn Nets", 40.683, -73.976, -5, 10),
    "CHA": ("Charlotte Hornets", 35.225, -80.839, -5, 230),
    "CHI": ("Chicago Bulls", 41.881, -87.674, -6, 180),
    "CLE": ("Cleveland Cavaliers", 41.496, -81.688, -5, 200),
    "DAL": ("Dallas Mavericks", 32.790, -96.810, -6, 140),
    "DEN": ("Denver Nuggets", 39.749, -105.008, -7, 1609),
    "DET": ("Detroit Pistons", 42.341, -83.055, -5, 180),
    "GSW": ("Golden State Warriors", 37.768, -122.388, -8, 5),
    "HOU": ("Houston Rockets", 29.751, -95.362, -6, 15),
    "IND": ("Indiana Pacers", 39.764, -86.155, -5, 220),
    "LAC": ("LA Clippers", 33.945, -118.341, -8, 30),
    "LAL": ("Los Angeles Lakers", 34.043, -118.267, -8, 90),
    "MEM": ("Memphis Grizzlies", 35.138, -90.051, -6, 80),
    "MIA": ("Miami Heat", 25.781, -80.188, -5, 2),
    "MIL": ("Milwaukee Bucks", 43.045, -87.917, -6, 190),
    "MIN": ("Minnesota Timberwolves", 44.979, -93.276, -6, 260),
    "NOP": ("New Orleans Pelicans", 29.949, -90.082, -6, 2),
    "NYK": ("New York Knicks", 40.751, -73.993, -5, 10),
    "OKC": ("Oklahoma City Thunder", 35.463, -97.515, -6, 370),
    "ORL": ("Orlando Magic", 28.539, -81.384, -5, 30),
    "PHI": ("Philadelphia 76ers", 39.901, -75.172, -5, 10),
    "PHX": ("Phoenix Suns", 33.446, -112.071, -7, 330),
    "POR": ("Portland Trail Blazers", 45.532, -122.667, -8, 15),
    "SAC": ("Sacramento Kings", 38.580, -121.500, -8, 8),
    "SAS": ("San Antonio Spurs", 29.427, -98.438, -6, 200),
    "TOR": ("Toronto Raptors", 43.643, -79.379, -5, 80),
    "UTA": ("Utah Jazz", 40.768, -111.901, -7, 1288),
    "WAS": ("Washington Wizards", 38.898, -77.021, -5, 10),
}

# Every spelling seen in the odds data, NBA stats, ESPN and the schedule files.
_ALIASES = {
    # NBA tricodes for old franchises
    "NJN": "BKN", "NOH": "NOP", "NOK": "NOP", "SEA": "OKC", "CHH": "CHA", "BRK": "BKN", "PHO": "PHX",
    "GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS", "CHO": "CHA",
    # Full names, current and historical
    "Los Angeles Clippers": "LAC", "New Jersey Nets": "BKN", "Seattle SuperSonics": "OKC",
    "Charlotte Bobcats": "CHA", "New Orleans Hornets": "NOP", "New Orleans/Oklahoma City Hornets": "NOP",
    "LA Lakers": "LAL",
}


def code(name: str) -> str:
    """Return the canonical team code for any name, nickname or abbreviation the sources use."""
    n = str(name or "").strip()
    if n in TEAMS:
        return n
    if n in _ALIASES:
        return _ALIASES[n]
    for c, t in TEAMS.items():
        if t[0].lower() == n.lower():
            return c
    nick = n.split()[-1].lower() if n else ""
    for c, t in TEAMS.items():  # "Warriors", "Trail Blazers" etc.
        if t[0].lower().endswith(nick) and nick:
            return c
    raise KeyError(f"Unknown team: {name!r}")


def name(c: str) -> str:
    return TEAMS[c][0]


def travel_km(a: str, b: str) -> float:
    """Great-circle distance between two arenas."""
    if a == b:
        return 0.0
    _, la1, lo1, _, _ = TEAMS[a]
    _, la2, lo2, _, _ = TEAMS[b]
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = p2 - p1, math.radians(lo2 - lo1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def tz(c: str) -> int:
    return TEAMS[c][3]


def elevation(c: str) -> int:
    return TEAMS[c][4]


def season_of(date) -> str:
    """'2025-26' for any date from Aug 2025 through Jul 2026."""
    y, m = date.year, date.month
    start = y if m >= 9 else y - 1
    if y == 2020 and m in (9, 10):  # the 2019-20 bubble playoffs ran into October 2020
        start = 2019
    return f"{start}-{str(start + 1)[2:]}"
