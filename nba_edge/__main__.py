"""Command line.

    python -m nba_edge setup        download the public historical data and build the tables   (~2 min)
    python -m nba_edge backtest     walk-forward backtest 2013-14 onward, writes web/data.json   (~5 min)
    python -m nba_edge backfill     results + closing lines from ESPN since the history ends
    python -m nba_edge retrain      refit every model on all games so far
    python -m nba_edge live --once  one live pass, writes web/live.json
    python -m nba_edge live         keep running (every 10 minutes; retrains nightly)
    python -m nba_edge serve        open the dashboard at http://localhost:8000
    python -m nba_edge cloud        one scheduled cloud run (GitHub Actions calls this every 5 minutes)
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")
KYLE = "https://raw.githubusercontent.com/kyleskom/NBA-Machine-Learning-Sports-Betting/master/Data/"
NOCT = "https://raw.githubusercontent.com/NocturneBear/NBA-Data-2010-2024/main/"
FILES = [(KYLE + "OddsData.sqlite", "OddsData.sqlite")] + [
    (NOCT + f, f) for f in ["regular_season_box_scores_2010_2024_part_1.csv",
                            "regular_season_box_scores_2010_2024_part_2.csv",
                            "regular_season_box_scores_2010_2024_part_3.csv",
                            "play_off_box_scores_2010_2024.csv"]]


def setup(force=False):
    import pandas as pd  # noqa: F401
    from . import history
    os.makedirs(RAW, exist_ok=True)
    for url, name in FILES:
        path = os.path.join(RAW, name)
        if force or not os.path.exists(path):
            print("downloading", name)
            urllib.request.urlretrieve(url, path)
    g = history.load_games(os.path.join(RAW, "OddsData.sqlite"))
    pg = history.load_player_games(RAW, g)
    g.to_parquet(os.path.join(DATA, "games.parquet"))
    pg.to_parquet(os.path.join(DATA, "player_games.parquet"))
    print(f"games: {len(g):,}  player lines: {len(pg):,}")


def backtest():
    import pandas as pd
    from . import backtest as B
    from . import features as F
    g = pd.read_parquet(os.path.join(DATA, "games.parquet"))
    pg = pd.read_parquet(os.path.join(DATA, "player_games.parquet"))
    print("building features…")
    X = F.build(g, pg)
    X.to_parquet(os.path.join(DATA, "features.parquet"))
    print("walk-forward backtest…")
    G, wh, imp, fc = B.run(X, verbose=True)
    G.to_parquet(os.path.join(DATA, "predictions.parquet"))
    met = B.metrics(G, wh, imp, fc, F.EloParams())
    os.makedirs(os.path.join(ROOT, "web"), exist_ok=True)
    B.export(G, met, os.path.join(ROOT, "web", "data.json"))
    s = met["summary"]
    print(f"Brier  market {s['market']['brier']:.4f}   ensemble {s['ensemble']['brier']:.4f}   "
          f"stats-only {s['stats_lr']['brier']:.4f}")


def serve(port=8000):
    import http.server
    import functools
    import webbrowser
    web = os.path.join(ROOT, "web")
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=web)
    print(f"dashboard: http://localhost:{port}   (Ctrl+C to stop)")
    webbrowser.open(f"http://localhost:{port}")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), h).serve_forever()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nba_edge", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup").add_argument("--force", action="store_true")
    sub.add_parser("backtest")
    b = sub.add_parser("backfill")
    b.add_argument("--start", default="2026-01-08")
    sub.add_parser("retrain")
    sub.add_parser("cloud", help="one scheduled cloud run (used by GitHub Actions)")
    lv = sub.add_parser("live")
    lv.add_argument("--once", action="store_true")
    lv.add_argument("--minutes", type=int, default=10)
    sv = sub.add_parser("serve")
    sv.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    if a.cmd == "setup":
        setup(a.force)
    elif a.cmd == "backtest":
        backtest()
    elif a.cmd == "serve":
        serve(a.port)
    else:
        from . import live
        st = live._store()
        if a.cmd == "backfill":
            live.backfill(st, a.start)
            live.sync_players(st)
            st.meta("backfilled", live.now())
        elif a.cmd == "retrain":
            live.retrain(st)
        elif a.cmd == "cloud":
            live.cloud_pass()
        elif a.cmd == "live":
            if a.once:
                if not st.meta("backfilled"):
                    live.backfill(st)
                    live.sync_players(st)
                    st.meta("backfilled", live.now())
                live.run_once(st)
            else:
                live.loop(a.minutes)


if __name__ == "__main__":
    sys.exit(main())
