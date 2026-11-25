"""Scoring forecasts and simulating bets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import decimal_odds, implied, logit


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def log_loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def summary(p, y):
    return {"n": int(len(y)), "brier": brier(p, y), "log_loss": log_loss(p, y),
            "accuracy": float(np.mean((p > 0.5) == (y == 1)))}


def calibration(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 30:
            out.append({"pred": float(p[m].mean()), "actual": float(y[m].mean()), "n": int(m.sum())})
    return out


def bootstrap_diff(a, b, y, nights, n=2000, seed=0):
    """Brier(a) - Brier(b) with a 95% interval that resamples whole game nights (games on the
    same night share news, so they are not independent)."""
    rng = np.random.default_rng(seed)
    d = (a - y) ** 2 - (b - y) ** 2
    df = pd.DataFrame({"d": d, "night": nights}).groupby("night")["d"].agg(["sum", "count"])
    s, c = df["sum"].to_numpy(), df["count"].to_numpy()
    k = len(df)
    draws = rng.integers(0, k, size=(n, k))
    stats = s[draws].sum(1) / c[draws].sum(1)
    return {"mean": float(d.mean()), "ci_low": float(np.percentile(stats, 2.5)),
            "ci_high": float(np.percentile(stats, 97.5))}


def encompassing(p_model, p_market, y):
    """Does the model know anything the price doesn't? Regress the result on the market's log-odds
    and the model's disagreement with it. A significant disagreement coefficient = real information."""
    from math import erfc, sqrt
    lm = logit(p_market)
    X = np.column_stack([np.ones(len(y)), lm, logit(p_model) - lm])
    b = np.array([0.0, 1.0, 0.0])
    for _ in range(50):  # Newton-Raphson for the logistic MLE (no statsmodels: some Windows setups block its DLLs)
        p = 1 / (1 + np.exp(-(X @ b)))
        H = (X * (p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(H, X.T @ (y - p))
        b = b + step
        if np.abs(step).max() < 1e-10:
            break
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    names = ["intercept", "logit_market", "model_minus_market"]
    return [{"name": n, "coef": float(c), "se": float(s), "p_value": float(erfc(abs(c / s) / sqrt(2)))}
            for n, c, s in zip(names, b, se)]


def kelly_fraction(p, dec):
    b = dec - 1
    return np.maximum((p * b - (1 - p)) / b, 0)


def bets(p_home, ml_home, ml_away, y, min_edge=0.02, kelly_mult=0.25, cap=0.02):
    """Pick the side with the higher expected value; bet if EV clears min_edge."""
    dh, da = decimal_odds(ml_home), decimal_odds(ml_away)
    evh, eva = p_home * dh - 1, (1 - p_home) * da - 1
    home = evh >= eva
    ev = np.where(home, evh, eva)
    pp = np.where(home, p_home, 1 - p_home)
    d = np.where(home, dh, da)
    won = np.where(home, y == 1, y == 0)
    take = ev > min_edge
    stake = np.minimum(kelly_mult * kelly_fraction(pp, d), cap)
    flat = np.where(won, d - 1, -1.0)
    return pd.DataFrame({"take": take, "home": home, "ev": ev, "dec": d, "won": won, "stake": stake,
                         "flat": flat})


def betting_summary(b: pd.DataFrame, nights, n_boot=1000, seed=0):
    t = b[b["take"]]
    if not len(t):
        return {"n_bets": 0}
    roi = float(t["flat"].mean())
    rng = np.random.default_rng(seed)
    g = pd.DataFrame({"f": t["flat"].to_numpy(), "night": np.asarray(nights)[b["take"].to_numpy()]})
    agg = g.groupby("night")["f"].agg(["sum", "count"])
    s, c = agg["sum"].to_numpy(), agg["count"].to_numpy()
    draws = rng.integers(0, len(agg), size=(n_boot, len(agg)))
    boot = s[draws].sum(1) / c[draws].sum(1)
    return {"n_bets": int(len(t)), "win_rate": float(t["won"].mean()), "flat_roi": roi,
            "flat_roi_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "avg_ev_claimed": float(t["ev"].mean())}


def no_vig(ml_home, ml_away):
    ih, ia = implied(ml_home), implied(ml_away)
    return ih / (ih + ia), ih + ia
