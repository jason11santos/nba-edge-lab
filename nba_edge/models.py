"""Models and the self-adjusting ensemble.

Five forecasters, each refit on a schedule using only games that have already happened:

  market      the sportsbook moneyline with the vig removed (no fitting)
  stats_lr    logistic regression on ratings + schedule + lineup features (never sees the price)
  stats_gbm   gradient-boosted trees (LightGBM) on the same features
  mkt_lr      logistic regression that starts from the price and adds our features
  mkt_gbm     boosted trees that start from the market's log-odds and learn only corrections to it

The ensemble blends them with exponentially weighted log-loss ("Hedge"): after every game night each
model's recent loss is updated and weights shift toward whichever has been forecasting best lately.
A forgetting factor lets the weights move when the league changes. Recent games also count more when
the models are refit (sample weights with a half-life), so the fits themselves drift with the league.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .features import LINEUP, STATS, logit, sigmoid

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None
    from sklearn.ensemble import HistGradientBoostingClassifier

NAMES = ["market", "stats_lr", "stats_gbm", "mkt_lr", "mkt_gbm"]
MKT_EXTRA = ["spread_logit", "total"]
V1 = ["elo_logit", "b2b_home", "b2b_away", "form_diff"]


@dataclass
class Params:
    halflife_days: float = 1200.0     # recency weighting when refitting
    lr_c: float = 0.01
    offset_lam: float = 2000.0        # L2 on feature weights in the market-anchored regression
    gbm: dict = field(default_factory=lambda: dict(
        n_estimators=300, learning_rate=0.02, num_leaves=8, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0))
    resid_gbm: dict = field(default_factory=lambda: dict(
        n_estimators=150, learning_rate=0.01, num_leaves=4, min_child_samples=400,
        subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=20.0))
    hedge_eta: float = 2.0           # how hard the ensemble chases recent winners
    hedge_decay: float = 0.995       # per game night; ~140-night memory
    use: tuple = ("market", "stats_lr", "stats_gbm", "mkt_lr", "mkt_gbm")


def _prep_lin(X: pd.DataFrame, cols, mu=None, sd=None):
    A = X[cols].astype(float).copy()
    for c in LINEUP:
        if c in A and c != "has_lineup":
            A[c] = A[c].fillna(0.0)          # no lineup data = no adjustment; has_lineup flags it
    A = A.fillna(A.median() if mu is None else mu)
    if mu is None:
        mu, sd = A.mean(), A.std().replace(0, 1)
    return ((A - mu) / sd).to_numpy(), mu, sd


class OffsetLogit:
    """Logistic regression z = a + b*market_logit + x.beta, with an L2 penalty on beta only.
    The price is never shrunk, so the model starts at the market and only moves when the features
    have earned it on past games."""

    def __init__(self, lam=50.0):
        self.lam = lam

    def fit(self, m, A, y, w):
        from scipy.optimize import minimize
        n, k = A.shape
        Z = np.column_stack([np.ones(n), m, A])

        def f(th):
            z = Z @ th
            p = sigmoid(z)
            ll = -(w * (y * z - np.logaddexp(0, z))).sum()
            pen = 0.5 * self.lam * (th[2:] ** 2).sum()
            g = Z.T @ (w * (p - y))
            g[2:] += self.lam * th[2:]
            return ll + pen, g

        th0 = np.zeros(k + 2)
        th0[1] = 1.0
        self.theta = minimize(f, th0, jac=True, method="L-BFGS-B").x
        return self

    def predict(self, m, A):
        return sigmoid(self.theta[0] + self.theta[1] * m + A @ self.theta[2:])


class Forecasters:
    def __init__(self, prm: Params):
        self.p = prm
        self.fitted = {}

    def fit(self, T: pd.DataFrame, asof: pd.Timestamp):
        T = T[T["home_win"].notna() & T["mkt_logit"].notna()]
        y = T["home_win"].to_numpy()
        age = (asof - T["date"]).dt.days.to_numpy()
        w = 0.5 ** (age / self.p.halflife_days)
        w = w / w.mean()
        stats_cols = STATS
        mkt_cols = ["mkt_logit"] + MKT_EXTRA + STATS
        f = {}
        # stats_lr
        A, mu, sd = _prep_lin(T, stats_cols)
        m = LogisticRegression(C=self.p.lr_c, max_iter=2000).fit(A, y, sample_weight=w)
        f["stats_lr"] = (m, stats_cols, mu, sd)
        # v1: the October 2026 model (Elo + back-to-backs + form), kept as a yardstick
        A, mu1, sd1 = _prep_lin(T, V1)
        f["v1"] = (LogisticRegression(C=1.0, max_iter=1000).fit(A, y, sample_weight=w), V1, mu1, sd1)
        # mkt_lr : market log-odds enters unpenalised; our features get an L2 penalty
        A, mu2, sd2 = _prep_lin(T, mkt_cols[1:])
        m2 = OffsetLogit(self.p.offset_lam).fit(T["mkt_logit"].to_numpy(), A, y, w)
        f["mkt_lr"] = (m2, mkt_cols[1:], mu2, sd2)
        # GBMs
        f["stats_gbm"] = (self._gbm(T[stats_cols], y, w, self.p.gbm, None), stats_cols)
        init = T["mkt_logit"].to_numpy()
        f["mkt_gbm"] = (self._gbm(T[stats_cols + MKT_EXTRA], y, w, self.p.resid_gbm, init),
                        stats_cols + MKT_EXTRA)
        self.fitted = f
        return self

    @staticmethod
    def _gbm(X, y, w, prm, init):
        if lgb is not None:
            m = lgb.LGBMClassifier(objective="binary", verbose=-1, **prm)
            m.fit(X, y, sample_weight=w, init_score=init)
            return m
        return HistGradientBoostingClassifier(max_iter=prm["n_estimators"], learning_rate=prm["learning_rate"],
                                              max_leaf_nodes=prm["num_leaves"]).fit(X, y, sample_weight=w)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        out["market"] = X["p_market"]
        m, cols, mu, sd = self.fitted["v1"]
        out["v1"] = m.predict_proba(_prep_lin(X, cols, mu, sd)[0])[:, 1]
        m, cols, mu, sd = self.fitted["stats_lr"]
        out["stats_lr"] = m.predict_proba(_prep_lin(X, cols, mu, sd)[0])[:, 1]
        m, cols, mu, sd = self.fitted["mkt_lr"]
        out["mkt_lr"] = m.predict(X["mkt_logit"].fillna(0).to_numpy(), _prep_lin(X, cols, mu, sd)[0])
        m, cols = self.fitted["stats_gbm"]
        out["stats_gbm"] = m.predict_proba(X[cols])[:, 1]
        m, cols = self.fitted["mkt_gbm"]
        raw = m.predict(X[cols], raw_score=True) if lgb is not None else logit(m.predict_proba(X[cols])[:, 1])
        out["mkt_gbm"] = sigmoid(X["mkt_logit"].to_numpy() + raw)
        # Without a price the market-anchored models fall back to the stats models.
        nomkt = X["p_market"].isna()
        out.loc[nomkt, "mkt_lr"] = out.loc[nomkt, "stats_lr"]
        out.loc[nomkt, "mkt_gbm"] = out.loc[nomkt, "stats_gbm"]
        return out

    def importance(self):
        m, cols = self.fitted["stats_gbm"]
        if lgb is None:
            return {}
        gain = m.booster_.feature_importance("gain")
        return dict(sorted(zip(cols, (gain / gain.sum()).round(4).tolist()), key=lambda kv: -kv[1]))

    def coefs(self):
        """Market-anchored regression, in plain units: intercept, market slope, per-feature weights."""
        m, cols, mu, sd = self.fitted["mkt_lr"]
        th = m.theta
        return {"intercept": float(th[0]), "market": float(th[1]), "cols": list(cols),
                "coef": th[2:].round(5).tolist(), "mu": mu.round(5).tolist(), "sd": sd.round(5).tolist()}


class Hedge:
    """Exponentially weighted forecaster combination with forgetting."""

    def __init__(self, names, eta, decay):
        self.names, self.eta, self.decay = list(names), eta, decay
        self.loss = np.zeros(len(self.names))

    def weights(self):
        z = -self.eta * (self.loss - self.loss.min())
        w = np.exp(z)
        return w / w.sum()

    def combine(self, P: pd.DataFrame) -> np.ndarray:
        w = self.weights()
        M = P[self.names].to_numpy()
        avail = ~np.isnan(M)
        M = np.nan_to_num(M)
        ww = avail * w
        return (M * ww).sum(1) / ww.sum(1)

    def update(self, P: pd.DataFrame, y: np.ndarray):
        M = np.clip(P[self.names].to_numpy(), 1e-4, 1 - 1e-4)
        ll = -(y[:, None] * np.log(M) + (1 - y[:, None]) * np.log(1 - M))
        ll = np.where(np.isnan(ll), np.nanmean(ll, axis=1, keepdims=True), ll)
        self.loss = self.decay * self.loss + ll.sum(0)

    def state(self):
        return dict(zip(self.names, self.weights().round(4).tolist()))


def walk_forward(X: pd.DataFrame, test_from: str, prm: Params, refit_days: int = 14, train_from: str = "2008-08-01",
                 test_to: str | None = None, verbose=False):
    """Predict every game from `test_from` on, refitting every `refit_days` on everything before.
    Returns per-game predictions for each forecaster plus the ensemble, and the weight history."""
    X = X.sort_values(["date", "game_id"]).reset_index(drop=True)
    test_from = pd.Timestamp(test_from)
    D = X[X["date"] >= pd.Timestamp(train_from)]
    test = D[(D["date"] >= test_from) & (D["home_win"].notna())]
    if test_to:
        test = test[test["date"] <= pd.Timestamp(test_to)]
    fc = Forecasters(prm)
    hedge = Hedge(prm.use, prm.hedge_eta, prm.hedge_decay)
    preds, whist, last_fit, imp = [], [], None, []
    for date, day in test.groupby("date", sort=True):
        if last_fit is None or (date - last_fit).days >= refit_days:
            fc.fit(D[D["date"] < date], date)
            last_fit = date
            imp.append((str(date.date()), fc.importance()))
            if verbose:
                print("refit", date.date(), hedge.state())
        P = fc.predict(day)
        P["ensemble"] = hedge.combine(P)
        P["game_id"] = day["game_id"].to_numpy()
        preds.append(P)
        whist.append((str(date.date()), hedge.state()))
        hedge.update(P, day["home_win"].to_numpy())
    out = pd.concat(preds)
    return out, whist, imp, fc
