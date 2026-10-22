"""
- split the data
- train the baseline
- observe the metrics
"""

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from data import load_all, TRAIN_END,HOLDOUT_START
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split


def metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE%": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }

def time_split(train):
    """Fit on Jan-Aug, test on Sep-Oct."""
    fit = train[train.date <= TRAIN_END]
    holdout = train[train.date >= HOLDOUT_START]
    return fit, holdout

def baseline(fit, holdout):
    """basic linear model"""
    m = LinearRegression().fit(fit[["distance"]], fit.posted_rate)
    pred = m.predict(holdout[["distance"]])
    print(f"\nlearned rule: rate = {m.coef_[0]:.2f} * distance + {m.intercept_:.2f}")
    return metrics(holdout.posted_rate.values, pred)


# fixed mapping, not cat.codes -- december contains only "Dry Van", so codes
# derived per-file would give it 0 while train derives its own order. a fixed
# map guarantees the same number means the same truck everywhere.
EQUIPMENT = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}

FEATURES = ["distance", "weight", "equipment",
            "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
            "market_index", "quote_signal",
            "dow", "month", "day_of_year", "week_of_year"]


def add_features(df):
    """Calendar features from the date, plus equipment as a stable code."""
    out = df.copy()
    out["dow"] = out.date.dt.dayofweek
    out["month"] = out.date.dt.month
    out["day_of_year"] = out.date.dt.dayofyear
    out["week_of_year"] = out.date.dt.isocalendar().week.astype(int)
    out["equipment"] = out.equipment.map(EQUIPMENT)
    return out


def gbm():
    return HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.06, min_samples_leaf=40,
        l2_regularization=1.0, random_state=0)

def error_profile(y_true, y_pred):
    """RMSE squares the errors, so a handful of bad rows can dominate it.
    This asks: how concentrated is the damage?"""
    sq = (y_true - y_pred) ** 2
    order = np.argsort(-sq)
    total = sq.sum()
    print(f"\n{'worst N rows':<16}{'% of rows':>12}{'% of squared error':>22}")
    for n in [10, 50, 100, 500]:
        share = 100 * sq[order[:n]].sum() / total
        print(f"{n:<16}{100 * n / len(y_true):11.2f}%{share:21.1f}%")

def split_comparison(train_f):
    """What would a shuffled split have claimed? Same model, same data,
    only the split changes."""
    a, b = train_test_split(train_f, test_size=0.2, random_state=0)  # shuffled
    m = gbm().fit(a[FEATURES], np.log(a.posted_rate))
    shuffled = metrics(b.posted_rate.values, np.exp(m.predict(b[FEATURES])))
    print(f"\nshuffled split  MAE {shuffled['MAE']:.2f}   <- dishonest")
    print(f"time split      MAE {118.66:.2f}   <- honest")


def worst_rows(holdout, y_pred, n=15):
    """Are the worst rows weird LOADS, or normal loads at weird PRICES?"""
    out = holdout.copy()
    out["predicted"] = y_pred
    out["ratio"] = out.posted_rate / out.predicted
    cols = ["pickup", "delivery", "distance", "weight", "equipment",
            "date", "market_index", "posted_rate", "predicted", "ratio"]
    worst = out.assign(e=(out.posted_rate - out.predicted).abs()) \
               .sort_values("e", ascending=False)
    print(worst[cols].head(n).to_string())


def trim(fit_f, features=FEATURES, n_mad=4.0):
    """Drop training rows the model cannot explain.

    Fit once, read the residuals, cut the extreme tail. MAD rather than SD
    because the contamination would inflate the SD and widen the very band
    meant to exclude it.
    """
    
    m = gbm().fit(fit_f[features], np.log(fit_f.posted_rate))
    resid = np.log(fit_f.posted_rate) - m.predict(fit_f[features])
    sigma = 1.4826 * (resid - resid.median()).abs().median()
    keep = resid.abs() <= n_mad * sigma
    print(f"\ntrim at {n_mad} MAD: dropping {(~keep).sum()} of {len(fit_f)} "
          f"({100 * (~keep).mean():.2f}%)")
    return fit_f[keep]






if __name__ == "__main__":
    train, valid, dec = load_all()
    fit, holdout = time_split(train)
    print(f"fit     {len(fit):6}  {fit.date.min().date()} .. {fit.date.max().date()}")
    print(f"holdout {len(holdout):6}  {holdout.date.min().date()} .. {holdout.date.max().date()}")
    print(baseline(fit, holdout))

    fit_f, hold_f = add_features(fit), add_features(holdout)
    y_hold = hold_f.posted_rate.values

    # (a) predict the price directly
    raw = gbm().fit(fit_f[FEATURES], fit_f.posted_rate).predict(hold_f[FEATURES])
    print("\ngbm, raw target :", metrics(y_hold, raw))

    # (b) predict log(price), then undo the log
    log = gbm().fit(fit_f[FEATURES], np.log(fit_f.posted_rate)).predict(hold_f[FEATURES])
    print("gbm, log target :", metrics(y_hold, np.exp(log)))

    error_profile(y_hold, np.exp(log))
    worst_rows(hold_f, np.exp(log))

    # finding out the contaminated rows
    out = hold_f.copy()
    out["predicted"] = np.exp(log)
    out["ratio"] = out.posted_rate / out.predicted
    print("\n--- most inflated ---")
    print(out.nlargest(5, "ratio")[["distance", "posted_rate", "predicted", "ratio"]].to_string())
    print("\n--- most deflated ---")
    print(out.nsmallest(5, "ratio")[["distance", "posted_rate", "predicted", "ratio"]].to_string())

    bad = (out.ratio > 1.5) | (out.ratio < 0.7)
    print(f"\ncontaminated: {bad.sum()} of {len(out)}  ({100*bad.mean():.2f}%)")

    # trimming
    fit_t = trim(fit_f)
    trimmed = gbm().fit(fit_t[FEATURES], np.log(fit_t.posted_rate)) \
                   .predict(hold_f[FEATURES])
    print("gbm, log target, trimmed :", metrics(y_hold, np.exp(trimmed)))
