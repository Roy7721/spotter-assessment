"""Adversarial validation: is Nov-Dec actually different from Jan-Oct?

Train a classifier to tell training rows from validation rows.
  AUC ~ 0.50  -> the two periods are indistinguishable, no covariate shift
  AUC high    -> they differ, and the per-feature scores say which column moved

Date-derived columns are deliberately excluded. Including them would let the
classifier just read the calendar and score AUC 1.0, which answers nothing.
The question is whether the LOADS differ, not whether the dates differ.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from data import load_all,TRAIN_END, HOLDOUT_START

FEATURES = ["distance", "weight", "equipment", "pickup_lat", "pickup_lon",
            "delivery_lat", "delivery_lon", "market_index", "quote_signal"]


def build(train, valid):
    """Stack both periods into one table, labelled 0 = train, 1 = validation."""
    X = pd.concat([train[FEATURES], valid[FEATURES]], ignore_index=True)
    # converting equipment into codes
    X["equipment"] = X["equipment"].astype("category").cat.codes
    # creating zeros and ones
    y = np.r_[np.zeros(len(train)), np.ones(len(valid))]
    return X, y


def auc_of(X, y):
    """Out-of-fold AUC, so the score is never read off rows the model memorised."""
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06,
                                         random_state=0)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, prob)


def range_table(train, valid, dec):
    """Do Nov-Dec values fall INSIDE the range the model was trained on?

    Trees interpolate inside the training range and clamp outside it, so this
    is the check that matters -- not whether a feature moved, but whether it
    moved somewhere the model has never been.
    """
    print(f"\n{'feature':<14}{'train 1%':>10}{'train 99%':>11}"
          f"{'valid med':>11}{'outside':>9}{'dec med':>10}")
    print("-" * 65)
    for c in FEATURES:
        if c == "equipment":
            continue
        lo, hi = train[c].min(), train[c].max()
        out = ((valid[c] < lo) | (valid[c] > hi)).mean() * 100
        dmed = f"{dec[c].median():10.2f}" if c in dec.columns else f"{'-':>10}"
        print(f"{c:<14}{train[c].quantile(.01):10.2f}{train[c].quantile(.99):11.2f}"
              f"{valid[c].median():11.2f}{out:8.2f}%{dmed}")

def period_table(train, valid):
    """Market level in each period.

    The point: the holdout already sits in the same market as the rows we are
    scored on, so the error it reports can be trusted.
    """
    pieces = [
        ("FIT      (train-test)", train[train.date <= TRAIN_END]),
        ("HOLDOUT  (train-test)", train[train.date >= HOLDOUT_START]),
        ("validation  november", valid[valid.date.dt.month == 11]),
        ("validation  december", valid[valid.date.dt.month == 12]),
    ]
    print(f"\n{'piece':<24}{'rows':>7}{'from':>13}{'to':>13}"
          f"{'mi_min':>9}{'mi_med':>9}{'mi_max':>9}")
    print("-" * 84)
    for name, d in pieces:
        print(f"{name:<24}{len(d):7d}{str(d.date.min().date()):>13}"
              f"{str(d.date.max().date()):>13}{d.market_index.min():9.4f}"
              f"{d.market_index.median():9.4f}{d.market_index.max():9.4f}")
        


def main():
    train, valid, dec = load_all()
    X, y = build(train, valid)
    print(f"train rows {(y == 0).sum():,.0f}    validation rows {(y == 1).sum():,.0f}")

    auc = auc_of(X, y)
    print(f"\nadversarial validation AUC (all features) = {auc:.4f}")
    if auc < 0.55:
        print("  -> periods are essentially indistinguishable: no covariate shift")
    elif auc < 0.70:
        print("  -> mild shift, worth naming in the report")
    else:
        print("  -> strong shift, the per-feature scores below say which column")

    print("\nper-feature AUC (that feature alone, to localise any shift):")
    for c in X.columns:
        print(f"  {c:<16}{auc_of(X[[c]], y):.4f}")

    range_table(train, valid, dec)
    period_table(train=train, valid=valid)


if __name__ == "__main__":
    main()
