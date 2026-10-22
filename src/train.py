"""
- split the data
- train the baseline
- observe the metrics
"""

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data import load_all, TRAIN_END,HOLDOUT_START



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

if __name__ == "__main__":
    train, valid, dec = load_all()
    fit, holdout = time_split(train)
    print(f"fit     {len(fit):6}  {fit.date.min().date()} .. {fit.date.max().date()}")
    print(f"holdout {len(holdout):6}  {holdout.date.min().date()} .. {holdout.date.max().date()}")