"""
- two model is required since 
-  december-chart-inputs.csv is missing columns that
validation.csv has:
  A   all features          -> the 12,000 validation loads
  B   without quote_signal  -> the 31 december rows

Both models fit on ALL labelled data (jan-oct). The jan-aug/sep-oct split in
train.py was for measuring; here we want every row we have.
"""

import numpy as np
import pandas as pd

from data import Root, load_all
from train import FEATURES, add_features, gbm, trim

FEATURES_DEC = [c for c in FEATURES if c != "quote_signal"]

 # have to inspect 
def city_coords(train):
    """city -> lat, lon. Coordinates are constant per city here, so either
    half of a trip gives the same answer."""
    a = train[["pickup", "pickup_lat", "pickup_lon"]].set_axis(
        ["city", "lat", "lon"], axis=1)
    b = train[["delivery", "delivery_lat", "delivery_lon"]].set_axis(
        ["city", "lat", "lon"], axis=1)
    return pd.concat([a, b]).groupby("city").first()


def fill_december(dec, train, valid):
    """Recover december's missing columns from data already provided.
    Nothing here is invented."""
    out = dec.copy()
    coords = city_coords(train)
    for side in ("pickup", "delivery"):
        out[f"{side}_lat"] = out[side].map(coords.lat)
        out[f"{side}_lon"] = out[side].map(coords.lon)

    out["market_index"] = out.date.map(valid.groupby("date").market_index.mean())

    gaps = out[["pickup_lat", "delivery_lat", "market_index"]].isna().sum().sum()
    assert gaps == 0, f"{gaps} december values could not be recovered"
    return out


def main():
    train, valid, dec = load_all()
    train_f = add_features(train)

    # ---- model A: the 12,000 validation loads ----
    fit_a = trim(train_f, FEATURES)
    model_a = gbm().fit(fit_a[FEATURES], np.log(fit_a.posted_rate))
    pred_a = np.exp(model_a.predict(add_features(valid)[FEATURES]))

    path_a = Root / "validation-predictions.csv"
    pd.DataFrame({"load_id": valid.load_id,
                  "predicted_rate": pred_a.round(2)}).to_csv(path_a, index=False)
    print(f"wrote {path_a.name}  {len(pred_a)} rows  "
          f"${pred_a.min():.0f} - ${pred_a.max():.0f}")

    # ---- model B: the december chart ----
    fit_b = trim(train_f, FEATURES_DEC)
    model_b = gbm().fit(fit_b[FEATURES_DEC], np.log(fit_b.posted_rate))
    dec_full = add_features(fill_december(dec, train, valid))
    pred_b = np.exp(model_b.predict(dec_full[FEATURES_DEC]))

    out_b = dec[["pickup", "delivery", "distance", "equipment", "weight", "date"]].copy()
    out_b["date"] = out_b.date.dt.strftime("%Y-%m-%d")
    out_b["predicted_rate"] = pred_b.round(2)
    path_b = Root / "december-predictions.csv"
    out_b.to_csv(path_b, index=False)
    print(f"wrote {path_b.name}  {len(out_b)} rows  "
          f"${pred_b.min():.2f} - ${pred_b.max():.2f}  "
          f"swing {100 * (pred_b.max() / pred_b.min() - 1):.2f}%")


if __name__ == "__main__":
    main()