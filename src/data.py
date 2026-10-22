'''
- weight has null values and negative
- market index has null values, we will use daily mean for safety
- data column can be used for creating new columns
- weight's null values can be filled by grouping the equipment. 
'''

from pathlib import Path

import pandas as pd

Root = Path(__file__).resolve().parents[1]

Data = Root/ 'data'

Files = {
    'train' :  Data / "train-test.csv",
    "valid" :  Data/ "validation.csv",
    "dec" :  Data / "december-chart-inputs.csv"
}

def clean(df):

    out = df.copy()
    # changing type of date column
    out['date'] = pd.to_datetime(out['date'])

    # taking abs values for weights
    if 'weight' in out.columns:
        out['weight'] = out["weight"].abs()

        # filling weight's null value using equipment median
        out['weight']= out['weight'].fillna(
            out.groupby("equipment")["weight"].transform('median')
        )

    # using daily data for filling the null value of market index
    if 'market_index' in out.columns:
        daily = out.groupby('date')['market_index'].transform('mean')
        out['market_index'] = out['market_index'].fillna(daily)

    return out

def load_all():
    train = clean(pd.read_csv(Files['train']))
    valid = clean(pd.read_csv(Files['valid']))
    dec = clean(pd.read_csv(Files['dec']))
    return train, valid, dec

if __name__ == "__main__":
    train, valid, dec = load_all()
    for name, df in [("train", train), ("valid", valid), ("dec", dec)]:
        print(f"{name:6} {df.shape}  nulls {int(df.isna().sum().sum())}")


