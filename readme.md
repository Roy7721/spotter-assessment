# Freight Rate Prediction — Spotter ML Assessment

**Rana Roy** · [ranaroy4007@gmail.com](mailto:ranaroy4007@gmail.com)

Predicts `posted_rate` for 12,000 unlabelled freight loads, and produces the fixed
31-day December price chart.

Final holdout performance: **MAE $110.67, MAPE 4.75%**, against a distance-only
baseline of MAE $196.95.

**The submitted report is [`reports/REPORT.pdf`](reports/REPORT.pdf).** It is built
from [`reports/REPORT.md`](reports/REPORT.md) by `python reports/make_html.py`, which
produces a self-contained HTML file to print. The raw thought process, in the order
it happened, is in [`notes.md`](notes.md).

---

## Setup

The four provided CSVs are not committed to this repository. Place them in `data/`:

```
data/
  train-test.csv
  validation.csv
  december-chart-inputs.csv
  validation-predictions-template.csv
```

Then:

```bash
python -m pip install -r requirements.txt
```

Python 3.11+. Dependencies: pandas, numpy, scikit-learn, matplotlib.

## Running

Each script is independent and prints its own findings. Run them in this order:

```bash
python src/data.py          # loading and cleaning checks
python src/drift_check.py   # adversarial validation + feature range table
python src/train.py         # time split, baseline, model comparison, error analysis
python src/predict.py       # final models, writes both output files
```

`src/predict.py` writes `validation_predictions.csv` (12,000 rows) and
`december-predictions.csv` (31 rows).

Then validate both with the provided scorer:

```bash
python score.py --predictions validation_predictions.csv \
                --december-predictions december-predictions.csv
```

This writes `scorer_results/candidate_december.png`.

## Layout

| path | what it is |
|---|---|
| `src/data.py` | file paths, the shared `clean()` function, the train/holdout dates |
| `src/drift_check.py` | adversarial validation — is Nov–Dec a different distribution? |
| `src/train.py` | time split, baseline, model comparison, error concentration, trimming |
| `src/predict.py` | the two final models and both output files |
| `notebooks/tester.ipynb` | exploration and the evidence behind each claim in the report |
| `reports/REPORT.pdf` | **the submitted report** |
| `reports/REPORT.md` | the report source, and `make_html.py` builds it for printing |
| `notes.md` | working notes, unedited |

## Notes on the approach

- **Time-based split**, not a shuffled one — fit on Jan–Aug, measure on Sep–Oct,
  because the task is to predict the future from the past.
- **One `clean()` function** shared by all three files, so training and inference
  cannot drift apart.
- **Two models.** `december-chart-inputs.csv` has no `quote_signal`, and that column
  measures as a per-load value rather than a daily one, so it cannot be recovered
  from a daily mean. The December model is trained without it rather than being fed
  a fabricated value.
- **MAE and MAPE are the headline metrics.** On the holdout, 50 rows out of 9,523
  carry 89% of the squared error, so RMSE alone measures those 50 rows rather than
  the model. This is covered in detail in the report.
