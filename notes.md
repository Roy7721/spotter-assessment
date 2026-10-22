# working notes

My thinking as it happened, in order. Unedited.

---

## first look at the files

- train_test is the given data to do experiment and train model
- have to decide how to make split. maybe data distribution shift

this are the columns of train_test.csv:

```
load_id        pickup        delivery       pickup_lat    pickup_lon
delivery_lat   delivery_lon  distance       equipment     weight
date           market_index  quote_signal   posted_rate
```

have to query what does market indix, quote singnal and posted rate means?

- mm validation csv is seems fine.
- mm validation template only contain truck id and predicted price. maybe i have to fill the column here
- mm december csv are lack of columns. mm something is not good here. let me check... there is no lat and long.. mm plus no track info.

i can pick the pickup lat- long and delivery lat lon from other given datas. mm but
just date is changing. no truck info is given. same starting and same distance.
mmmm .... maybe either i have to do feature engineering or i have to train model
without them. and maybe with them so that i can understand are those features are a
good contributor in price. if yes then i have to think what i can do.. meanwhile
validation csv got some value... maybe we take mean of each day and fill here? lets
see what happens next.

---

## lets inspect the data more carefully now

- [ ] train-test data... weight and market index has some null values.... shape is good. mm weight is negative.... let me check carefully..292 rows × 14 columns.. this much info containing weight as negative,,, mmm let me check if i given info about this.. no really. so i assume it is a data entry error.  no duplicate value found. data ranges from 1 january to 31 october.
- [ ] both validation dataset has same num of rows. validation dataset has some num values though.
- [ ] i will use a baseline first, then some advance ml algo to choice the winner.
- [ ] score.py for visualization.
- [ ] i have to maintain a git repo
- [ ] need to record a loom video

---

## data preparation

now lets think about data preparation........ first weight and market index has null
values. and some weights are negative. first work is to  make this weight positive,
thenn market index.... mm i cant use global mean.... it will be safe i use daily mean
cause so many deliveries done in one day. it will be safer since no instruction is
given,,, mmm,, date is thing that i can use..mm i can break the date into different
columns. So, Yeah,, data cleaning file is done i hope.

---

## the split, and checking for drift

i have train a baseline model first. but which metric to use to judge the model?
since its a regression problem i hope we should judge based on rmse, mae and related
metrics.  before that the data split is important. this is time series data and for
sure data distribution shift can create a problem. and train-test-split is for sure
not a good choice for time series data.

first, we will check for data_drift. whether data belongs to `Nov-Dec actually
different from Jan-Oct`? auc 0.5 is expected that indicates no data distribution
shift. data column needs to be exculded. and lets train a classifier. lets first
build the data and concate them and append 0 to train data and 1 to valid. Now lets
define an auc...  result is saying AUC (all features) = 0.8736.. that means data
distribution is shifting.

but are they really crossing the boundary of training data? lat and lon crossing
boundary but auc is not that bad. slightly higher than 0.5... then who is actually
causing the drift? let me send one by one column.... so market index is the factor.
but our model performance can be dropped if this range exceed the training data
range. for now we got that the drift is large but benign. now, Nov–Dec market values
are different from Jan–Oct — lower. But they stay inside the Jan–Oct range. 0 of
12,000 rows fall outside.

meanwhile, we are thinking to make an holdout dataset to check and evaluation and i
think sep to oct is a perfect fit cause the values or range remains same as nov-dec.
so we may have a good examined value to check and compare and make decision.

another important thing, that is, we got some new cities on the validation set which
lead to increase the percentage of data failing outside the training data. and and we
should use lat lon rather than city names. not because numbers beat strings - you can
encode strings fine - but because 8 of the validation cities NEVER appear in training.
a city-name encoding is built from the 64 training cities, so it has no slot for
chicago. lat/lon is just position, and chicago's position sits between cities the
model already knows. another benefit is we already know the range of market index
given in validation.csv that gives us hope that model may perform well.

---

## the baseline

ok now the baseline. first ran the split to check — fit 38477 (jan 01 to aug 31),
holdout 9523 (sep 01 to oct 31). dates are clean, no overlap. good.

now the simplest possible model. just distance. nothing else. why? because if i jump
straight to xgboost and get MAE 115 i have no idea if thats good or bad. i need a
number that says "this is what you get for free".

```
learned rule:  rate = 1.85 * distance + 268.90

RMSE  654.41
MAE   196.95
MAPE  9.87%
```

mm that is actually interesting. it is saying every load has a fixed cost of around
268 dollars no matter what, then 1.85 dollar for each mile. that makes sense in real
life — loading, paperwork, the driver showing up. so the price structure is roughly
linear in distance. whatever i build later is refining this, not replacing it.

mm 9.87% error from a model that knows literally one column. so distance is carrying
most of the signal.

> BUT something is strange here and i want to note it before i forget.
> RMSE is 654 and MAE is only 196. that is 3.3 times bigger.
> normally rmse and mae are close-ish. rmse punishes big errors much harder
> because it squares them. so a gap this wide means a few rows are VERY
> wrong, not that everything is a little wrong. but something is not right there.
> i dont know what it is yet. parking it. but i will come back to this for sure.

so 196.95 is the number to beat. everything from here gets measured against it.

---

## is the target skewed?

meanwhile i do suspect that target is right skewed from my experience with price data
but thats just me saying it. where is the evidence? skewness is a number so lets just
measure it.

```
skewness    1.902      0 = symmetric, positive = long right tail
kurtosis   13.283      heavy tails
mean      2373.98
median    2030.76      mean is sitting ABOVE median

q0.50     2030.76
q0.90     4365.82
q0.99     5972.83
max      25533.00      4.3x bigger than the 99th percentile !!
```

three different things all saying the same thing. skew is 1.9 which is strong. mean
above median is the classic signature , the long tail on the right drags the mean up
while the median stays put. and the quantiles show it best: half the loads are under
2030, the 99th percentile is 5972 which is fine, but the max is 25533. so the top
fraction of a percent stretches way way further than everything else.

then i took log and measured again:

```
log skewness  -0.491
log mean       7.573
log median     7.616     now basically identical
```

so the log actually fixed it. and this explains why the log target helped:

```
gbm raw target : RMSE 634.90  MAE 125.67  MAPE 5.78%
gbm log target : RMSE 636.89  MAE 118.57  MAPE 5.20%
```

MAE down from 197 (baseline) to 118.57. MAPE from 9.87% to 5.20%.

mm but RMSE. baseline 654.42, raw gbm 634.90, log gbm 636.89. it went UP. my MAE
improved 40% and my RMSE improved 3%. and the log transform which clearly helped made
RMSE slightly worse. something is going on with a small number of rows. time to stop
parking it and find it.

---

## finding the rows that cause the error

ok so i stopped parking it. if a few rows are doing the damage then i should be able
to COUNT them. so i sorted the squared errors and looked at how much the top rows
carry.

```
worst 10 rows   = 0.11% of rows but 43.1% of total squared error
worst 50 rows   = 0.53% of rows but 89.0%
worst 100 rows  = 1.05% of rows but 96.9%
worst 500 rows  = 5.25% of rows but 98.8%
```

mm. 50 rows out of 9523. 89 percent. so my rmse is not measuring my model at all, its
measuring 50 rows. and now the earlier thing makes sense -- my mae improved 44% and
rmse improved 3% because mae counts all 9523 rows equally and rmse is basically just
those 50. and the log transform making rmse slightly worse also makes sense now,
because log stops chasing extreme values which is exactly the right thing to do for
the other 99.5%.

so note for the report: rmse alone is a bad metric for this problem. i will still
report it but mae and mape are where the actual skill shows.

---

## what ARE those rows?

now the obvious question. what ARE those rows? i could guess but lets just print them
and look.

example - baltimore -> oklahoma city, 1760.3 miles, 29521 lbs, dry van, market index
1.005. nothing weird. totally ordinary load.

```
structural price (1.85*distance + 268.90)   $ 3,525
my model said                               $ 3,491
actual posted rate                          $20,132
```

mmmm. hold on. my model is within 1 percent of the structural price of freight. so
the model is NOT wrong. the row is wrong. the posted rate is 5.7x what that load is
actually worth. i was blaming my model this whole time.

there must be many rows like this. let me count the whole holdout.

```
144 of 9523 = 1.51% contaminated.
73 inflated,  median 3.36x
71 deflated,  median 0.30x
```

73 up and 71 down.

---

## can i see them?

before giving up on predicting them i want to SEE them. so i ran a pca on the
features and coloured the contaminated rows.

```
separation = 0.141 sd  (features only)
separation = 0.136 sd  (features + price)
```

basically zero. they are not a cluster. they sit right on top of the normal rows.

mm but this failure is actually useful. it tells me isolationforest, lof, dbscan -
none of them can ever work here. all of those look for points that are FAR from the
other points. these points are not far from anything. which is why residuals find
them and distance based detectors cannot. good, one whole family of methods ruled out
with evidence instead of me just trying them randomly.

---

## the trimming rule

i should drop them. but i need a rule. let me inspect the data again..

i cannot use standard deviation, because the contaminated rows ARE the giant errors,
so they would inflate the sd themselves and widen the exact band that is supposed to
exclude them. the thing i am trying to catch would be hiding itself. so i use median
absolute deviation instead:

result:

```
removed 628 of 38477 rows  (1.63%)
mae 118.66 -> 110.67
mape 5.20% -> 4.75%
```

so deleting 1.6% of my own training data made the model better.

final table:

```
baseline (distance only)   RMSE 654.42   MAE 196.95   MAPE 9.88%
gbm raw target             RMSE 634.90   MAE 125.67   MAPE 5.78%
gbm log target             RMSE 636.89   MAE 118.66   MAPE 5.20%
gbm log + trimmed          RMSE 634.96   MAE 110.67   MAPE 4.75%
```

---

## back to december

now back to the december problem i parked at the very beginning. december file has no
market_index and no quote_signal. i said earlier maybe take the daily mean from
validation.csv. so is that actually a good idea? lets check before doing it instead
of after.

i measured two things for each column. how much the DAYS differ from each other, and
how much the loads on the SAME day differ from each other.

```
                between-day sd    within-day sd
market_index        0.1665           0.0248
quote_signal        0.1294           0.2606
```

mm this is clean. market_index -- days differ by 0.167 but loads on the same day only
differ by 0.025. so almost all of it is between days. it is a property of the DAY.
makes sense, so a daily mean is an excellent estimate for it. my original guess was
right.

quote_signal -- the within-day number is BIGGER than the between-day number. 0.261 vs
0.129. so loads on the same day differ from each other MORE than the days differ from
each other. it is a property of the LOAD, not the day. so the daily mean would be a
bad estimate for quote_signal. i would be injecting more noise than there is signal.

so, two models.

```
model A -> the 12000 validation loads. they have the real quote_signal. uses
           everything.
model B -> the 31 december rows. trained without quote_signal at all, so it
           never looks for it. slightly less informed
```

and nothing in december is invented. coordinates come from a city->lat/lon lookup
built from training (coords are constant per city, and both december cities appear
there). market_index comes from the daily mean of validation.csv joined on date,
which covers all 31 december dates.

december output: $823.40 to $846.25. a 2.77% swing across the month.

```
mon 826.72  tue 833.63  wed 841.86  thu 843.91
fri 838.57  sat 831.52  sun 824.82
```

peaks thursday, bottoms sunday.

---

## sanity checks

but i dont want to just trust my own output. so a sanity check -- does this lane
really trade at that price? the same lane appears 32 times in training, range $757.93
to $973.75, median $848.61. my chart sits comfortably inside the range the lane
actually trades at. good. the level is right, not just the shape.

also checked the 12000 predictions against the training distribution:

```
         predicted    training
1%          333.32      327.17
25%        1266.69     1251.55
50%        2038.24     2030.76
75%        3380.50     3330.75
99%        5811.93     5972.83
```

everything lines up except the 99th which sits a bit below training. that is expected
and i think it is correct -- i am predicting the clean centre instead of chasing
contaminated spikes. score.py passes, 12000 and 31 validated.

---

## the final model

last thing. the jan-aug / sep-oct split existed only to MEASURE. now that the
measuring is done the final model gets refit on all 48000 labelled rows (jan-oct).
throwing away sep-oct from the final fit would be wasting labelled data for no
reason.

---

## if i had more time

- consensus trimming. right now one model's residual decides which row dies. i would
  rather make 3 or 4 different algorithms agree a row is unpredictable before
  deleting it. protects against deleting rows that one weak model simply failed on.
- quantify the floor. compute what rmse a PERFECT model would get if it priced every
  ordinary load exactly right and was only wrong on the 1.51% contaminated. that
  tells me how much of my remaining gap is even reducible.
- using optuna for more good hypermeter tunning
- prediction intervals instead of single numbers, so whoever uses this can see which
  quotes are confident and which are not.
- mlops pipeline that lead the project to production level
