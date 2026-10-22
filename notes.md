- train_test is the given data to do experiment and train model
- have to decide how to make split. maybe data distribution shift
- 

| load_id | pickup | delivery | pickup_lat | pickup_lon | delivery_lat | delivery_lon | distance | equipment | weight | date | market_index | quote_signal | posted_rate |
| ------- | ------ | -------- | ---------- | ---------- | ------------ | ------------ | -------- | --------- | ------ | ---- | ------------ | ------------ | ----------- |

this are the columns of train_test.csv. have to query what does market indix, quote singnal and posted rate means?

mm validation csv is seems fine.
mm validation template only contain truck id and predicted price. maybe i have to fill the column here
mm december csv are lack of columns. mm something is not good here. let me check... there is no lat and long.. mm plus no track info.
i can pick the pickup lat- long and delivery lat lon from other given datas. mm but just date is changing. no truck info is given. same starting and
same distance. mmmm .... maybe either i have to do feature engineering or i have to train model without them. and maybe with them so that i can understand are those features are a good contributor in price. if yes then i have to think what i can do.. meanwhile validation csv got some value... maybe we take mean of each day and fill here? lets see what happens next.


lets inspect the data more carefully now:
-----------------------------------------

- [ ] train-test data... weight and market index has some null values.... shape is good. mm weight is negative.... let me check carefully..292 rows × 14 columns.. this much info containing weight as negative,,, mmm let me check if i given info about this.. no really. so i assume it is a data entry error.  no duplicate value found. data ranges from 1 january to 31 october.
- [ ] both validation dataset has same num of rows. validation dataset has some num values though.
- [ ] i will use a baseline first, then some advance ml algo to choice the winner.
- [ ] score.py for visualization.
- [ ] i have to maintain a git repo
- [ ] need to record a loom video



* now lets think about data preparation........ first weight and market index has null values. and some weights are negative. first work is to  make this weight positive, thenn market index.... mm i cant use global mean.... it will be safe i use daily mean cause so many deliveries done in one day. it will be safer since no instruction is given,,, mmm,, date is thing that i can use..mm i can break the date into different columns. So, Yeah,, data cleaning file is done i hope.
* i have train a baseline model first. but which metric to use to judge the model? since its a regression problem i hope we should judge based on rmse, mae and related metrics.  before that the data split is important. this is time series data and for sure data distribution shift can create a problem. and train-test-split is for sure not a good choice for time series data.
  first, we will check for data_drift. whether data belongs to` Nov-Dec actually different from Jan-Oct`? auc 0.5 is expected that indicates no data distribution shift. data column needs to be exculded. and lets train a classifier. lets first build the data and concate them and append 0 to train data and 1 to valid. Now lets define an auc...  result is saying AUC (all features) = 0.8736.. that means data distribution is shifting. but are they really crossing the boundary of training data? lat and lon crossing boundary but auc is not that bad. slightly higher than 0.5... then who is actually causing the drift? let me send one by one column.... so market index is the factor. but our model performance can be dropped if this range exceed the training data range. for now we got that the drift is large but benign. now, Nov–Dec market values are different from Jan–Oct — lower. But they stay inside the Jan–Oct range. 0 of 12,000 rows fall outside. meanwhile, we are thinking to make an holdout dataset to check and evaluation and i think sep to oct is a perfect fit cause the values or range remains same as nov-dec. so we may have a good examined value to check and compare and make decision. another important thing, that is, we got some new cities on the validation set which lead to increase the percentage of data failing outside the training data. and and we should use lat lon rather than city names. not because numbers beat strings - you can encode strings fine - but because 8 of the validation cities NEVER appear in training. a city-name encoding is built from the 64 training cities, so it has no slot for chicago. lat/lon is just position, and chicago's position sits between cities the model already knows.another benefit is  we already know the range of market index given in validation.csv that gives us hope that model may perform well.
