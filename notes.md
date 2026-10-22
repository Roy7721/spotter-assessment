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
*
