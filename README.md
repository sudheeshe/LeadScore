
# Lead Scoring using Machine Learning





### What Lead Score means...??

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/1_.jpg?raw=true)


- `Lead scoring:` is a popular methodology used by marketing and sales teams 
  to determine how likely their leads are to buy. It is a process where you assign a score (often 1-100) to your leads.

- The lead score tells you, your leads’ buying intention. The higher the score, the more likely they’ll buy.

- The simple need of a Lead scoring is to remove guesswork, so you’d spend time on leads who are the most likely to convert.


## Business Scenario

- The Bank is looking for an Effective Lead Scoring System which achieve the following.

        1. Lower marketing and acquisition costs
        2. Higher conversion rates with less time wasted
        3. Increase in sales and marketing team alignment: When you implement a method for scoring leads, 
           you’re ensuring that every lead passed onto sales team are qualified, boosting your conversion rate, and strengthening the relationship between the these two departments.
        4. Higher revenue

- Client is looking for an end result which a ML model, which can categorize given leads into `COLD LEADS`,`WARM LEADS`,`HOT LEADS`.


## Data understanding & EDA

- The data is related with direct marketing campaigns of a Portuguese banking institution. 
- The marketing campaigns were based on phone calls. 
- Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.
- In the dataset, we have customers’ personal information, their contact activity information, previous campaign information, and some social stats, and we also have information about which leads are converted and not converted yet.


###### 🔗 Data Description
[click here](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

Let's see the sample of data 

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/2_.png?raw=true)

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/3_.png?raw=true)

- The data have 20 columns including target column `y` after dropping `duration` columns
- There is 41188 records
- There is no missing values and 1528 duplicate values (3.7%) of total data

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/4_.png?raw=true)

- There are many High correlation alerts which we will see in detail later on.

Now lets see each column in detail

`Column - age`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/5_.png?raw=true)

- Mean age is `40 years`
- Min age shows `17 years` and Max is `98 years`
- There are some outliers towards higher end in column since `95 percentile is 58 years` but `max age is 98 years`
- From alerts `age have high correlation with job` column

Let's see distribution

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/6_.png?raw=true)

- its a little `Right Skewed` distribution

`Column - job`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/7_.png?raw=true)

- `type of job`
- `job` column shows 12 categories.
- From alerts `job have high correlation with age and education` columns

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/8_.png?raw=true)

- Let's see the frequencies of each category

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/9_.png?raw=true)


`Column - marital`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/10_.png?raw=true)

- `marital status `
- `marital` column shows 4 categories

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/11_.png?raw=true)

- category `married` and `single` are in majority


`Column - education`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/12_.png?raw=true)

- `education` column have 8 categories
- From alerts `education have high correlation with job` columns

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/13_.png?raw=true)


`Column - default`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/14_.png?raw=true)

- `has credit in default?`
- `default` column shows 3 categories 

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/15_.png?raw=true)

- 79.1 % are `non defaule customers` and 20.9% customers atatus are `unkown` and only 0.1% are `default`
- This column is of no use on prediction.

`Column - housing`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/16_.png?raw=true)

- `has housing loan?`
- `housing` column have 3 categories

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/17_.png?raw=true)

- `housing` have high correlation with `loan` colum

`Column - loan`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/18_.png?raw=true)

- `has personal loan`
- `loan` column have 3 categories

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/19_.png?raw=true)

- 82.4% customers not taken personal loan and 15.2% not have personal loan and 2.4% is unknown status
- `loan` have high correlation with `housing` column

`Column - contact`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/20_.png?raw=true)

- `contact communication type`
- `contact` column have 2 categories

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/21_.png?raw=true)

- majority are cell phone users 63.5% and 36.5% are telephone users
- `contact` have high correlation with `month, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, and nr.employed` columns

`Column - month`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/22_.png?raw=true)

- `last contact month of year`
- `month` have 10 categories, sine `Jan and Feb` are not available in dataset.

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/23_.png?raw=true)

- `month` have high correlation with `contact, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, and nr.employed` columns

`Column - day_of_week`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/24_.png?raw=true)

- `last contact day of the week`
- `day_of_week` have 5 categories, weekends are not founded in dataset

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/25_.png?raw=true)


`Column - campaign`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/26_.png?raw=true)

- `number of contacts performed during this campaign and for this client`
- `column` shows outliers towards higher end `95th percentile is 7 times` and `max value is 56 times`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/27_.png?raw=true)

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/28_.png?raw=true)

- from the above plot we can see mostly company got contacted only once and maximum number of time is 8.

`Column - pdays`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/29_.png?raw=true)

- `number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/30_.png?raw=true)

- `pdays` have 27 categories. in 96.3% customers was not contacted on previous marketing campaign.
- `pdays` can be removed from dataset


`Column - previous`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/31_.png?raw=true)

- `number of contacts performed before this campaign and for this client (numeric)`
- `previous` column shows 7 categories, Minimum shows 0 times and maximum shows 7 times
- `previous` shows high correlation with `pdays,poutcome and euribor3m` columns

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/32_.png?raw=true)

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/33_.png?raw=true)

- common value shows 0 and 1

`Column - poutcome`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/34_.png?raw=true)

- `outcome of the previous marketing campaign`
- shows `failure 10% , nonexistent 86.3%, and success is only 3.3%`.

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/35_.png?raw=true)

- `poutcome` shows high correlation with `pdays,previous, cons.price.idx, cons.conf.idx, euribor3m and nr.employed` columns


`Column - emp.var.rate`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/36_.png?raw=true)

- `employment variation rate/ Index- quarterly indicator (numeric)`
- employment variation rate/ Index tell the change in employment from last quarter to this quarter
- Like example if previous employment rate (measure of the proportion of a country's working age population that is employed. This includes people that have stopped looking for work) 
  was 6.1 and current is 6.8 means employment variation rate/ Index will be 0.7 (i.e. 6.8 - 6.1)
  Similarly if previous was 6.8 and current is 5.5 then employment variation rate/ Index will be -1.3.

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/37_.png?raw=true)

- let's see the common values

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/38_.png?raw=true)

- `emp.var.rate` shows high correlation with `contact,month, poutcome, cons.conf.idx, euribor3m and nr.employed` columns


`Column - cons.price.idx`


![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/39_.png?raw=true)

- `consumer price index - monthly indicator (numeric)` 

#### 🔗 what consumer price index [Click the link](https://www.investopedia.com/terms/c/consumerpriceindex.asp)



![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/40_.png?raw=true)

- Let's see common values

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/41_.png?raw=true)

- `cons.price.idx` shows high correlation with `contact,month, poutcome, emp.var.rate, euribor3m and nr.employed` columns

`Column - cons.conf.idx`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/42_.png?raw=true)

- `consumer confidence index - monthly indicator (numeric)`
- Consumer Confidence Index indicates that measures how optimistic or pessimistic consumers are regarding their expected financial situation. The CCI is based on the premise that if consumers are optimistic, they will spend more and stimulate the economy but if they are pessimistic then their spending patterns could lead to an economic slowdown or recession.

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/43_.png?raw=true)

- Let's see common values

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/44_.png?raw=true)


`Column - euribor3m`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/45_.png?raw=true)

- `euribor 3 month rate - daily indicator (numeric)`
- euribor rates is the basic rate of interest used in lending between banks on the European Union interbank market and also used as a reference for setting the interest rate on other loans.

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/46_.png?raw=true)

-Let's see common values

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/47_.png?raw=true)


`Column - nr.employed`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/48_.png?raw=true)

- `number of employees - quarterly indicator (numeric)`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/49_.png?raw=true)

- Let's see common values

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/50_.png?raw=true)


`Column - y`

![alt text](https://github.com/sudheeshe/Cross_Sell/blob/main/Images_for_readme/51_.png?raw=true)

- `the client subscribed a term deposit? (binary: 'True','False')`

- The target column is imbalanced


#### Let's see correlations using heatmap 

![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/52_.jpg?raw=true)


#### Let's Bi-variate analysis between Independent columns with Target column 

![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/53_.jpg?raw=true)

![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/54_.jpg?raw=true)

![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/55_.jpg?raw=true)

![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/56_.jpg?raw=true)


### Let's see outliers using boxplot

![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/57_.jpg?raw=true)



#### Observations are

- Customers of `age` between 30 and 40 years tend to convert more than other age categories
- Customers who work in services, admin, blue-collar, technician, retired, management and students categories have more chance of conversion.
- Customers who are married or single have more chance of conversion.
- Customers who've done high school, degree or professional course have more chance of conversion.
- Customers who doesn't have  personal loan have more chance of conversion.
- Customers who use cell phone have more chance of conversion than telephone.
- Chances of conversion is more during `April to August`.
- Customers who doesn't convert even after contacting 4 times during campaign. are very rare to convert even if we try to contact more number of times.
- From `pdays` column we can see Chances of conversion of customers who was not part of last campaign is high.
- From `previous` column we can see Customers who were never contacted before have higher conversion.
- From `poutcome` column we can see Customers who were not part of last campaign and Customers who converted during last campaign have higher conversion.
- From `cons.price.idx` column we can see Customers conversion was decreasing with increase in consumer price index (Inflation)
- Columns `cons.price.idx, cons.conf.idx, euribor3m, and nr.employed` doesn't shows any visible relationship with `y`.


## Feature Engineering

- Dropped `default`,`pdays`, and `duration` columns.
- Renaming column names `emp.var.rate to emp_var_rate`,`cons.price.idx to cons_price_idx`,`cons.conf.idx to cons_conf_idx`,`nr.employed to nr_employed`
- Dropped duplicate values from dataset (2008 records)
- Categories which is lass than 5% occurrence in `job`, `education`and `month` are combined to a single category named `other`
- Similarly, categories less than 5% in `campaign` column is combined to a class called `more_than_4`
- Similarly, categories less than 5% in `previous` column is combined to a class called `more_than_1`
- Applying  OrdinalEncoder on `education, campaign, previous` columns
- Applying OnehotEncoding on `job`, `marital`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`
- Applying LabelEncoding on target column `y`.
- Oversampling minority class using SMOTE
- Note: - VIF used for feature selection but model gave poor results.

![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/79_.png?raw=true)



## Model Building and Evaluation
- Used Logistic Regression, Decision Tree, SVC, Random Forest and XGBoost initially
- Decision tree was getting biased towards majority class
- SVC was taking more time for training, so due to lack of resource I've not used SVC.
- Logistic Regression gave better result compared to Random Forest and XGBoost classifiers.
- Results are below.

#### Logistic Regressor
![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/76_.jpg?raw=true)

#### Random Forest Classifier
![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/77_.jpg?raw=true)

#### XGBoost Classifier
![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/78_.jpg?raw=true)

## Final Results and Prediction

- I've used 3 classes 

         1. Hot Leads - threshold is >70
         2. Warm Leads - threshold is >=30 and <=70
         3. Cold Leads - threshold is <30


##### 🔗 Find the deployment link on Heroku

[click here](https://leadscoremodel.herokuapp.com/)

##### 🔗 Project explanation video link

[click here](https://youtu.be/JAvUlAkJtqM)


## Failed Experiements

- Tried multiple ways to solve this imbalanced classification probem.

- Feature engg steps 1 to step 9 were same
- Since `emp.var.rate`,`cons.price.idx`, `cons.conf.idx`, `euribor3m`,`nr.employed` columns are actually continuous values, but because of only few range of data present it shows up like categorical. 
   So I'm applying equal width binning on these column and converting into classes.
- For outlier handling I've tried with dropping outliers, mean & median has tried.
- Since we have Imbalanced data, while dropping the outliers we are loosing few records from minor category. So this method is not chosen.
- For Mean and median methods, I've first converted all the outliers into Null values and then tried Imputed these nulls are  with mean as well as median.  
  But results were not good. It changed the distribution of data.

![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/58_.jpg?raw=true)


![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/59_.jpg?raw=true)

- Tried with KNNImputer, which has given better result

![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/60_.jpg?raw=true)

- To handle skewness in `age` column, we have tried with `square_root, cube_root, squaring` the column. but didn't work well.

![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/61_.jpg?raw=true)

![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/62_.jpg?raw=true)

![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/63_.jpg?raw=true)

- `Log10 transformation` and `Yeo_Johnson transformation` gave good results. We can choose either one of them.

![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/64_.jpg?raw=true)

![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/65_.jpg?raw=true)



## Model training approaches

#### Method-1 (Applying Class Weights)

- Tried to apply class_weight parameter on LogisticRegressor, RandomForest, SVC, XGBoost classifiers
- Here we are giving more weightage to minority class, so that miss-classification on minority will be highly penalized.
- Results were poor on RandomForest and XGBoost
- Logistic Regression gave better result but recall was less 51 but in Final model we got 62.

![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/80_.jpg?raw=true)


#### Method-2 (Cluster based)

- The dataset shows possibility of 2 clusters other than target column classes when I've used KMeans.

![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/66_.jpg?raw=true)

- The prediction from KMeans is concatenated to the dataset and let's see the sample of data
- From sample data we are considering 3 records, In this `2 records have cluster = 1` and `3rd record have cluster = 0`.

![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/67_.jpg?raw=true)

![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/68_.jpg?raw=true)

- Now we need to concatenate y column also to create separate dataset for cluster 0 and cluster 1

![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/69_.jpg?raw=true)

- Let's create separate dataframes for based on cluster

![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/70_.jpg?raw=true)

- Let's see value_counts of target colum on each dataset

![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/71_.jpg?raw=true)

## Process Flow Diagram

![alt text](https://github.com/sudheeshe/LeadScore/blob/main/Images_for_readme/Process_flow.png?raw=true)

- Even this approach failed Random Forest gave recall of average 0.21 on two clusters, XGBoost gave avg. recall of 0.20, Logistic Regressor gave avg. recall of 0.15.

References:
### Precision-Recall curve blog 
[click here](https://medium.com/@douglaspsteen/precision-recall-curves-d32e5b290248#:~:text=In%20a%20perfect%20classifier%2C%20AUC,have%20AUC%2DPR%20%3D%200.5.)
