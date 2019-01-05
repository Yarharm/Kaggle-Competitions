import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# Competition description:
#           Metric used => RMSLE (root mean squared log error) between actual and predicted auction prices
#           model => RandomForestRegressor


# get DataFrame
df = pd.read_csv('Train_small.csv', low_memory=False, parse_dates=["saledate"])

# adjust SalePrice for the RMSLE
df.SalePrice = np.log(df.SalePrice)

# convert saledate column to multiple useful date features
df['saledate'] = np.datetime_as_string(df['saledate'].values)
df['saledate'] = df['saledate'].str.slice(0, 10)
# extra features for the saledate
year = lambda x: datetime.strptime(x, "%Y-%m-%d").year
day_of_week = lambda x: datetime.strptime(x, "%Y-%m-%d").weekday()
month = lambda x: datetime.strptime(x, "%Y-%m-%d").month
week_number = lambda x: datetime.strptime(x, "%Y-%m-%d").strftime('%V')
# add extra features and drop original saledate
df['year'] = df['saledate'].map(year)
df['day_of_week'] = df['saledate'].map(day_of_week)
df['month'] = df['saledate'].map(month)
df['week_number'] = df['saledate'].map(week_number)
df['week_number'] = df['week_number'].astype('int64')

# drop columns full of NaN
df.drop(['saledate', 'Steering_Controls', 'Differential_Type', 'Hydraulics_Flow', 'Grouser_Tracks', 'Coupler_System', 'Coupler',
         'Ride_Control', 'Forks', 'Enclosure', 'ProductGroupDesc', 'ProductGroup', 'state', 'fiProductClassDesc', 'ProductSize'], axis=1, inplace=True)

# map 'UsageBand' ordinal categorical data
usage_band_mapping = {'Low' : 2,
                      'Medium' : 1,
                      'High' : 0}
df['UsageBand'] = df['UsageBand'].map(usage_band_mapping)

# drop rows and columns full of NaN


# add missing values (mean) [Interpolation with SimpleImputer?]
df = pd.get_dummies(df)
df.fillna(df.mean(), inplace=True)

# fit  model (Using all processors)
m = RandomForestRegressor(n_jobs=-1)
y = df.SalePrice
X = df.drop('SalePrice', axis=1)
m.fit(X, y)

# get R^2 of the prediction
print(m.score(X,y))

#print(df['saledate'].dtype)
#print(df.tail(2).transpose())