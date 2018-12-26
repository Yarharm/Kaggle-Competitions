import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
# fastat library to work with date column conveniently

# Project:
#           Metric used => RMSLE (root mean squared log error) between actual and predicted auction prices
#           model => RandomForestRegressor


# get DataFrame
df = pd.read_csv('Train.csv', low_memory=False, parse_dates=["saledate"])

# adjust SalePrice for the RMSLE
df.SalePrice = np.log(df.SalePrice)
# ADJUST 'SALEDATE' COLUMN!!!!!!!!!!!!!!!

# map 'UsageBand' ordinal categorical data
usage_band_mapping = {'Low' : 2,
                      'Medium' : 1,
                      'High' : 0}
df['UsageBand'] = df['UsageBand'].map(usage_band_mapping)

# solve missing values problem!!!!!!!

# fir  model (Using all processors)
#m = RandomForestRegressor(n_jobs=-1)
#m.fit(df.drop('SalePrice', axis=1), df.SalePrice)
print(df.head(3).transpose())
#print(df.tail(2).transpose())