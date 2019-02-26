import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import re
import datetime
from lightgbm import LGBMRegressor
import warnings

# set warnings
warnings.filterwarnings('ignore')

# Get data
df = pd.read_csv('sales_train_v2.csv', low_memory=False, parse_dates=['date'],
                 infer_datetime_format=True, dayfirst=True)  # Object to Date
test_set = pd.read_csv('test.csv', low_memory=False)
shops = pd.read_csv('shops.csv', low_memory=False)
item_cat = pd.read_csv('item_categories.csv', low_memory=False)
items = pd.read_csv('items.csv', low_memory=False)

# Monthly sales
df = df.groupby([df.date.apply(lambda x: x.strftime('%Y-%m')), 'item_id', 'shop_id']).sum().reset_index()
df = df[['date', 'item_id', 'shop_id', 'item_cnt_day']]
# Create pivot table
df = df.pivot_table(values='item_cnt_day', index=['item_id', 'shop_id'],
                    columns='date', fill_value=0).reset_index()

# Merge train set with pivot table
df = pd.merge(test_set, df, on=['item_id','shop_id'], how='left')
df = df.fillna(0)

# Drop categorical data
df.drop(['item_id', 'shop_id'], axis=1, inplace=True)

# Train/Test taking target final available month '2015-10'
target = '2015-10'
y_train = df[target]
X_train = df.drop(labels=[target], axis=1)
print('X_train: ', X_train.shape)

# Keras adjustment
X_train = X_train.as_matrix()
X_train = X_train.reshape((214200, 33, 1))

y_train = y_train.as_matrix()
y_train = y_train.reshape(214200, 1)

# Drop first month for LSTM memorization
X_test = df.drop(labels=['2013-01'], axis=1)
X_test = X_test.as_matrix()
X_test = X_test.reshape((214200, 33, 1))
print('X_train: ', X_train.shape)
print('X_test: ', X_test.shape)
print('y_train: ', y_train.shape)

# Model
m = LGBMRegressor(n_estimators=200,
                  learning_rate=0.03,
                  num_leaves=32,
                  colsample_bytree=0.9497036,
                  subsample=0.8715623,
                  max_depth=8,
                  reg_alpha=0.04,
                  reg_lambda=0.073,
                  min_split_gain=0.0222415,
                  min_child_weight=40)

m.fit(X_train, y_train)
y_pred = m.predict(X_test).clip(0., 20.)

# Submission
preds = pd.DataFrame(y_pred, column=['item_cnt_month'])
preds.to_csv('sub.csv', index_label='ID')
