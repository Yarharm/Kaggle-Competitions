import pandas as pd
import numpy as np
import re


df = pd.read_csv('train.csv', low_memory=False)

# General exploration
def gen_exploration():

    # Get shape of data
    print(df.shape)

    # Get all column
    print(df.columns)

    # Get information(dtypes, mean, mode)
    print(df.info())
    print(df.describe())

def missing_data(df):
    print((df.isna().sum()[df.isna().sum() != 0] / df.shape[0] * 100).sort_values(ascending=False))


## MISSING VALUES
# 1) Movies that do not belong to collection => None
df['belongs_to_collection'] = df['belongs_to_collection'].fillna('None')
def collection_features(row, id=False):
    if row == 'None' and id:
        return 0
    elif row == 'None' and not id:
        return 'None'
    feature_lst = re.findall('(?<=:\s)(.*?),', row)
    if id:
        return int(feature_lst[0])
    else:
        return feature_lst[1].replace("'", "")

# Extract collection_id and collection_name
df['collection_id'] = df['belongs_to_collection'].map(lambda x: collection_features(x, id=True))
df['collection_name'] = df['belongs_to_collection'].map(lambda x: collection_features(x, id=False))

# 2) Homepage
df['homepage'] = df['homepage'].fillna('None')

# 3) Tagline
df['tagline'] = df['tagline'].fillna('None')

# 4) Keywords
df['Keywords'] = df['Keywords'].fillna('None')

# Calculate number of ids in each Keywords
# This will allow comparison based on the numbers of available keywords
df['Keywords_count'] = df['Keywords'].map(lambda x: x.count('{') if x != 'None' else 0)
print(sorted(df['Keywords_count'].unique()))
print(df.groupby('Keywords_count').id.count())
#missing_data(df)
# Explore Target





## FEATURE ENGINEERING

# 2 groups => Have collection_id and do not
# 2 groups => Have colletion_name and do not
# 2 groups => Have homepage and do not
# 2 groups => Have tagline and do not
# Group number of keyword in groups( No keys, Few, Medium, A lot)
