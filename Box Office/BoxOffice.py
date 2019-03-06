import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import re
import datetime
from collections import Counter
import eli5
import shap
train = pd.read_csv('train.csv', low_memory=False)
test = pd.read_csv('test.csv', low_memory=False)

# Target
y_train = train['revenue']

# Combine train/test
train.drop('revenue', axis=1, inplace=True)
df = pd.concat((train, test), sort=False).reset_index(drop=True)


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

# 2) Homepage
df['homepage'] = df['homepage'].fillna('None')

# 3) Tagline
df['tagline'] = df['tagline'].fillna('None')

# 4) Keywords
df['Keywords'] = df['Keywords'].fillna('None')

# Calculate number of ids in each Keywords
# This will allow comparison based on the numbers of available keywords
df['Keywords_count'] = df['Keywords'].map(lambda x: x.count('{') if x != 'None' else 0)
# print(sorted(df['Keywords_count'].unique()))
# print(df.groupby('Keywords_count').id.count())

# 5) Production companies
df['production_companies'] = df['production_companies'].fillna('None')
df['production_companies_count'] = df['production_companies'].map(lambda x: x.count('{') if x != 'None' else 0)
#print(sorted(df['production_companies_count'].unique()))
#print(df['production_companies_count'].value_counts())

# 6) Production countries
df['production_countries'] = df['production_countries'].fillna('None')

df['production_countries_count'] = df['production_countries'].map(lambda x: x.count('{') if x != 'None' else 0)
# print(df['production_countries'].unique())

# 7) Spoken_languages
df['spoken_languages'] = df['spoken_languages'].fillna('None')
df['spoken_languages_count'] = df['spoken_languages'].map(lambda x: x.count('{') if x != 'None' else 0)

# 8) crew (For some reason there are three genders[0, 1, 2])
df['crew'] = df['crew'].fillna('None')
df['crew_count'] = df['crew'].map(lambda x: x.count('{') if x != 'None' else 0)


# Extract most dominant crew gender type
def crew_gender(row):
    if row == 'None':
        return 'None'
    genders = re.findall(r'(?<=\s)[0-9]\b', row)
    genders = Counter(genders)
    return genders.most_common(1)[0][0]


df['dominant_crew_gender'] = df['crew'].map(lambda x: crew_gender(x))

# 9) Cast
df['cast'] = df['cast'].fillna('None')
df['cast_count'] = df['cast'].map(lambda x: x.count('{') if x != 'None' else 0)


# Extract gender and order from cast
def cast_gender(row):
    if row == 'None' or len(row) == 2:
        return 'None'
    genders = re.findall(r"'gender': [0-9]\b", row)
    genders = [x[-1] for x in genders]
    genders = Counter(genders)
    return genders.most_common(1)[0][0]


df['dominant_cast_gender'] = df['cast'].map(lambda x: cast_gender(x))

# 10) Overview
df['overview'] = df['overview'].fillna('None')

# 11) genres
df['genres'] = df['genres'].fillna('None')
df['genres_count'] = df['genres'].map(lambda x: x.count('{') if x != 'None' else 0)

# 12) runtime
df['runtime'] = df['runtime'].fillna(0)

# 13) Missing data from test set
df['title'] = df['title'].fillna('None')
df['status'] = df['status'].fillna('None')

# Explore Target
# print(y_train)
y_train = np.log1p(y_train)
# sns.distplot(y_train)
# plt.show()


## EDA
# Pair plot for Discrete variables
# sns.pairplot(df[['budget', 'runtime', 'popularity', 'revenue']])
# plt.show()


## FEATURE ENGINEERING

# 2 groups => Have homepage and do not
df['has_homepage'] = df['homepage'].map(lambda x: 0 if x == 'None' else 1)
df['has_homepage'] = df['has_homepage'].apply(str)

# 2 groups => Have tagline and do not
df['has_tagline'] = df['tagline'].map(lambda x: 0 if x == 'None' else 1)
df['has_tagline'] = df['has_tagline'].apply(str)

# 2 groups => Belong to collection and do not
df['part_of_collection'] = df['collection_id'].map(lambda x: 0 if x == 0 else 1)
df['part_of_collection'] = df['part_of_collection'].apply(str)


# Date: Day, Month, Year, isWeekend, Holiday, Number of movies release at the same time
df[['release_month', 'release_day', 'release_year']] = df['release_date'].str.split('/', expand=True).replace(np.nan,
                                                                                                              '0').astype(
    str)
releaseDate = pd.to_datetime(df['release_date'])
df['release_dayofweek'] = releaseDate.dt.dayofweek.apply(str)
df['release_quarter'] = releaseDate.dt.quarter.apply(str)

# Split by 'en' and the rest
df['original_language'] = df['original_language'].map(lambda x: 1 if x == 'en' else 0)
df['original_language'] = df['original_language'].apply(str)

# Normilize budget
df['budget'] = np.log1p(df['budget'])

# Label encoding
labels = ['dominant_crew_gender', 'dominant_cast_gender',
          'has_homepage', 'has_tagline', 'part_of_collection',
          'release_month', 'release_day', 'release_year',
          'release_dayofweek', 'release_quarter', 'original_language']
labels_encoded = pd.get_dummies(df[labels])

## Additional ratio features
# Mean runtime per year
df['meanruntimeByYear'] = df.groupby("release_year")["runtime"].aggregate('mean')


# Drop columns
drop_columns = ['poster_path', 'id', 'belongs_to_collection',
                'genres', 'homepage', 'imdb_id', 'overview',
                'production_companies', 'production_countries',
                'tagline', 'title', 'Keywords', 'cast', 'crew',
                'collection_id', 'homepage', 'tagline', 'spoken_languages',
                'popularity', 'status', 'original_title', 'release_date']

df.drop(drop_columns, axis=1, inplace=True)
df.drop(labels, axis=1, inplace=True)  # Drop encoded categorical data

df = pd.concat([df, labels_encoded], axis=1)  # Append encoded labels



## Modeling
print(df.shape)
print(y_train.shape)
X_train = df.iloc[:y_train.shape[0], :]
X_test = df.iloc[y_train.shape[0]:, :]
print(X_train.shape)
print(X_test.shape)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1)

params = {'num_leaves': 30,
         'min_data_in_leaf': 20,
         'objective': 'regression',
         'max_depth': 5,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.2,
         "verbosity": -1}
m = lgb.LGBMRegressor(**params, n_estimators=20000, n_jobs=-1)
m.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',
        verbose=1000, early_stopping_rounds=200)


# ELI5 and SHAP analysis(Kaggle tutorial)
eli5.show_weights(model1, feature_filter=lambda x: x != '<BIAS>')

explainer = shap.TreeExplainer(model1, X_train)
shap_values = explainer.shap_values(X_train)

shap.summary_plot(shap_values, X_train)

top_cols = X_train.columns[np.argsort(shap_values.std(0))[::-1]][:10]
for col in top_cols:
    shap.dependence_plot(col, shap_values, X_train)