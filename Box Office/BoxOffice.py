import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

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
#print(sorted(df['Keywords_count'].unique()))
#print(df.groupby('Keywords_count').id.count())

# 5) Production companies
df['production_companies'] = df['production_companies'].fillna('None')
df['production_companies_count'] = df['production_companies'].map(lambda x: x.count('{') if x != 'None' else 0)
#print(sorted(df['production_companies_count'].unique()))

# 6) Production countries
df['production_countries'] = df['production_countries'].fillna('None')

df['production_countries_count'] = df['production_countries'].map(lambda x: x.count('{') if x != 'None' else 0)
#print(df['production_countries'].unique())

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

# Explore Target
#print(df['revenue'])
#sns.distplot(df['revenue'])
#plt.show()

## EDA
# Pair plot for Discrete variables


## FEATURE ENGINEERING

# 2 groups => Have homepage and do not
df['has_homepage'] = df['homepage'].map(lambda x: 0 if x == 'None' else 1)
df['has_homepage'] = df['has_homepage'].apply(str)

# 2 groups => Have tagline and do not
df['has_tagline'] = df['tagline'].map(lambda x: 0 if x == 'None' else 1)
df['has_tagline'] = df['has_tagline'].apply(str)


# 2 groups => Have collection_id and do not
# 2 groups => Have colletion_name and do not

# Group number of keyword in groups( No keys, Few, Medium, A lot)
# Group by number of production companies that produced a movie
# Group by number of production countries
# Group with and without crew
# Group with and without cast
# Group with and without overview
# Group small and big amount of genres
# Group by zero, small, medium, long runtime
# Group by local movies(Enlish original title) and foreign(Other languages
# Group by number of spoken languages(More languages covered => Bigger audience)
# Group by Sequels/Prequels and Initial release
# Date: Day, Month, Year, Weekend, Holiday, Number of movies release at the same time
# Come up with something for original languages
# Come up with something for popularity (Continuous variable)
# Come up with status (Released, Rumored)



# Useless columns
drop_columns = ['poster_path', 'id', 'belongs_to_collection',
                'genres', 'homepage', 'imdb_id', 'overview',
                'production_companies', 'production_countries',
                'tagline', 'title', 'Keywords', 'cast', 'crew']
df.drop(drop_columns, axis=1, inplace=True)
#print(df.columns)
