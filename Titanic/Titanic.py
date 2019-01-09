import pandas as pd
import numpy as np


df = pd.read_csv('train.csv', low_memory=False)

# Missing values columns => Age, Cabin, Embarked
#print(df.isna().sum())

# Encode Sex column (Nominal feature => OneHotEncoding)
convert_sex = pd.get_dummies(df['Sex'], drop_first=True)
df.drop(['Sex'], axis=1, inplace=True)
df = df.join(convert_sex)

# Embarked column
df['Embarked'] = df['Embarked'].fillna('C')
convert_embarked = pd.get_dummies(df['Embarked'], drop_first=True)
df.drop(['Embarked'], axis=1, inplace=True)
df = df.join(convert_embarked)
print(df.head(n=2).transpose())


# IMPUTE ALL MISSING VALUES, PROCESS STRINGS
# WHAT TO DO WITH NAMES?