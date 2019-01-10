import pandas as pd
import numpy as np
import re


df = pd.read_csv('train.csv', low_memory=False)

# Missing values columns => Age, Cabin, Embarked
#print(df.isna().sum())

# Encode Sex column (Nominal feature => OneHotEncoding)
#convert_sex = pd.get_dummies(df['Sex'], drop_first=True)
#df.drop(['Sex'], axis=1, inplace=True)
#df = df.join(convert_sex)

# Embarked column
#df['Embarked'] = df['Embarked'].fillna('C')
#convert_embarked = pd.get_dummies(df['Embarked'], drop_first=True)
#df.drop(['Embarked'], axis=1, inplace=True)
#df = df.join(convert_embarked)

# Title is the only relevant info in Name column
#def find_titles(title):
#    return re.search('\s.*\.', title).group()[1:]
#df['Name'] = df['Name'].apply(find_titles)

# Convert cabin
def cabin_to_deck(cab):
    """ Convert cabinet to the floor where Cabinet is located.
    The idea is that the floor should affect chances of survival
    Passengers without the Cabinet are in new group U => Unknown
    :param cab: string
        Cabinet with the floor and number
    :return: string
        Chop number part of the cabinet and return only it's floor
    """
    return re.search('^.{1}', cab).group()

df['Cabin'].fillna("Unknown", inplace=True)
df['Cabin'] = df['Cabin'].apply(cabin_to_deck)
# Floor order
deck = {'A': 0, 'B': 1, 'C': 2,
        'D': 3, 'E': 4, 'F': 5,
        'G': 6, 'T': 7, 'U': 8}
df['Cabin'] = df['Cabin'].map(deck)
#print(df[['Cabin', 'Survived']].groupby(['Cabin']).mean())
print(df.head(n=2).transpose())

# IMPUTE ALL MISSING VALUES, PROCESS STRINGS