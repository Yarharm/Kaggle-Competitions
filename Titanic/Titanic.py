from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import re


train = pd.read_csv('train.csv', low_memory=False)
y_train = train['Survived']
df = pd.read_csv('data.csv', low_memory=False)
# Encode Sex column (Nominal feature => OneHotEncoding)
convert_sex = pd.get_dummies(df['Sex'], drop_first=True)
df = df.join(convert_sex)
# Embarked column
df['Embarked'] = df['Embarked'].fillna('S')
convert_embarked = pd.get_dummies(df['Embarked'])
df = df.join(convert_embarked)

# Title is the only relevant info in Name column
def find_titles(title):
    return re.search(',\s{1}(.*?)\.', title).group()[2:-1]
df['Name'] = df['Name'].apply(find_titles)
df['Name'].replace(to_replace='^(?!(Mr|Miss|Mrs|Master)$).*',
                   value='Other', inplace=True, regex=True)
convert_title = pd.get_dummies(df['Name'])
df = df.join(convert_title)
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

# Impute missing Age values
#temp_df = df.copy()
#temp_df.dropna(inplace=True, subset=['Age'])
#mean_ = int(temp_df['Age'].median())  # Median
#df.fillna(mean_, inplace=True)
#df['Age'] = df['Age'].astype('int64')

index_NaN_age = list(df["Age"][df["Age"].isnull()].index)

for i in index_NaN_age :
    age_med = df["Age"].median()
    age_pred = df["Age"][((df['SibSp'] == df.iloc[i]["SibSp"])
                          & (df['Parch'] == df.iloc[i]["Parch"])
                          & (df['Pclass'] == df.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        df['Age'].iloc[i] = age_pred
    else :
        df['Age'].iloc[i] = age_med

# Identify family sizes
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
# Split family sizes
df['Single'] = df['FamilySize'].map(lambda s: 1 if s == 1 else 0)
df['SmallF'] = df['FamilySize'].map(lambda s: 1 if s == 2 else 0)
df['MedF'] = df['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
df['LargeF'] = df['FamilySize'].map(lambda s: 1 if s >= 5 else 0)

# Map Fare
def map_fare(price):
    if price <= 8:
        return 0.0
    elif price > 8 and price <= 15:
        return 1.0
    elif price > 15 and price <= 31:
        return 2.0
    else:
        return 3.0
df['Fare'] = df['Fare'].apply(map_fare)
# List of dropped columns
drop_list = ['Sex', 'Embarked',
             'SibSp', 'Parch',
             'Ticket', 'Name']
df.drop(drop_list, inplace=True, axis=1)
df = df.apply(pd.to_numeric, errors='coerce', axis=1)
# TRAINING
X_train = df.iloc[:891, :]
X_test = df.iloc[891:, :]
m = RandomForestClassifier(n_estimators=60,
                           n_jobs=-1,
                           random_state=1,
                           criterion='gini',
                           max_depth=8,
                           max_features='auto')
m = m.fit(X_train,y_train)
print('Score on training set: %.3f' % m.score(X_train, y_train))
predicted = m.predict(X_test)
result = pd.DataFrame({'PassengerId': X_test['PassengerId'],
                       'Survived': predicted})
result['PassengerId'] = result['PassengerId'].astype('Int32')
result['Survived'] = result['Survived'].astype('Int32')
print(result.dtypes)
result.to_csv('submission.csv', index=False)