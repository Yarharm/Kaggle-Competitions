from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import re


df = pd.read_csv('train.csv', low_memory=False)

# Encode Sex column (Nominal feature => OneHotEncoding)
convert_sex = pd.get_dummies(df['Sex'], drop_first=True)
df = df.join(convert_sex)

# Embarked column
df['Embarked'] = df['Embarked'].fillna('C')
convert_embarked = pd.get_dummies(df['Embarked'], drop_first=True)
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
#print(df[['Cabin', 'Survived']].groupby(['Cabin']).mean())

# Impute missing age values
temp_df = df.copy()
temp_df.dropna(inplace=True, subset=['Age'])
mean_ = int(temp_df['Age'].mean())
df.fillna(mean_, inplace=True)
df['Age'] = df['Age'].astype('int64')

# Identify family sizes
df['FamilySize'] = df['SibSp'] + df['Parch']

drop_list = ['Sex', 'Embarked',
             'SibSp', 'Parch',
             'Ticket', 'Name']
df.drop(drop_list, inplace=True, axis=1)
df = df.apply(pd.to_numeric, errors='coerce', axis=1)
# TRAINING
X_train = df.drop('Survived',axis=1)
y_train = df['Survived']

# ::=> SIMPLE RANDOM FOREST
#m = RandomForestClassifier(n_jobs=-1, n_estimators=10)
#m.fit(X_train,y_train)
#print(m.score(X_train,y_train))

# ::=> PIPELINE WITH RANDOM FOREST AND K-FOLD VALIDATION
#pipe = make_pipeline(StandardScaler(),
#                     RandomForestClassifier(n_estimators=100,
#                                            random_state=1))
#pipe.fit(X_train, y_train)
#y_pred = pipe.predict(X_test)  # TEST CASE HERE
#print("Accuracy: %.3f" % pipe.score(X_train, y_train))
# K-fold cross-validation
#scores = cross_val_score(estimator=pipe,
#                         X=X_train,
#                         y=y_train,
#                         cv=10,
#                         n_jobs=-1)
#print('Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# SVC
pipe = make_pipeline(StandardScaler(),
                     SVC(random_state=1))
param_range = [0.0001, 0.001, 0.01, 0.1,
               1.0, 10.0, 100.0, 1000.0]

param_grid = [{'svc__C': param_range,
               'svc__kernel': ['linear']},
              {'svc__C': param_range,
               'svc__kernel': ['rbf'],
               'svc__gamma': param_range}]

#gs = GridSearchCV(estimator=pipe,
 #                 param_grid=param_grid,
 #                 scoring='accuracy',
 #                 cv = 10,
 #                 n_jobs=-1)
#gs = gs.fit(X_train, y_train)
#print("Score %.3f" % gs.best_score_)
#print(gs.best_params_)

# SCV WITH NESTED CROSS-VALIDATION
gs = GridSearchCV(estimator=pipe,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=2)

scores = cross_val_score(gs, X_train, y_train,
                         scoring='accuracy', cv=5)
print('Accuracy %.3f' % np.mean(scores))