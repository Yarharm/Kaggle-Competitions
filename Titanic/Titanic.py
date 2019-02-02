import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Get data
train_data = pd.read_csv('train.csv', low_memory=False)
test_data = pd.read_csv('test.csv', low_memory=False)

PassengerId = test_data['PassengerId']  # for future submission


## STEP 1: DETECT OUTLIERS
# Analyse effect of outliers on the outcome
#def detect_outliers(df, n, features):
#    outliers = []
#    for col in features:
#        pass


# STEP 2: Correlation analysis
def relation_graphs(data, heatmap_features=None):
    # Heatmap
    if heatmap_features is None:
        heatmap = sns.heatmap(data.corr(), annot=True,
                              fmt=".2f", cmap='coolwarm')
    else:
        heatmap = sns.heatmap(data[heatmap_features].corr(),
                              annot=True, fmt=".2f",
                              cmap='coolwarm')
    # Barplot [SibSP vs Survived]
    factor_SibSp = sns.factorplot(x='SibSp', y='Survived',
                                  data=data, kind='bar',
                                  size=5, palette='muted')\
        .set_ylabels('Survival probab')
    # Barplot [Parch vs Survived]
    factor_Parch = sns.factorplot(x='Parch', y='Survived',
                                  data=data, kind='bar',
                                  size=5, palette='muted')\
        .set_ylabels('Survival probab')
    # FacetGrid [Age vs Survived]
    facet_Age = sns.FacetGrid(data, col='Survived').map(sns.distplot, 'Age')

    # Barplot [Sex vs Survived]
    barplot_sex = sns.barplot(x='Sex', y='Survived',
                              data=data).set_ylabel('Survival Probab')

    # Barplot [Pclass vs Survived]
    factor_Pcalss = sns.factorplot(x='Pclass', y='Survived',
                                 data=data, kind='bar', size=5)\
        .set_ylabels('survival prob')
    plt.show()
#relation_graphs(train_data)


# STEP 3: Join two sets
y_train = train_data['Survived'].values
train_data.drop(['Survived'], axis=1, inplace=True)

# Number of input values for train and test data
train_len = train_data.shape[0]
test_len = test_data.shape[0]

# Get join DataFrame
df = pd.concat(objs=[train_data, test_data], axis=0, sort='False').reset_index(drop=True)


# STEP 4: Add missing values and transform skewed data
# 'Fare' adjustment
#print(df['Fare'].isna().sum())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Fare'] = np.log1p(df['Fare'])

# 'Embarked' adjustment
#print(df['Embarked'].isna().sum())
mode_embarked = df['Embarked'].mode()[0]  # most frequent in 'Embarked'
df['Embarked'] = df['Embarked'].fillna(mode_embarked)

# 'Sex' adjustment
lbl = LabelEncoder()
lbl.fit(list(df['Sex'].values))
df['Sex'] = lbl.transform(df['Sex'].values)

# 'Age' adjustment (263 missing values)
# Age is heavily correlated with Pclass, Parch and SibSp
# Apply meadian based on this correlation
index_NaN_age = list(df["Age"][df["Age"].isnull()].index)  # index of all nan values

#for i in index_NaN_age:
#    age_med = df["Age"].median()
#    age_pred = df["Age"][((df['SibSp'] == df.iloc[i]["SibSp"]) & (df['Parch'] == df.iloc[i]["Parch"]) &
#                          (df['Pclass'] == df.iloc[i]["Pclass"]))].median()

#    if not np.isnan(age_pred):
#        df['Age'].iloc[i] = age_pred
#    else:
#        df['Age'].iloc[i] = age_med
#print(df['Age'].isna().sum())

# 'Cabin' adjustment (1014 missing values)
