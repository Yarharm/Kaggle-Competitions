import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
import missingno as msno
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Get data
train_data = pd.read_csv('train.csv', low_memory=False)
test_data = pd.read_csv('test.csv', low_memory=False)

PassengerId = test_data['PassengerId']

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
y_train = train_data['Survived'].values.astype(int)
train_data.drop(['Survived'], axis=1, inplace=True)

# Number of input values for train and test data
train_len = train_data.shape[0]

# Get join DataFrame
df = pd.concat(objs=[train_data, test_data], axis=0, sort='False').reset_index(drop=True)


# STEP 4: Add missing values and transform skewed data
print(msno.bar(df))

# Swarmplot
plt.figure(figsize=(12,12))
sns.swarmplot(x="Sex", y="Age", hue='Pclass', data=df,
              size=10, palette=['orange','brown','purple'])

plt.figure(figsize=(12,12))
sns.swarmplot(x="Sex", y="Age", hue='Survived',
              data=df, size=10, palette='viridis')
plt.show()

# Crosstab
pd.crosstab([df.Sex,df.Survived],df.Pclass, margins=True).style.background_gradient(cmap='autumn_r')

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

for i in index_NaN_age:
    age_med = df["Age"].median()
    age_pred = df["Age"][((df['SibSp'] == df.iloc[i]["SibSp"]) & (df['Parch'] == df.iloc[i]["Parch"]) &
                          (df['Pclass'] == df.iloc[i]["Pclass"]))].median()

    if not np.isnan(age_pred):
        df['Age'].iloc[i] = age_pred
    else:
        df['Age'].iloc[i] = age_med
#print(df['Age'].isna().sum())
#print(df.isna().sum())

# 'Cabin' adjustment (1014 missing values)
#print(df['Cabin'].dtype)
df['Cabin'] = df['Cabin'].map(lambda x: x[0] if not pd.isna(x) else 'X')

# FEATURE ENGINEERING
# Titles
titles = [i.split(',')[1].split('.')[0].strip() for i in df['Name']]
df['Title'] = pd.Series(titles)  # add new feature
#print(df['Title'].unique())

# Get most dominant titles
df['Title'] = df['Title'].replace([
    'Don', 'Rev', 'Dr', 'Major', 'Lady',
    'Sir', 'Mile', 'Col', 'Capt', 'the Countess',
    'Jonkheer', 'Dona'], 'Rest')
df['Title'] = df['Title'].map({'Mr': 0, 'Mrs': 1,
                               'Miss': 1, 'Master': 2,
                               'Rest': 3, 'Mme': 1,
                               'Ms': 1, 'Mlle': 1})
df['Title'] = df['Title'].astype(int)
#print(df.isna().sum())
#print(df['Title'].unique())

# Name is unnecessary at this point
df.drop(['Name'], axis=1, inplace=True)

# Add Family size
df['Fsize'] = df['SibSp'] + df['Parch'] + 1
df['Single'] = df['Fsize'].map(lambda x: 1 if x == 1 else 0)
df['SFsize'] = df['Fsize'].map(lambda x: 1 if x == 2 else 0)
df['MSingle'] = df['Fsize'].map(lambda x: 1 if 3 <= x <= 4 else 0)
df['LSingle'] = df['Fsize'].map(lambda x: 1 if x >= 5 else 0)

# Ticket
#print(df['Ticket'].head())
# Extract prefixes
def process_ticket(ticket):
    if not ticket.isdigit() and '.' not in ticket:
        return ticket.split()[0]
    elif not ticket.isdigit():
        return ticket.split()[0][:-1]
    else:
        return 'X'
df['Ticket'] = df['Ticket'].map(process_ticket)

# Pclass
#print(df['Pclass'].dtype)
df['Pclass'] = df['Pclass'].astype('category')

#print(df.shape)
# Aggregate nominal data and drop unnecessary features
df.drop(['PassengerId'], axis=1, inplace=True)
df = pd.get_dummies(df, columns=['Embarked', 'Title',
                                 'Cabin', 'Ticket',
                                 'Pclass'])
#print(df.shape)


# MODELING
X_train = df[: train_len]
X_test = df[train_len:]
#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)


# Ensembling with Majority Vote (Voting Classifier)
# Base classifiers: SVC, AdaBOOST, RMC, ETC, GradBoosting
kfold = StratifiedKFold(10)

# AdaBOOST
DTC = DecisionTreeClassifier(random_state=1, max_features='auto',
                             class_weight='balanced', max_depth=None)
adaCL = AdaBoostClassifier(base_estimator=DTC)
ada_grid = {'base_estimator__criterion': ['gini', 'entropy'],
            'base_estimator__splitter': ['random', 'best'],
            'algorithm': ['SAMME', 'SAMME.R'],
            'n_estimators': [5, 10, 15, 25,
                             35, 45, 50, 60],
            'learning_rate': [0.0001, 0.001, 0.01, 0.1,
                              0.2, 0.3, 0.5, 1.0, 1.5]}
ada_gs = GridSearchCV(adaCL, param_grid=ada_grid,
                     cv=kfold, scoring='accuracy',
                     n_jobs=-1)
ada_gs.fit(X_train, y_train)
ada_best = ada_gs.best_estimator_
print(ada_best)

# ExtraTrees
ETC = ExtraTreesClassifier()

etc_grid = {'n_estimators': [10, 50, 100, 150, 200],
            'max_depth': [None],
            'max_features': [1, 3, 10,
                             'sqrt', 'log2'],
            'min_samples_split': [2, 3, 6],
            'min_samples_leaf': [1, 3, 6],
            'bootstrap': [False],
            'criterion': ['gini', 'entropy']
            }

etc_gs = GridSearchCV(ETC, param_grid=etc_grid,
                      cv=kfold, scoring='accuracy',
                      n_jobs=-1)
etc_gs.fit(X_train, y_train)

etc_best = etc_gs.best_estimator_
print(etc_best)

# RandomForest
RFC = RandomForestClassifier()

rfc_grid = {'max_depth': [None],
            'max_features': [1, 3, 6, 10],
            'min_samples_split': [2, 3, 6, 10],
            'min_samples_leaf': [1, 3, 6, 10],
            'bootstrap': [False],
            'n_estimators': [100, 150, 200,
                             250, 300, 350],
            'criterion': ['gini', 'entropy']}
rfc_gs = GridSearchCV(RFC, param_grid=rfc_grid,
                      cv=kfold, scoring='accuracy',
                      n_jobs=-1)
rfc_gs.fit(X_train, y_train)
rfc_best = rfc_gs.best_estimator_
print(rfc_best)

# Gradient booosting
GBC = GradientBoostingClassifier()

gbc_grid = {'loss': ['deviance'],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'n_estimators': [100, 150, 200,
                             300, 350],
            'min_samples_leaf': [70, 100, 150],
            'max_depth': [4, 6, 8],
            'max_features': [0.3, 0.1, 'sqrt', 'log2']
            }
gbc_gs = GridSearchCV(GBC, param_grid=gbc_grid,
                      cv=kfold, scoring='accuracy',
                      n_jobs=-1)
gbc_gs.fit(X_train, y_train)
gbc_best = gbc_gs.best_estimator_
print(gbc_best)

# SVC
SVC = SVC(probability=True)

svc_grid = {'C': [1, 10, 500, 100,
                  200, 300, 1000],
            'kernel': ['rbf', 'poly'],
            'gamma': [0.001, 0.01, 0.1, 1]}
svc_gs = GridSearchCV(SVC, param_grid=svc_grid,
                      cv=kfold, scoring='accuracy',
                      n_jobs=-1)
svc_gs.fit(X_train, y_train)
svc_best = svc_gs.best_estimator_
print(svc_best)

# Ensemble
votingClass = VotingClassifier(
    estimators=[('adac', ada_best),
                ('extc', etc_best),
                ('rfc', rfc_best),
                ('gbc', gbc_best),
                ('svc', svc_best)], voting='soft', n_jobs=-1)

# ROC curve
ada_fpr, ada_tpr, thresholds = roc_curve(y_test, ada_best.predict_proba(X_test)[:,1])
etc_fpr, etc_tpr, etc_thresholds = roc_curve(y_test, etc_best.predict_proba(X_test)[:,1])
svc_fpr, svc_tpr, svc_thresholds = roc_curve(y_test, svc_best.predict_proba(X_test)[:,1])
gbc_fpr, gbc_tpr, ada_thresholds = roc_curve(y_test, gbc_best.predict_proba(X_test)[:,1])

plt.figure(figsize=(9,9))
log_roc_auc = roc_auc_score(y_test, ada_best.predict(X_test))
print("logreg model AUC = {} " .format(log_roc_auc))
rf_roc_auc = roc_auc_score(y_test, etc_best.predict(X_test))
print("random forest model AUC ={}" .format(rf_roc_auc))
knn_roc_auc = roc_auc_score(y_test, svc_best.predict(X_test))
print("KNN model AUC = {}" .format(knn_roc_auc))
gbc_roc_auc = roc_auc_score(y_test, gbc_best.predict(X_test))
print("GBC Boost model AUC = {}" .format(gbc_roc_auc))

# Plot Adaboost ROC
plt.plot(ada_fpr, ada_tpr, label='Logistic Regression')

# Plot Extra trees ROC
plt.plot(etc_fpr, etc_tpr, label='Random Forest')

# Plot SVC ROC
plt.plot(svc_fpr, svc_tpr, label=' KnnClassifier')

# Plot GradientBooseting Boost ROC
plt.plot(gbc_fpr, gbc_tpr, label='GradientBoostingclassifier')

# Plot Base Rate ROC
plt.plot([0, 1], [0, 1], label='Base Rate' 'k--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()

votingClass = votingClass.fit(X_train, y_train)
pred = votingClass.predict(X_test)
pred = pd.Series(pred, name='Survived')
result = pd.concat([PassengerId, pred], axis=1)
print(result.shape)
result.to_csv('ensemble_sub.csv', index=False)
