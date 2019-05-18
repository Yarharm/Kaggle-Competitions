import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import eli5
from eli5.sklearn import PermutationImportance
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import shap
from IPython.display import display
import os

# Get Data
train = pd.read_csv("train.csv", low_memory=False)
test = pd.read_csv("test.csv", low_memory=False)

#print(train.describe())
#print(train.isnull().any().any())


# Target analysis
#print(train['target'].value_counts())

# MODELING
y_train = train['target']
X_train = train.drop(['id', 'target'], axis=1)
X_test = test.drop('id', axis=1)

# Rescale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Prepare folds
n_folds = 20
folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2)
repeated_folds = RepeatedStratifiedKFold(n_splits=10, n_repeats=20, random_state=42)

def train_model(X, X_test, y, params, folds=folds, model_type='lgb',
                plot_feature_importacne=False, averaging='usual',model=None):
    oof = np.zeros(len(X_train))
    predictions = np.zeros(len(X_test))
    scores = []
    feature_importacne = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        X_train_temp, X_valid = X_train[train_index], X_train[valid_index]
        y_train_temp, y_valid = y_train[train_index], y_train[valid_index]

        # Logistic regression
        if model_type == 'sklearn':
            m = model
            m.fit(X_train_temp, y_train_temp)
            y_pred_valid = m.predict(X_valid)
            score = roc_auc_score(y_valid, y_pred_valid)  # Binary classification based on TP/FP
            y_pred = m.predict_proba(X_test)[:, 1]

        # CatBoost
        if model_type == 'cat':
            m = CatBoostClassifier(iterations=2000, eval_metric='AUC', **params)
            m.fit(X_train_temp, y_train_temp, eval_set=(X_valid, y_valid),
                      cat_features=[], use_best_model=True, verbose=False)
            y_pred_valid = m.predict(X_valid)
            y_pred = m.predict(X_test)

        # LGB
        if model_type == 'lgb':
            train_data = lgb.Dataset(X_train_temp, label=y_train_temp)
            valid_data = lgb.Dataset(X_valid, label=y_valid)

            m = lgb.train(params,
                          train_data,
                          num_boost_round=2000,
                          valid_sets=[train_data, valid_data],
                          verbose_eval=500,
                          early_stopping_rounds=200)

            y_pred_valid = m.predict(X_valid)
            y_pred = m.predict(X_test, num_iteration=m.best_iteration_)

        # Append scores
        oof[valid_index] = y_pred_valid.reshape(-1, )
        scores.append(roc_auc_score(y_valid, y_pred_valid))

        # Predictions averaging
        if averaging == 'usual':
            predictions += y_pred
        elif averaging == 'rank':
            predictions += pd.Series(y_pred).rank().values
        predictions /= n_folds

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    return oof, predictions, scores