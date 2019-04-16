import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
# Get data
train = pd.read_csv('train.csv').drop("id", axis='columns')
targets = train['target']
train.drop('target', axis='columns', inplace=True)
test = pd.read_csv('test.csv').drop("id", axis='columns')

# Prepare models
lgbm = LGBMClassifier(objective='binary',
                      boosting_type='gbdt',
                      n_estimators=500,
                      metric='auc',
                      learning_rate=0.009,
                      num_leaves=8,
                      feature_fraction=0.5,
                      bagging_fraction=0.5,
                      bagging_freq=1,
                      max_depth=3,
                      reg_alpha=0.75,
                      reg_lambda=0.75)
rf = RandomForestClassifier(n_estimators=500, random_state=12)
nb = GaussianNB()
svc = SVC(kernel='rbf', C=1.0, gamma='auto', probability=True)
lr = LogisticRegression(max_iter=1000, class_weight='balanced', penalty='l1', C=0.1, solver='liblinear')
knn = KNeighborsClassifier(n_neighbors=63)
sgd = SGDClassifier(eta0=1, max_iter=1000, tol=0.0001, alpha=0.01, l1_ratio=1.0, learning_rate='adaptive', loss='log', penalty='elasticnet')

# Stacking Classifier
StackingModel = StackingCVClassifier(classifiers=[knn, rf, nb, svc, sgd, lgbm],
                            use_probas=True,
                            use_features_in_secondary=True,
                            meta_classifier=lr,
                            cv=5)

# Performance analysis
num_folds = 6
folds = KFold(n_splits=num_folds, random_state=16)

test_result = np.zeros(len(test))
auc_score = 0
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, targets)):
    print("Fold: ", fold_ + 1)

    X_train, y_train = train.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid, y_valid = train.iloc[val_idx], targets.iloc[val_idx]

    StackingModel.fit(X_train.values, y_train.values)

    y_pred = StackingModel.predict_proba(X_valid)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print(auc)
    auc_score += auc

    preds = StackingModel.predict_proba(test)
    test_result += preds[:, 1]

# print the average AUC across the folds and compute the final results on the test data
auc_score = auc_score / folds.n_splits
print("AUC score: ", auc_score)

submission = pd.read_csv('sample_submission.csv')
submission['target'] = test_result
submission.to_csv('stacking_submission.csv', index=False)