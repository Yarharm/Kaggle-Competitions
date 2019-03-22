import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Get Data
train = pd.read_csv("train.csv", low_memory=False)
test = pd.read_csv("test.csv", low_memory=False)

print(train.describe())
print(train.isnull().any().any())


# Target analysis
print(train['target'].value_counts())

# Distribution of a small sample(First 28 features)
plt.figure(figsize=(26, 24))
for i, col in enumerate(list(train.columns)[2:30]):
    plt.subplot(7, 4, i + 1)
    plt.hist(train[col])
    plt.title(col)
#plt.show()


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

# Logistic regression analysis
m = LogisticRegression(class_weight='balanced', penalty='l1',
                       C=0.1, solver='liblinear')

oof = np.zeros(len(X_train))
predictions = np.zeros(len(X_test))
scores = []
boolean = True
for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train, y_train)):
    # Varify splits
    if boolean:
        print("Fold splits are", train_index, valid_index)
        boolean = False
    X_train_temp, X_valid = X_train[train_index], X_train[valid_index]
    y_train_temp, y_valid = y_train[train_index], y_train[valid_index]
    m.fit(X_train_temp, y_train_temp)
    y_pred_valid = m.predict(X_valid)
    print(y_pred_valid.shape)
    score = roc_auc_score(y_valid, y_pred_valid)  # Binary classification based on TP/FP
    print('Score is', score)
    y_pred = m.predict_proba(X_test)[:, 1]

    # Append scores
    oof[valid_index] = y_pred_valid.reshape(-1, )
    scores.append(roc_auc_score(y_valid, y_pred_valid))

print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
