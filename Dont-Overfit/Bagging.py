# Bagging vs Boosting analysis

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

# Get data
train = pd.read_csv("train.csv", low_memory=False)
test = pd.read_csv("test.csv", low_memory=False)

# Save test data
X_test  = test.drop("id", axis=1)

# Get X_train and y_train
cols = ["target", "id"]
X = train.drop(cols, axis=1)
y = train["target"]

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Helpers for Model analysis
def modelAnalysis(predVal, actualVal):
    print(classification_report(predVal, actualVal))
    print(confusion_matrix(predVal, actualVal))
    print('Accuracy is ', accuracy_score(predVal, actualVal))


## MODELS
# Random Forest Classifier(Bagging)
RFC = RandomForestClassifier(max_depth=2)
RFC.fit(X_train, y_train)
y_pred = RFC.predict(X_val)
modelAnalysis(y_pred, y_val)

# Bagging Classifier(Bagging)
bagModel = BaggingClassifier()
bagModel.fit(X_train, y_train)
y_pred = bagModel.predict(X_val)
modelAnalysis(y_pred, X_val)

# AdaBoost Classifier (Boosting)
AdaBoost = AdaBoostClassifier()
AdaBoost.fit(X_train, y_train)
y_pred = AdaBoost.predict(X_val)
modelAnalysis(y_pred, y_val)