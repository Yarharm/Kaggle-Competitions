# Bagging vs Boosting analysis

import pandas as pd
from sklearn.model_selection import train_test_split
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

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
def analyse_model(X_train, X_val, y_train, y_val, model="RFC",
                  feature_importance=False):

    if model == "RFC":
        m = RandomForestClassifier(max_depth=2)
    elif model == "BagModel":
        m = BaggingClassifier()
    elif model == "AdaBoost":
        m = AdaBoostClassifier()
    elif model == "GradBoost":
        m = GradientBoostingClassifier()
    elif model == "LDA":
        m = LinearDiscriminantAnalysis()
    elif model == "QDA":
        m = QuadraticDiscriminantAnalysis()

    m.fit(X_train, y_train)
    y_pred = m.predict(X_val)
    print(model + " performance: ")
    modelAnalysis(y_pred, y_val)

    if feature_importance:
        perm = PermutationImportance(m, random_state=1).fit(X_val, y_val)
        eli5.show_weights(perm, feature_names=X_val.columns.tolist(), top=50)


# Bagging models
analyse_model(X_train, X_val, y_train, y_val, model="RFC", feature_importance=True)
analyse_model(X_train, X_val, y_train, y_val, model="BagModel", feature_importance=False)

# Boosting models
# Additional models to analyse: Light GBM, CatBoost, XGBM
analyse_model(X_train, X_val, y_train, y_val, model="AdaBoost", feature_importance=False)
analyse_model(X_train, X_val, y_train, y_val, model="GradBoost", feature_importance=True)
analyse_model(X_train, X_val, y_train, y_val, model="LDA", feature_importance=False)
analyse_model(X_train, X_val, y_train, y_val, model="QDA", feature_importance=False)

