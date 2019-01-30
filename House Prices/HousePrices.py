import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm, skew
from scipy.special import boxcox1p

# Models in test
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score

train = pd.read_csv('train.csv', low_memory=False)
test = pd.read_csv('test.csv', low_memory=False)

# Save 'Id' column
train_ID = train['Id']
test_ID = test['Id']

# Drop 'Id' column
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)


# Identify outliers
def outlier(train):
    """
    Identify extreme outliers in any feature (Ex Ground Living Area)
    :param train: training DataFrane
    :return:
    """
    fig, ax = plt.subplots()
    ax.scatter(x=train['GrLivArea'],
               y=train['SalePrice'],
               c='purple')
    plt.xlabel('GrLivArea', fontsize=12)
    plt.ylabel('SalePrice', fontsize=12)
    plt.show()
    return


# outlier(train)

# Delete extreme outliers
train.drop(train[(train['GrLivArea'] > 4000) &
                 (train['SalePrice'] < 300000)].index,
           inplace=True)


# outlier(train)

# Target variable analysis
def target_analysis(train):
    """
    Identify how skewed targer feature is
    :param train: training DataFrame
    :return: Normal distribution
    """
    sns.distplot(train['SalePrice'], fit=norm)

    # Fitted parameters of the function
    (mu, sigma) = norm.fit(train['SalePrice'])

    # Plot the distribution
    # Place legend on the axis
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and'
                '$\sigma=$ {:.2f})'.format(mu, sigma)],
               loc='best')
    plt.title('SalePrice distribution')
    plt.ylabel('Frequency')
    plt.show()
    return


# TARGET VARIABLE IS RIGHT SKEWED.
# Transfrom to normally distributed using Log-transformation
# Log-transformation deals with skewed data
train['SalePrice'] = np.log1p(train['SalePrice'])
# target_analysis(train)

# FEATURE ENGINEERING
# Concat Train and Test
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train['SalePrice'].values
data = pd.concat((train, test), sort=False).reset_index(drop=True)
data.drop(['SalePrice'], axis=1, inplace=True)

# Analyse missing data
def percent_missing_data_by_feature(data):
    """
    Percentage of missing data in each feature
    :param data: training DaraFrame
    :return: barplot
    """
    # Missing data for the first 30 features
    data_na = (data.isna().sum() / len(data)) * 100
    data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]

    # Graph most missing columns
    f, ax = plt.subplots(figsize=(15, 12))
    plt.xticks(rotation='90')  # rotate labels on x-axis
    sns.barplot(x=data_na.index, y=data_na)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    plt.show()
    return


# percent_missing_data_by_feature(data)

# Data Correlation analysis
def corr_map(train_data):
    """
    Get correlation map
    :param train_data: training DataFrame
    :return: heatmap plot
    """
    corrmat = train_data.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=0.9, square=True)
    plt.show()
    return


# corr_map(train)

# Impute missing values
data['PoolQC'].fillna('None', inplace=True)
data['MiscFeature'].fillna('None', inplace=True)
data['Alley'].fillna('None', inplace=True)
data['Fence'].fillna('None', inplace=True)
data['FireplaceQu'].fillna('None', inplace=True)

# LogFrontage is similar in the median of all neighbourhoods
data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))

# Garage
for col in ('GarageType', 'GarageFinish',
            'GarageQual', 'GarageCond'):
    data[col].fillna('None', inplace=True)
# 0 cars in Garage that do not exists
for col in ('GarageYrBlt', 'GarageArea',
            'GarageCars'):
    data[col].fillna(0, inplace=True)

# Basement
for col in ('BsmtFinSF1', 'BsmtFinSF2',
            'BsmtUnfSF', 'TotalBsmtSF',
            'BsmtFullBath', 'BsmtHalfBath'):
    data[col].fillna(0, inplace=True)
for col in ('BsmtQual', 'BsmtCond',
            'BsmtExposure', 'BsmtFinType1',
            'BsmtFinType2'):
    data[col].fillna('None', inplace=True)

# Masonry
data['MasVnrType'].fillna('None', inplace=True)
data['MasVnrArea'].fillna(0, inplace=True)

# General zoning
data['MSZoning'].fillna(data['MSZoning'].mode()[0], inplace=True)

# Utilities can be removed
data.drop(['Utilities'], axis=1, inplace=True)

# Functional (NA => Typ)
data['Functional'].fillna('Typ', inplace=True)

# Electrical
data['Electrical'].fillna(data['Electrical'].mode()[0], inplace=True)

# Kitched quality
data['KitchenQual'].fillna(data['KitchenQual'].mode()[0], inplace=True)

# Exterior
data['Exterior1st'].fillna(data['Exterior1st'].mode()[0], inplace=True)
data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0], inplace=True)

# SaleType
data['SaleType'].fillna(data['SaleType'].mode()[0], inplace=True)

# BuildingClass
data['MSSubClass'].fillna('None', inplace=True)

# /////////////////////////////////////////
data['MSSubClass'] = data['MSSubClass'].astype(str)
data['OverallCond'] = data['OverallCond'].astype(str)
data['YrSold'] = data['YrSold'].astype(str)
data['MoSold'] = data['MoSold'].astype(str)
# /////////////////////////////////////////
# Label encode ordinal features
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond',
        'GarageQual', 'GarageCond', 'ExterQual',
        'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual',
        'BsmtFinType1', 'BsmtFinType2', 'Functional', 'Fence',
        'BsmtExposure', 'GarageFinish', 'LandSlope', 'LotShape',
        'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass',
        'OverallCond', 'YrSold', 'MoSold')
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(data[c].values))
    data[c] = lbl.transform(data[c].values)

# Add total house area feature
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']

# VERIFY SKEWED FEATURES
# get all non-string features
numeric_featrs = data.dtypes[data.dtypes != 'object'].index
skewed_featrs = data[numeric_featrs].apply(lambda x: skew(x)).sort_values(ascending=False)
# Transform to DataFrame
skewness = pd.DataFrame({'Skew': skewed_featrs})

# Transorm highly skewed data with Box-Cox transormation
# Box-Cox applies log-transormation for lmbda = 0
skewness = skewness[abs(skewness) > 0.75]  # Highly skewed for the deviation of 0.75+
for feature in skewness.index:
    data[feature] = boxcox1p(data[feature], 0.15)

# Transform the rest of the nominal features
data = pd.get_dummies(data)
train = data[:ntrain]
test = data[ntrain:]

# MODELING
# Cross valiation strategy
n_folds = 5  # For relatively small dataset decrease from 10 to 5


# Cross_val_score does not shuffle data
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=1).get_n_splits(train.values)  # Non-stratified
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)


# LASSO and Elastic Net regressions => sensitive towards outliers (Apply RobustScaler())
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=2))

# Kernel Ridge
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

# Gradient boost with 'huber' for robustness => (Requires GridSearch polishing)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=1)

# Evaluate LASSO, ENet, KRR and GBoost
#score = rmsle_cv(lasso)
#print('Lasso score: %.4f' % score.mean())

#score = rmsle_cv(ENet)
#print('ENet score: %.4f' % score.mean())

#score = rmsle_cv(KRR)
#print('KRR score: %.4f' % score.mean())

#score = rmsle_cv(GBoost)
#print('GBoost score: %.4f' % score.mean())


# Simple Stacking class
class StackedModels():
    """
    Exploting stacking as a part of Ensemble technique
    Ensemble (Boosting, Bagging, Stacking)
    """

    def __init__(self, models):
        self.models = models
        return None

    def fit(self, X, y):
        """
        Fit data in clones (self.models_) of the original models
        :param X: array-like {n_records, n_features) training data
        :param y: array {n_records} target data
        :return: reference to the object on which it was called
        """
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        """
        Average predictions on the cloned models
        :param X: array-like {n_records, n_features) training data
        :return:
        """
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        # print('Shape of the predictions inside predict call %.4f' % predictions.shape)
        return np.mean(predictions, axis=1)  # mean of each row


averaged_models = StackedModels(models=(ENet, GBoost,
                                        KRR, lasso))
#score = rmsle_cv(averaged_models)
#print('Averaged models score %.4f' % score.mean())

# Meta-model Stacking class
class StackedMetaModel():
    """
    Fit part:
    Meta model is trained on out-of-folds predictions.
    Out-of-folds predictions contain predictions of base models on each fold.
    Folds are obtained by KFold.
    Predict part:
    Predictions of the base models are averaged and fed to the meta model,
    which yeilds final prediction.
    """
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        return None

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds,
                      shuffle=True,
                      random_state=1)
        out_of_fold_predictions = np.zeros((X.shape[0],  # predictions of the base models
                                            len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
        # Train cloned meta-model
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([np.column_stack([model.predict(X)
                                                          for model in base_models]).mean(axis=1)
                                         for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


stacked_meta_model = StackedMetaModel(base_models=(ENet, GBoost, KRR),
                                      meta_model=lasso,
                                      n_folds=5)
#score = rmsle_cv(stacked_meta_model)
#print('Stacked meta model score: %.4f' % score.mean())

# Using simple base models does not yeild a big change
# Submission
sub = pd.DataFrame()
stacked_meta_model.fit(train.values, y_train)
sub['Id'] = test_ID
sub['SalePrice'] = np.expm1(stacked_meta_model.predict(test.values))
sub.to_csv('submission.csv', index=False)