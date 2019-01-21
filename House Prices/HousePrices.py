import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew

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
    fig, ax = plt.subplots()
    ax.scatter(x=train['GrLivArea'],
               y=train['SalePrice'],
               c='purple')
    plt.xlabel('GrLivArea', fontsize=12)
    plt.ylabel('SalePrice', fontsize=12)
    plt.show()
    return

#outlier(train)

# Delete extreme outliers
train.drop(train[(train['GrLivArea']>4000) &
                 (train['SalePrice']<300000)].index,
           inplace=True)
#outlier(train)

# Target variable analysis
def target_analysis(train):
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

# Get quantile-quantile plot
    fig = plt.figure()
    res = stats.probplot(train['SalePrice'], plot=plt)
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
    corrmat = train_data.corr()
    plt.subplots(figsize=(12,9))
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
            'BsmtUnfSF','TotalBsmtSF',
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
