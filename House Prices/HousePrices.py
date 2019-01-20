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



