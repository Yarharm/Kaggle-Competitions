import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import re
# Get data
df = pd.read_csv('sales_train_v2.csv', low_memory=False)
test_set = pd.read_csv('test.csv', low_memory=False)
shops = pd.read_csv('shops.csv', low_memory=False)
item_cat = pd.read_csv('item_categories.csv', low_memory=False)
items = pd.read_csv('items.csv', low_memory=False)

def get_tables():
    print(df.head(2).transpose())
    print('\nShops')
    print(shops.head(2).transpose())
    print('\nItem Categories')
    print(item_cat.head(2).transpose())
    print('\nItems')
    print(items.head(2).transpose())

#print(test_set.head(3).transpose())
#print(df.dtypes)
#print(df.describe())
#print(df.head(2).transpose())
#get_tables()

# Explore Target
target = df['item_cnt_day']
sns.distplot(target, fit=norm)
# Fitted parameters of the function
(mu, sigma) = norm.fit(target)

# Plot the distribution
# Place legend on the axis
plt.legend(['Normal dist. ($\mu=$ {:.2f} and'
            '$\sigma=$ {:.2f})'.format(mu, sigma)],
           loc='best')
plt.title('Items cold distribution')
plt.ylabel('Frequency')
#plt.show()

## Combine categories in groups
print(item_cat['item_category_name'].unique())
# Hash table of categories
category_table = {}
# Retrive first word from group: ^([^\s]+)
item_cat['item_category_name'].map(lambda x: category_table.update({re.match('^([^\s]+)', x)[0]: 1}))
for key, value in category_table.items():
    print(key)



## DISTRIBUTE ITEMS TO CATEGORIES
## UTILIZE SHOP NAMES
## ??????????UTILIZE ITEM NAME????????????

## EDA
# Scatterplot matrix(Pairplot)
cols = ['date', 'date_block_num', 'item_price', 'item_cnt_day']
sns.pairplot(df[cols], height=2.5)
plt.tight_layout()
plt.show()

# Correlation matrix
cm = np.corrcoef(df.values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True,
                 annot=True, square=True,
                 fmt='.2f', annot_kws={'size': 15},
                 yticklabels=cols, xticklabels=cols)
plt.show()
