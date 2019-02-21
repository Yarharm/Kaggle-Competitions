import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import re
import datetime
import warnings

# set warnings
warnings.filterwarnings('ignore')
# Get data
df = pd.read_csv('sales_train_v2.csv', low_memory=False)
test_set = pd.read_csv('test.csv', low_memory=False)
shops = pd.read_csv('shops.csv', low_memory=False)
item_cat = pd.read_csv('item_categories.csv', low_memory=False)
items = pd.read_csv('items.csv', low_memory=False)

# Format Date column

# Cpnvert from string(object) to the datetime object(datetime64)
df['date'] = df['date'].map(lambda x: datetime.datetime.strptime(x, '%d.%m.%Y'))

sales = df.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'date': ['min', 'max'],
                                                                  'item_price': 'mean',
                                                                  'item_cnt_day': 'sum'})

# Plot number of items per category
cat = items.groupby('item_category_id').item_id.count()
cat = cat.sort_values(by='item_id', ascending=False)

plt.figure(figsize=(8, 4))
ax = sns.barplot(cat.item_category_id, cat.item_id, alpha=0.8)
plt.title('Items per Category')
plt.ylabel('Number of items', fontsize=10)
plt.xlabel('Number of categories', fontsize=10)
plt.show()
print(cat)

# Single series analysis
def get_tables():
    print(df.head(2).transpose())
    print('\nShops')
    print(shops.head(2).transpose())
    print('\nItem Categories')
    print(item_cat.head(2).transpose())
    print('\nItems')
    print(items.head(2).transpose())
def get_df():
    print(df.head(2).transpose())
#print(test_set.head(3).transpose())
#print(df.dtypes)
#print(df.describe())
#print(df.head(2).transpose())
#get_tables()

# Explore Target (CONTAINS OUTLIER)
target = df['item_cnt_day']
def target_pot():
    fig, ax = plt.subplots(figsize=(16, 5))
    sns.distplot(target, ax=ax)

    # BoxPlot
    fig, ax = plt.subplots(figsize=(12, 3))
    sns.boxplot(x=target, data=df)

    plt.show()
## Combine categories in groups
#print(item_cat['item_category_name'].unique())


# Substitute category name by the first word only
item_cat['item_category_name'] = item_cat['item_category_name'].map(lambda x: re.match('^([^\s]+)', x)[0])

# Mapping item_id => item_category_id
item_id_to_item_cat = {}
for index, row in items.iterrows():
    item_id_to_item_cat.update({row['item_id']: row['item_category_id']})

# Mapping item_category_id => category_name
item_cat_to_cat_name = {}
for index, row in item_cat.iterrows():
    item_cat_to_cat_name.update({row['item_category_id']: row['item_category_name']})

# Map item ids
df['item_name'] = df['item_id'].map(lambda x: item_cat_to_cat_name[item_id_to_item_cat[x]])

## UTILIZE SHOP NAMES

# Separate by shop location (Moskva, Omsk) and shop type (TRK, TC)
# Mapping shop_id => shop_name
shop_id_to_shop_name = {}
for index, row in shops.iterrows():
    shop_id_to_shop_name.update({row['shop_id']: row['shop_name']})

df['shop_location'] = df['shop_id'].map(lambda x: shop_id_to_shop_name[x].split()[0])
df['shop_type'] = df['shop_id'].map(lambda x: shop_id_to_shop_name[x].split()[1])


## Explore item_price (CONTAINT OUTLIER => BOXPLOT)
def price_plot():
    sns.set(style='whitegrid')
    sns.boxplot(x=df['item_price'])

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.scatter(df['item_price'], df['item_cnt_day'])
    ax.set_xlabel('Price for the item distribution')

    plt.show()


# APPENDING PARTIAL NOMINAL DATA TO THE DF
#dummy = pd.get_dummies(df['item_name'])
#df = pd.concat([df, dummy], axis=1)

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
