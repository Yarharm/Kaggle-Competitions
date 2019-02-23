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

# Duplicated rows
#print(df[df.duplicated(keep=First)])
#df.drop_duplicates()

## MEMORY SAVING MECHANISM(DOWNCASTING)
def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df


# Create pivot table
sales_by_item_id = df.pivot_table(index=['item_id'], values=['item_cnt_day'],
                                  columns='date_block_num', aggfunc=np.sum, fill_value=0).reset_index()
sales_by_item_id.columns = sales_by_item_id.columns.droplevel().map(str)
sales_by_item_id = sales_by_item_id.reset_index(drop=True).rename_axis(None, axis=1)
sales_by_item_id.columns.values[0] = 'item_id'
#print(sales_by_item_id)


# Products with no sales for the last 6 months
outdated_items = sales_by_item_id[sales_by_item_id.loc[:, '27':].sum(axis=1) == 0]
print(outdated_items)

## Format Date column
# Convert from string(object) to the datetime object(datetime64)
#df['date'] = df['date'].map(lambda x: datetime.datetime.strptime(x, '%d.%m.%Y'))
#df['day'] = df['date'].apply(lambda x: x.day)
#df['month'] = df['date'].apply(lambda x: x.month)
#df['year'] = df['date'].apply(lambda x: x.year)
#df['weekend'] = df['date'].apply(lambda x: x.weekday())  # 0 = Monday, 6 = Sunday

#sales = df.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'date': ['min', 'max'],
#                                                                  'item_price': 'mean',
#                                                                  'item_cnt_day': 'sum'})

# Left join item_category_id to the main DataFrame
#df = df.join(items['item_category_id'])
#print(df.columns)






# Add item count per month
#mapping = df.groupby('date_block_num').item_cnt_day.sum()
#df['item_cnt_month'] = df['date_block_num'].apply(lambda x: mapping[x])
#print(df.tail(2).transpose())

# Total sales per month
m_sales = df.groupby('date_block_num').item_cnt_day.sum()
def montly_sales(sales):
    plt.figure(figsize=(12, 8))
    plt.ylabel('Sales')
    plt.xlabel('Time')
    plt.title('Sales per month')
    plt.plot(sales)
    plt.show()
    ## Conclusion: Observed seasonality

# Explore trend, seasonality and residuals
def stats(sales):
    residual_1 = sm.tsa.seasonal_decompose(sales.values, freq=12, model='multiplicative')
    residual_1.plot()
    residual_2 = sm.tsa.seasonal_decompose(sales.values, freq=12, model="additive")
    residual_2.plot()
    plt.show()

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
