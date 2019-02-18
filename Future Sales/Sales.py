import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Get train data
df = pd.read_csv('sales_train_v2.csv', low_memory=False)
print(df.columns)
print(df.head(3))
print(df.dtypes)
print(df.info())
print(df.describe())
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