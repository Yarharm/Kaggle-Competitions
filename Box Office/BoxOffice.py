import pandas as pd



df = pd.read_csv('train.csv', low_memory=False)

# General exploration
def gen_exploration():

    # Get shape of data
    print(df.shape)

    # Get all column
    print(df.columns)

    # Get information
    print(df.info())

    # Percentage of missing data
    print((df.isna().sum()[df.isna().sum() != 0] / df.shape[0] * 100).sort_values(ascending=False))

# Missing values imputation
print(df['belongs_to_collection'].unique())
