import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv", low_memory=False)

print(train.describe())
print(train.isnull().any().any())


# Target analysis
print(train['target'].value_counts())

# Distribution of a small sample(First 28 features)
plt.figure(figsize=(26, 24))
for i, col in enumerate(list(train.columns)[2:30]):
    plt.subplot(7, 4, i + 1)
    plt.hist(train[col])
    plt.title(col)
#plt.show()
