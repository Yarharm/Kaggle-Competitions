import pandas as pd
import string
from nltk.corpus import stopwords

# 25000 x 3 {id, sentiment, review}
df = pd.read_csv('labeledTrainData.tsv', sep='\t')
review_ID = df['id']
df = df.drop(['id'], axis=1)
#print(df.shape)

# Preprocess data
stop_words = set(stopwords.words('english'))

# Splitting into Tokens
#print(df['review'].head(1))
df['review'] = df['review'].map(lambda row: row.split())
#print(df['review'].head(1))

# Remove punctuation
tr_table = str.maketrans('', '', string.punctuation)
df['review'] = df['review'].map(lambda row: [word.translate(tr_table) for word in row])
#print(df['review'].head(1))

# Remove tokens that are not alphabetical
df['review'] = df['review'].map(lambda row: [word for word in row if word.isalpha()])
#print(df['review'].head(1))

# Remove stop words
df['review'] = df['review'].map(lambda row: [word for word in row if not word in stop_words])
#print(df['review'].head(1))

# Remove very short tokens
df['review'] = df['review'].map(lambda row: [word for word in row if len(word) > 1])

