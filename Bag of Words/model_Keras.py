import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import re
# RNN imports
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding


# 25000 x 3 {id, sentiment, review}
df = pd.read_csv('labeledTrainData.tsv', sep='\t')
review_ID = df['id']
df = df.drop(['id'], axis=1)
#print(df.shape)

## Preprocessing
stop_words = set(stopwords.words('english'))

# Remove <br /> occurabces
df['review'] = df['review'].map(lambda row: re.sub(r'<br />', '', row))

# Splitting into Tokens
tokenizer = RegexpTokenizer(r'\w+')  # Pick alphanumeric characters and drop the rest
df['review'] = df['review'].map(lambda row: tokenizer.tokenize(row))

# Remove tokens that are not alphabetical (Numbers)
df['review'] = df['review'].map(lambda row: [word for word in row if word.isalpha()])

# Remove stop words
df['review'] = df['review'].map(lambda row: [word for word in row if not word in stop_words])

# Reduce words to its root
stemmer = PorterStemmer()
df['review'] = df['review'].map(lambda row: ' '.join([stemmer.stem(word) for word in row]))

# Input and target data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'],
                                                    test_size=0.2, random_state=1)

## Performance analysis of a simple Logistic Regression
# Transform text to feature vectors and feed it to the estimator
lg_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.9)),
    ('lg', LogisticRegression(n_jobs=-1))])
#lg_pipe.fit(X_train, y_train)
#print('LogReg accuracy: %.4f' % lg_pipe.score(X_test, y_test))

# RNN Model
vocab_size = 20000  # Max number of words to keep

# Remove punctuation, tokenize and vectorize inputs
tokenz = Tokenizer(num_words=vocab_size)
tokenz.fit_on_texts(df['review'])
sequences = tokenz.texts_to_sequences(df['review'])

# Pad sequences which are less than 250(Average review length)
sequences = pad_sequences(sequences, maxlen=250)
print(sequences.shape)

### Network layers: Embedded, LSTM, *Most likely*MaxPool, Dense, Dropout(Like CNN), Dense
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=250, name='embd'))
#model.add(Bidirectional(LSTM(???????????????????????))
# !!!!!!!!MaxPool or AverPool!!!!!!!!!!
#model.add(Dense(????????????))
#model.add(Dropout(?????????))
#model.add(Dense(????????))
