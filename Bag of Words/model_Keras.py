import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import re
# RNN imports
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Bidirectional, \
                        GlobalAveragePooling1D, GlobalMaxPooling1D, Dropout

# 25000 x 3 {id, sentiment, review}
df = pd.read_csv('labeledTrainData.tsv', sep='\t')
test_df = pd.read_csv('testData.tsv', sep='\t')
test_id = test_df['id']

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

# Train and validation data
#X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'],
#                                                   test_size=0.2, random_state=1)

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
sequences = tokenz.texts_to_sequences(df['review'])  # split words and convert to int

# Pad sequences which are less than 250(Average review length)
sequences = pad_sequences(sequences, maxlen=250)

### Network layers: Embedded, LSTM, *Most likely*MaxPool, Dense, Dropout(Like CNN), Dense
model = Sequential()

# Turn input sequence into a dense vector (batch X 250 x 128)
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=250, name='embd'))
model.add(Bidirectional(LSTM(units=32, return_sequences=True)))  # batch x 250 x 64
model.add(GlobalMaxPooling1D())
model.add(Dense(units=20, activation=tf.nn.relu))
model.add(Dropout(rate=0.1))
model.add(Dense(units=1, activation=tf.nn.sigmoid))

# Configure loss and optimizer
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

res = model.fit(sequences, df['sentiment'].values,
                epochs=40,
                batch_size=128,
                verbose=1)

# Test data
test_sequence = tokenz.texts_to_sequences(test_df['review'])
test_sequence = pad_sequences(test_sequence, maxlen=250)
pred = model.predict(test_sequence)
pred = pred > 0.5
pred = [int(p) for p in pred]

sub = pd.DataFrame({'id': test_id.values,
                    'sentiment': pred})
sub.to_csv('submission.csv', index=False)
