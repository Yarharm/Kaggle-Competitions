import numpy as np
import tensorflow as tf
import os

# Process Text
with open('Hamlet.txt', 'r', encoding='utf-8') as file:
    text = file.read()
text = text[15858:]  # Clean up unnecessary info

# Convert chars to ints
chars = set(text)  # Eliminate repetitions
char_to_int = {char: i for i, char in enumerate(text)}  # char => int mapping
int_to_char = dict(enumerate(char_to_int))  # int => char mapping

text_ints = np.array([char_to_int[char] for char in text], dtype=np.int32)  # mapped ints array

# Convert text corpus into sequences of X and y
def reshape_d(sequence, batch_size, num_steps):
    """
    Reshape data into batches of sequences
    :param sequence: 1-D numpy array of integers (mapped text characters)
    :param batch_size: int, number of rows in reshaped data
    :param num_steps: int, number of columns for individual batch
    :return: Two 2-d arrays X and y, shape {batch_size, batch_size * num_steps)
    """
    batch_length = batch_size * num_steps
    num_batches = sequence // batch_size
    if num_batches * batch_length > (len(sequence) - 1):
        num_batches -= 1
    # Round up batch
    X = sequence[: num_batches * batch_length]
    y = sequence[1: num_batches * batch_length + 1]
    X_splits = np.split(X, batch_size)
    y_splits = np.split(y, batch_size)
    # Stack batches
    X = np.stack(X_splits)
    y = np.stack(y_splits)
    return X, y

def batch_generator(X, y, num_steps):
    """
    Split reshaped data to individual batches
    :param X: 2-D array of integers, shape {batch_size, batch_size * num_steps)
    :param y: 2-D array of integers, shape {batch_size, batch_size * num_steps)
    :param num_steps: int, number of columns for individual batch
    :return: Tuple of individual batches for arrays X and y
    """
    batch_size, batch_length = X.shape
    num_batches = batch_length // num_steps
    for i in range(num_batches):
        yield (X[:, i*num_steps:(i+1)*num_steps],
               y[:, i*num_steps:(i+1)*num_steps])

## RNN model
class RNN(object):
    def __init__(self, num_classes, batch_size=64,
                 num_steps=100, lstm_size=128,
                 num_layers=1, learning_rate=0.001,
                 keep_prob=0.5, grad_clip=5,
                 sampling=False):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_steps = num_steps,
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob  # Dropout layer
        self.grad_clip = grad_clip  # Clipping the gradient (Exploding Gradient Problem)

        self.g = tf.Graph()
        with self.g.as_default():
            self.build(sampling=sampling)  # Two diff graphs for sampling/training
            self.init_op = tf.global_variables_initializer()

    def build(self, sampling):
        """
        Sampling mode: batch_size = 1; num_steps = 1
        Training mode: batch_size = self.batch_size; num_steps = self.num_steps
        :param sampling: Bool, True: perform Sampling mode; False: perform Training mode
        :return: build multilayer RNN
        """
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size = self.batch_size
            num_steps = self.num_steps

        # Define placeholders
        tf_x = tf.placeholder(tf.int32,
                              shape=[batch_size, num_steps],
                              name='tf_x')
        tf_y = tf.placeholder(tf.int32,
                              shape=[batch_size, num_steps],
                              name='tf_y')
        tf_keep_proba = tf.placeholder(tf.float32,
                                       name='tf_keep_proba')

        # Encode features (One-hot):
        x_onehot = tf.one_hot(tf_x, depth=self.num_classes)
        y_onehot = tf.one_hot(tf_y, depth=self.num_classes)

        # Build RNN [cell:LSTM, wrapped in Dropout layer]
        cells = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(self.lstm_size),
                output_keep_prob = tf_keep_proba)
                for _ in range(self.num_layers)])

        # Def state
        self.init_state = cells.zero_state(
            batch_size, tf.float32
        )

        # Run individual sequence through the RNN
        # Get LSTM results
        lstm_outputs, self.final_state = tf.nn.dynamic_rnn(
            cells, x_onehot,
            initial_state=self.init_state
        )

        ### TO DO:
        # Probably need to reshape results (?????)
        # Get logits/ call softmax
        # Def Cost func and Optimizer
        # GOOglE GRAD CLIPPING

    def train(self):
        pass

#    ????
#    def predict(self):
#        pass