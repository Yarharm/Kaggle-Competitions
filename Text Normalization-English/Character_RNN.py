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

def get_top_char(probas, char_size, top_n=5):
    """
    Get character based on obtained probabilities
    :param probas:
    :param char_size:
    :param top_n:
    :return:
    """
    p = np.squeeze(probas)
    p[np.argsort(p)[:-top_n]] = 0.0
    p = p / np.sum(p)
    ch_id = np.random.choice(char_size, 1, p=p)[0]
    return ch_id
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

        seq_output_reshaped = tf.reahape(lstm_outputs,
                                         shape=[-1, self.lstm_size],
                                         name='seq_output_reshaped')

        logits = tf.layers.Dense(inputs=seq_output_reshaped,
                                 units=self.num_classes,
                                 activation=None,
                                 name='logits')

        proba = tf.nn.softmax(logits,
                              name='probabilities')
        y_reshaped = tf.reshape(y_onehot,
                                shape=[-1, self.num_classes],
                                name='y_reshaped')

        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped),
            name='cost')

        # Gradient clipping
        tvars = tf.trainable.variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(cost, tvars),
            self.grad_clip
        )

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            name='train_op'
        )
    def train(self, train_x, train_y, num_epochs):
        with tf.Session(graph=self.g) as session:
            session.run(self.init_op)

            n_batches = train_x.shape[1] // self.num_steps
            iterations = n_batches * num_epochs
            for epoch in range(num_epochs):
                new_state = session.run(self.init_state)
                loss = 0
                # Batch gen
                bgen = batch_generator(train_x, train_y, self.num_steps)
                for b, (batch_x, batch_y) in enumerate(bgen, 1):
                    iteration = epoch * n_batches + b

                    feed = {'tf_x:0': batch_x,
                            'tf_y:0': batch_y,
                            'tf_keep_proba:0': self.keep_prob,
                            self.init_state: new_state}
                    batch_cost, _, new_state = session.run(
                        ['cost:0', 'train_op', self.final_state],
                        feed_dict=feed
                    )

    def sample(self, output_length, starter_seq='The ',):
        """
        Calculate probabily of the next character based on the observed equence
        :param output_length: Observed sequence
        :param starter_seq:
        :return:
        """
        observed_seq = [ch for ch in starter_seq]
        with tf.Sessions(graph=self.g) as session:
            new_state = session.run(self.initial_state)
            # Run on starter sequence
            for ch in starter_seq:
                x = np.zeros((1, 1))
                x[0, 0] = char_to_int[ch]
                feed = {'tf_x:0': x,
                        'tf_keep_proba:0': 1.0,
                        self.init_state: new_state}
                proba, new_state = session.run(
                    ['probabilities:0', self.final_state],
                    feed_dict=feed
                )

            ch_id = get_top_char(proba, len(chars))
            observed_seq.append(int_to_char[ch_id])
            # Run Sampling on a new sequence
            for i in range(output_length):
                x[0,0] = ch_id
                feed = {'tf_x:0': x,
                        'tf_keep_proba:0': 1.0,
                        self.init_state: new_state}
                proba, new_state = session.run(
                    ['probabilities:0', self.final_state],
                    feed_dict = feed
                )

                ch_id = get_top_char(proba, len(chars))
                observed_seq.update(int_to_char(ch_id))
        return ''.join(observed_seq)