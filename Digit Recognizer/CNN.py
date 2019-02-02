import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Train data
train_data = pd.read_csv('Kaggle_MNIST/train.csv', low_memory=False)
train_data = train_data.to_numpy('float32')
X_train, y_train = train_data[:, 1:], train_data[:, 0]
X_train = X_train / 255.0
# Get validation set (12%)
#X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
#                                                      test_size=0.12, random_state=1)

# Test data
test_data = pd.read_csv('Kaggle_MNIST/test.csv', low_memory=True)
X_test = test_data.to_numpy('float32')
X_test = X_test / 255.0
test_ID = [(x + 1) for x in range(X_test.shape[0])]
#print(X_train.shape)
#print(y_train.shape)
#print(X_valid.shape)
#print(y_valid.shape)
#print(X_test.shape)

# Normalization
mean_val = np.mean(X_train, axis=0)  # mean for each feature
std_val = np.std(X_train)  # std for the whole set
X_train_norm = (X_train - mean_val)/std_val
#X_valid_norm = (X_valid - mean_val)/std_val
X_test_norm = (X_test - mean_val)/std_val

del X_train, X_test
# Mini-batches generator
def batch_gen(X, y, batch_size=64,
              shuffle=False, random_seed=None):
    index = np.arange(y.shape[0])

    if shuffle:
        data = np.column_stack((X, y))
        np.random.shuffle(data)
        X = data[:, :-1]
        y = data[:, -1].astype(int)

    for i in range(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size, :], y[i:i + batch_size])

# Low-Level API
# Convolutional layer
def conv_layer(input_tensor, name,
               kernel_size, n_output_channels,
               padding_mode='SAME', strides=(1, 1, 1, 1)):
    """
    Build convolutional layer
    :param input_tensor: Tensor given as an input to the layer
    :param name: Name of the layers, which is used as a scope
    :param kernel_size: Dimensions of the Kernel(filter)
    :param n_output_channels: Number of the feature maps
    :param padding_mode: Padding of the input image ('SAME' => Pooling, 'VALID' => Conv_layer)
    :param strides: tuple of rank 4 for sliding filter
    :return: fully build 2d convolutional layer
    """
    with tf.variable_scope(name):
        # input_tensor shape: {batch x width x height x channels}
        input_shape = input_tensor.get_shape().as_list()
        n_input_channels = input_shape[-1]

        # Kernel of shape [batch, in_height, in_width, in_channels]
        weights_shape = list(kernel_size) + \
            [n_input_channels, n_output_channels]

        # print(weights_shape)

        filter = tf.get_variable(name='_filter',
                                 shape=weights_shape)  # Xavier by default
        # bias
        biases = tf.get_variable(name='_biases',
                                 initializer=tf.zeros(
                                     shape=[n_output_channels]))
        # Conv layer
        conv = tf.nn.conv2d(input=input_tensor, filter=filter,
                            strides=strides, padding=padding_mode)
        # Add bias
        conv = tf.nn.bias_add(conv, biases, name='pre_activation')

        # Apply ReLU
        conv = tf.nn.relu(conv, name='activation')
        return conv

# Fully connected layer
def fc_layer(input_tensor, name,
             output_units, activation_func=None):
    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()[1:]  # [1, 14, 14, 64]

        # Numb of input features
        input_ftrs = np.prod(input_shape)  # [14 * 14 * 64]

        # reshape rank 1+ tensor to 2d array
        if len(input_shape) > 1:
            input_tensor = tf.reshape(input_tensor, shape=(-1, input_ftrs))

        # Init weights
        weights_shape = [input_ftrs, output_units]

        weights = tf.get_variable(name='_weights', shape=weights_shape)

        # Bias
        biases = tf.get_variable(name='_biases',
                                 initializer=tf.zeros(shape=[output_units]))

        # Propagate
        layer = tf.matmul(input_tensor, weights)  # batch x n_features * n_features x n_output

        # Add bias
        layer = tf.nn.bias_add(layer, biases, name='pre_activation')

        # Apply activation
        if(activation_func is None):
            return layer

        layer = activation_func(layer, name='activation')
        return layer

# Build network
def cnn(learning_rate=0.0001):
    ## Input data and labels
    tf_x = tf.placeholder(tf.float32, shape=[None, 784],
                          name='tf_x')
    tf_y = tf.placeholder(tf.int32, shape=[None],
                          name='tf_y')

    # Reshape 784 features to the rank 4 tensor [bathc_size x 28 x 28 x 1]
    tf_x_image = tf.reshape(tf_x, shape=[-1, 28, 28, 1], name='tf_x_prop_shape')

    # One hot encode labels
    tf_y_encode = tf.one_hot(indices=tf_y, depth=10,
                             dtype=tf.float32,
                             name='tf_y_encoded')

    # First layer (Convolutional) Out_shape = [batch_size x 24 x 24 x 32]
    l1 = conv_layer(tf_x_image, name='conv_1',
                    kernel_size=(5, 5), padding_mode='VALID',  # test 'SAME' padding
                    n_output_channels=32)

    # Second layer (Subsampling => Max Pooling) Out_shape = [batch_size x 12 x 12 x 32]
    # 2 x 2 decreases height and width by 50%
    l2 = tf.nn.max_pool(l1, ksize=[1, 2, 2, 1],  # [2 x 2 Pool size]
                        strides=[1, 2, 2, 1],  # X and Y axis move is 2
                        padding='SAME')

    # Third layer (Convolutional) Out_shape = [batch_size x 8 x 8 x 64]
    l3 = conv_layer(l2, name='conv_2',
                    kernel_size=(5, 5),
                    padding_mode='VALID',
                    n_output_channels=64)

    # Fourth layer (Subsampling => Max Pooling) Out_shape = [batch_size x 4 x 4 x 64]
    l4 = tf.nn.max_pool(l3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')

    # Fifth layer (Fully Connected) Out_shape = [batch_size x 1024]
    l5 = fc_layer(l4, name='fc_1',
                  output_units=1024,
                  activation_func=tf.nn.relu)

    # Dropout layer (Disable percentage of activation) (ONLY FOR TRAINING PHASE)
    keep_prob = tf.placeholder(tf.float32, name='fc_keep_prob')
    l6 = tf.nn.dropout(l5, keep_prob=keep_prob,
                       name='dropout_layer')

    # Final layer (Fully Connected) Out_shape = [batch_size x 10]
    l7 = fc_layer(l6, name='fc_2',
                  output_units=10,
                  activation_func=None)  # Default Linear is used

    # Predictions
    predictions = {
        'probabilities': tf.nn.softmax(l7, name='probabilities'),
        'labels': tf.cast(tf.argmax(l7, axis=1), tf.int32,
                          name='labels')}

    # Loss functions
    # Cross-entropy for calssification problem
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=l7,
                                                labels=tf_y_encode),
        name='cross_entropy_loss')

    # Optimization
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = optimizer.minimize(cross_entropy_loss, name='train_op')

# Train
def train(session, X_train, y_train, validation_set=None,
          initialize=True, epochs=20, shuffle=True,
          dropout=0.4, random_seed=None):

    X_data = np.array(X_train)
    y_data = np.array(y_train)

    # Init glob vars
    if initialize:
        session.run(tf.global_variables_initializer())

    # Random see for Batch_gen
    np.random.seed(1)
    for epoch in range(1, epochs + 1):
        batch_g = batch_gen(
            X_data, y_data,
            shuffle=shuffle)

        for i, (batch_x, batch_y) in enumerate(batch_g):
            feed = {'tf_x:0': batch_x,
                    'tf_y:0': batch_y,
                    'fc_keep_prob:0': dropout}
            loss, _ = session.run(
                ['cross_entropy_loss:0', 'train_op'],
                feed_dict=feed)

# Predict
def predict(session, X_test, return_proba=False):
    feed = {'tf_x:0': X_test,
            'fc_keep_prob:0': 1.0}
    if return_proba:
        return session.run('probabilities:0', feed_dict=feed)
    else:
        return session.run('labels:0', feed_dict=feed)

# Graph object
g = tf.Graph()

with g.as_default():
    cnn()

# Session
with tf.Session(graph=g) as session:
    train(session, X_train=X_train_norm, y_train=y_train,
          initialize=True, random_seed=123)

    preds = predict(session, X_test=X_test_norm,
                    return_proba=False)

#sub = pd.DataFrame()
#sub['ImageId'] = test_ID
#sub['Label'] = preds
#sub.to_csv('submission.csv', index=False)
#print(sub.shape)
