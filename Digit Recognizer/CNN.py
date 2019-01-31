import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Train data
train_data = pd.read_csv('train.csv', low_memory=False)
train_data = train_data.to_numpy('float32')
X_train, y_train = train_data[:, 1:], train_data[:, 0]
X_train = X_train / 255.0 * 0.99 + 0.01

# Get validation set (12%)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                      test_size=0.12, random_state=1)

# Test data
test_data = pd.read_csv('test.csv', low_memory=True)
X_test = test_data.values.to_numpy('float32')
X_test = X_test / 255.0 * 0.99 + 0.01

#print(X_train.shape)
#print(y_train.shape)
#print(X_valid.shape)
#print(y_valid.shape)
#print(X_test.shape)

# Normalization
mean_val = np.mean(X_train, axis=0)  # mean for each feature
std_val = np.std(X_train)  # std for the whole set
X_train_norm = (X_train - mean_val)/std_val
X_valid_norm = (X_valid - mean_val)/std_val
X_test_norm = (X_test - mean_val)/std_val

# Mini-batches generator
def batch_get(X, y, batch_size=64,
              shuffle=False, random_seed=None):
    index = np.arange(y.shape[0])  # [0 ... 10000)
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(index)
        X = X[index]
        y = y[index]

    for i in range(0, X.shape[0], batch_size):
        yield (X[i:i+batch_size, :], y[i:i+batch_size])

# Low-Level API
def conv_layer(input_tensor, name,
               kernel_size, n_output_channels,
               padding_mode='SAME', strides=(1, 1, 1, 1)):  # 'SAME' padding/ strides RANK 4???????
    with tf.variable_scope(name):
        # input_tensor shape: {batch x width x height x channels_in}
        input_shape = input_tensor.get_shape().as_list()
        n_input_channels = input_tensor[-1]

        weights_shape = list(kernel_size) + \
            [n_input_channels, n_output_channels]  # Probably 5x5 Kernel!!!!!!!!!!!!
        # print(weights_shape)

        weights = tf.get_variable(name='_weights',
                                  shape=weights_shape)
        print(weights)

        # bias
        biases = tf.get_variable(name='_biases',
                                 initializer=tf.zeros(
                                     shape=[n_output_channels]))
        print(biases)

        ### TO DO
        # APPLY CONVOLUTE ONTO INPUT DATA
        # ADD BIAS
        # APPLY ACTIVATION FUNCTION ON LOGITS (Relu/tanh????????????)