import numpy as np
import tensorflow as tf

from KnightSky.process import randomly_assign_train_test, next_batch


BOARD_SIZE = 64
LENGTH = 8
NUMBER_OF_PLAYERS = 2

X_placeholder = tf.placeholder(tf.float32, [None, BOARD_SIZE], name="X")
y_placeholder = tf.placeholder(tf.float32, [None, NUMBER_OF_PLAYERS], name="y")

LRNING_RATE = 0.00005

TENSORBOARD_DIR = "tmp/knight/5"


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='W')


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='B')


def conv_layer(X, W, b, name='conv'):
    with tf.name_scope(name):
        convolution = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
        activation = tf.nn.relu(convolution + b)

        tf.summary.histogram('weights', W)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activation', activation)

        return tf.nn.max_pool(activation, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


def fc_layer(X, W, b, name='fc'):
    with tf.name_scope(name):
        return tf.nn.relu(tf.matmul(X, W) + b)


def create_model():
    X = tf.reshape(X_placeholder, [-1, LENGTH, LENGTH, 1])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    model = conv_layer(X, W_conv1, b_conv1, name='conv1')

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    model = conv_layer(model, W_conv2, b_conv2, name='conv2')

    model = tf.reshape(model, [-1, 2*2*BOARD_SIZE])

    w1 = weight_variable([2*2*BOARD_SIZE, 200])
    b1 = bias_variable([200])

    model = fc_layer(model, w1, b1, name='fc1')

    w2 = weight_variable([200, 500])
    b2 = bias_variable([500])

    model = fc_layer(model, w2, b2, name='fc2')

    w3 = weight_variable([500, 200])
    b3 = bias_variable([200])

    model = fc_layer(model, w3, b3, name='fc3')

    w3 = weight_variable([200, NUMBER_OF_PLAYERS])
    b3 = bias_variable([NUMBER_OF_PLAYERS])

    y_predicted = tf.matmul(model, w3) + b3

    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_placeholder, logits=y_predicted))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(0.5).minimize(cross_entropy)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y_placeholder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    return optimizer, accuracy


def run_model(optimizer, accuracy, bitmap_X, bitmap_y):
    X_train, X_test, y_train, y_test = randomly_assign_train_test(bitmap_X, bitmap_y)
    print("training x {} training y {}".format(len(X_train), len(y_train)))
    print("testing x {} testing y {}".format(len(X_test), len(y_test)))

    with tf.Session() as sess:
        print("Session starting")

        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(TENSORBOARD_DIR)
        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())

        for epoch in range(10000):

            for i, (batch_X, batch_y) in enumerate(next_batch(X_train, y_train)):
                print('batch number: {}'.format(i))

                sess.run(optimizer, feed_dict={X_placeholder: batch_X, y_placeholder: batch_y})

                s = sess.run(merged_summary, feed_dict={X_placeholder: batch_X, y_placeholder: batch_y})
                writer.add_summary(s, i)

            print("Train accuracy {}".format(accuracy.eval({X_placeholder: X_train, y_placeholder: y_train})))
            print("Test accuracy {}".format(accuracy.eval({X_placeholder: X_test, y_placeholder: y_test})))

        print("Final Test accuracy {}".format(accuracy.eval({X_placeholder: X_test, y_placeholder: y_test})))
