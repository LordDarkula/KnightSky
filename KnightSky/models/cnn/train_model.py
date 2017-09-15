# -*- coding: utf-8 -*-

import tensorflow as tf

from KnightSky.preprocessing.split import randomly_assign_train_test, next_batch


BOARD_SIZE = 64
LENGTH = 8
NUMBER_OF_PLAYERS = 2

X_placeholder = tf.placeholder(tf.float32, [None, BOARD_SIZE], name="X")
y_placeholder = tf.placeholder(tf.float32, [None, NUMBER_OF_PLAYERS], name="y")
keep_prob_placeholder = tf.placeholder(tf.float32)

LRNING_RATE = 0.005
TRAIN_KEEP_PROB = 0.9
TEST_KEEP_PROB = 1

BATCH_SIZE = 100
NUMBER_OF_EPOCHS = 300

TENSORBOARD_DIR = "tmp/final/3"


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
        activation = tf.nn.relu(tf.matmul(X, W) + b)

        tf.summary.histogram('weights', W)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activation', activation)

        return activation


def create_model():
    X = tf.reshape(X_placeholder, [-1, LENGTH, LENGTH, 1])

    W_conv1 = weight_variable([6, 6, 1, 32])
    b_conv1 = bias_variable([32])

    model = conv_layer(X, W_conv1, b_conv1, name='conv1')

    W_conv2 = weight_variable([2, 2, 32, 64])
    b_conv2 = bias_variable([64])

    model = conv_layer(model, W_conv2, b_conv2, name='conv2')

    model = tf.reshape(model, [-1, 2*2*BOARD_SIZE])

    w1 = weight_variable([2*2*BOARD_SIZE, 1024])
    b1 = bias_variable([1024])

    model = fc_layer(model, w1, b1, name='fc1')

    model = tf.nn.dropout(model, keep_prob_placeholder)

    w_out = weight_variable([1024, 2])
    b_out = bias_variable([2])

    y_predicted = tf.matmul(model, w_out) + b_out

    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_placeholder, logits=y_predicted))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(LRNING_RATE).minimize(cross_entropy)

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

        for epoch in range(NUMBER_OF_EPOCHS):

            for i, (batch_X, batch_y) in enumerate(next_batch(X_train, y_train, batch_size=BATCH_SIZE)):

                sess.run(optimizer, feed_dict={X_placeholder: batch_X,
                                               y_placeholder: batch_y,
                                               keep_prob_placeholder: TRAIN_KEEP_PROB})

                s = sess.run(merged_summary, feed_dict={X_placeholder: batch_X,
                                                        y_placeholder: batch_y,
                                                        keep_prob_placeholder: TRAIN_KEEP_PROB})
                writer.add_summary(s, i)

            print("Train accuracy {}".format(accuracy.eval({X_placeholder: X_train,
                                                            y_placeholder: y_train,
                                                            keep_prob_placeholder: TRAIN_KEEP_PROB})))
            print("Test accuracy {}".format(accuracy.eval({X_placeholder: X_test,
                                                           y_placeholder: y_test,
                                                           keep_prob_placeholder: TEST_KEEP_PROB})))

        print("Final Test accuracy {}".format(accuracy.eval({X_placeholder: X_test,
                                                             y_placeholder: y_test,
                                                             keep_prob_placeholder: TEST_KEEP_PROB})))


