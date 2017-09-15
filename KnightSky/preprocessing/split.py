# -*- coding: utf-8 -*-

import random


def randomly_assign_train_test(X_data, y_data, test_size=0.1):
    data = list(zip(list(X_data), list(y_data)))
    random.shuffle(data)

    shuffled_X, shuffled_y = zip(*data)

    X_split_index = int(len(shuffled_X) * test_size)
    y_split_index = int(len(shuffled_y) * test_size)

    X_train, X_test = shuffled_X[X_split_index:], shuffled_X[:X_split_index]
    y_train, y_test = shuffled_y[y_split_index:], shuffled_y[:y_split_index:]
    return X_train, X_test, y_train, y_test


def next_batch(X, y, batch_size=100):
    n_batches = int(len(X) / batch_size)

    for batch in range(n_batches):
        start = (batch * batch_size)
        end = start + batch_size if start + batch_size < len(X) else len(X) - 1
        yield X[start:end], y[start:end]
