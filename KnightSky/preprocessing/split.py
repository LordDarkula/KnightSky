# -*- coding: utf-8 -*-

import random
import numpy as np


def randomly_assign_train_test(features, labels, test_size=0.1):
    """
    Splits up data into train and test subsets.

    :param features: 2D array storing features
    :type: features: list or np.array

    :param labels: 1D array storing labels
    :type: labels: list or np.array

    :param test_size: proportion of the data to be used for testing
    :type: test_size: float

    :return: training features, training labels, testing features, testing labels
    :rtype:  tuple containing np.arrays
    """
    data = list(zip(list(features), list(labels)))
    random.shuffle(data)

    # Unzips data
    shuffled_features, shuffled_labels = zip(*data)

    split_index = int(len(shuffled_features) * test_size)

    train_features, test_features = shuffled_features[split_index:], shuffled_features[:split_index]
    train_labels, test_labels = shuffled_labels[split_index:], shuffled_labels[:split_index]
    return np.array(train_features), np.array(test_features), np.array(train_labels), np.array(test_labels)


def next_batch(features, labels, batch_size=100):
    """
    Splits data into batches and yields it one batch at a time.

    :param features: 2D array storing features
    :type: features: list or np.array

    :param labels: 1D array storing labels
    :type: labels: list or np.array

    :param batch_size: number of data points in each batch
    :type: batch_size: int

    :return: batch
    :rtype:  list or np.array
    """
    number_of_batches = int(len(features) / batch_size)

    for batch in range(number_of_batches):
        start_index = (batch * batch_size)
        end_index = start_index + batch_size \
            if start_index + batch_size < len(features) \
            else len(features) - 1

        yield features[start_index:end_index], labels[start_index:end_index]
