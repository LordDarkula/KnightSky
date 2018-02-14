# -*- coding: utf-8 -*-

import random


def randomly_assign_train_test(features, labels, test_size=0.1):
    data = list(zip(list(features), list(labels)))
    random.shuffle(data)

    # Unzips data
    shuffled_features, shuffled_labels = zip(*data)

    split_index = int(len(shuffled_features) * test_size)

    train_features, test_features = shuffled_features[split_index:], shuffled_features[:split_index]
    train_labels, test_labels = shuffled_labels[split_index:], shuffled_labels[:split_index]
    return train_features, test_features, train_labels, test_labels


def next_batch(features, labels, batch_size=100):
    number_of_batches = int(len(features) / batch_size)

    for batch in range(number_of_batches):
        start_index = (batch * batch_size)
        end_index = start_index + batch_size \
            if start_index + batch_size < len(features) \
            else len(features) - 1

        yield features[start_index:end_index], labels[start_index:end_index]
