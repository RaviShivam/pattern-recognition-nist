import numpy as np
import pandas as pd


def get_train_dataset_100():
    df = pd.read_csv('data/processed_nist_data.csv', sep=',', header=None)
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle data
    df = df.groupby(0)
    train_data = df.apply(lambda x: x.sample(frac=0.01))

    raw_data = np.array(train_data)
    labels = raw_data[:, 0]
    data = raw_data[:, 1:]

    X_train = data
    y_train = labels

    return X_train, y_train


def get_train_dataset_10000():
    df = pd.read_csv('data/processed_nist_data.csv', sep=',', header=None)

    raw_data = np.array(df)
    labels = raw_data[:, 0]
    data = raw_data[:, 1:]

    X_train = data
    y_train = labels

    return X_train, y_train


def get_test_dataset_100():
    df = pd.read_csv('data/processed_nist_data.csv', sep=',', header=None)
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle data
    df = df.groupby(0)

    test_data = df.apply(lambda x: x.sample(frac=0.1))

    raw_data = np.array(test_data)
    labels = raw_data[:, 0]
    data = raw_data[:, 1:]
    X_test = data
    y_test = labels

    return X_test, y_test
