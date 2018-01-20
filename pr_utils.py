import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split




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


def get_train_features_dataset_10000():
    df = pd.read_csv('data/im_features_nist_data.csv', sep=',')

    raw_data = np.array(df)
    labels = raw_data[:, 0]
    data = raw_data[:, 1:]

    X_train = data
    y_train = labels

    return X_train, y_train


def get_train_features_dataset_100():
    df = pd.read_csv('data/im_features_nist_data.csv', sep=',')
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle data
    df = df.groupby("label")
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





def get_test_dataset_1000():
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


def get_test_features_dataset_1000():
    df = pd.read_csv('data/im_features_nist_data.csv', sep=',')
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle data
    df = df.groupby("label")

    test_data = df.apply(lambda x: x.sample(frac=0.1))

    raw_data = np.array(test_data)
    labels = raw_data[:, 0]
    data = raw_data[:, 1:]
    X_test = data
    y_test = labels

    return X_test, y_test

def experimentPCA_fulldata(classifier, data, filename=None, show_results=False):
    performance = {}
    n_features = len(data[0])
    X, y = data[:, 1:], data[:, 0]

    for c in range(1, 80):
        X_train, X_validate, y_train, y_validate = train_test_split(X, y, train_size=0.8, shuffle=True)
        pca = PCA(n_components=c)
        classifier.fit(pca.fit_transform(X_train), y_train)
        performance[c] = accuracy_score(y_validate, classifier.predict(pca.transform(X_validate)))*100
    handle_plot(performance, show_results, filename)
    return performance

def handle_plot(performance, show_results, filename):
    fig = plt.figure()
    plt.plot(performance.keys(), performance.values())
    if show_results:
        plt.show()
    if filename:
        pp = PdfPages("experiment-results/" + filename + ".pdf")
        pp.savefig(fig)
