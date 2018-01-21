import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
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


def get_raw_pixels(full_data=True):
    df = pd.read_csv('data/processed_nist_data.csv', sep=',', header=None)
    if full_data:
        df = df.as_matrix()
        X, y = df[:, 1:], df[:, 0]
        X_train, X_validate, y_train, y_validate = train_test_split(X, y, train_size=0.8, shuffle=True)
    else:
        df = df.sample(frac=1).reset_index(drop=True)  # shuffle data
        df = df.groupby(0)
        df100 = df.apply(lambda x: x.sample(frac=0.01)).as_matrix()
        X_train, y_train = df100[:, 1:], df100[:, 0]
        df1000 = df.apply(lambda x: x.sample(frac=0.1)).as_matrix()
        X_validate, y_validate = df1000[:, 1:], df1000[:, 0]

    return X_train, X_validate, y_train, y_validate


def experimentPCA_fulldata(classifier, full_data=True, filename=None, show_results=False, n_comp_auto=False):
    performance = {}

    if full_data:
        X_train, X_validate, y_train, y_validate = get_raw_pixels(full_data=True)
    else:
        X_train, X_validate, y_train, y_validate = get_raw_pixels(full_data=False)

    if not n_comp_auto:
        for n_comp in range(1, 30):
            print("processing c=", n_comp)
            pca = PCA(n_components=n_comp)
            classifier.fit(pca.fit_transform(X_train), y_train)
            performance[n_comp] = accuracy_score(y_validate, classifier.predict(pca.transform(X_validate))) * 100
        handle_plot(performance, show_results, filename)
    else:
        pca = PCA()
        pca.fit(X_train)
        variance = pca.explained_variance_
        n_comp = max(np.argwhere(variance > 0.9))[0]
        pca.n_components = n_comp
        classifier.fit(pca.fit_transform(X_train), y_train)
        performance[0] = accuracy_score(y_validate, classifier.predict(pca.transform(X_validate))) * 100

    return performance, n_comp


def handle_plot(performance, show_results, filename):
    fig = plt.figure()
    plt.title('Number of Components Retained vs Performance')
    plt.xlabel('Number of Components')
    plt.ylabel('Accuracy (%)')

    plt.plot(performance.keys(), performance.values())
    if show_results:
        plt.show()
    if filename:
        pp = PdfPages("experiment-results/" + filename + ".pdf")
        pp.savefig(fig)
        pp.close()


def to_csv(name, x):
    df = pd.DataFrame(x)  # 1st row as the column names
    df.to_csv(name, encoding='utf-8', index=False, sep=',', header=None)
