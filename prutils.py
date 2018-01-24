import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing as mp

from multiprocessing import Pool
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from functools import partial

"""
Dataset constants
"""
# train data
RAW_PIXELS_DATASET = "data/processed_nist_data.csv"
IM_FEATURES_DATASET= "data/im_features_nist_data.csv"

# test data
RAW_PIXELS_TEST= "data/preprocessed_test_nist_data.csv"
IM_FEATURES_TEST= ""

"""
Dataset readers
"""
def get_full_data(dataframe, split_validation=True):
    if type(dataframe) is not pd.core.frame.DataFrame:
        dataframe = pd.read_csv(dataframe)
    df = dataframe.as_matrix()
    if split_validation:
        return train_test_split(df[:, 1:], df[:, 0], test_size=0.2)
    else:
        return df[:, 1:], df[:, 0]

def get_random_batch(dataframe, split_validation=True, frac=0.01):
    if type(dataframe) is not pd.core.frame.DataFrame:
        dataframe = pd.read_csv(dataframe)
    dataframe = dataframe.groupby("label")
    trainset = dataframe.apply(lambda x: x.sample(frac=frac)).as_matrix()
    validateset = dataframe.apply(lambda x: x.sample(frac=0.1)).as_matrix()

    X_train, y_train = trainset[:, 1:], trainset[:, 0]
    X_validate, y_validate = validateset[:, 1:], validateset[:, 0]
    if split_validation:
        return X_train, X_validate, y_train, y_validate
    else:
        return X_train, y_train



"""
Classifier performance estimator
"""
def estimate_classifier_performance_transform(classifier, dataframe, transformer):
    X, y = get_full_data(dataframe, split_validation=False)
    return estimate_classifier_performance(classifier, transformer.transform(X), y)

def estimate_classifier_performance_normal(classifier, dataframe):
    X, y = get_full_data(dataframe, split_validation=False)
    return estimate_classifier_performance(classifier, X, y)

def estimate_classifier_performance(classifier, X_test, y_test):
    return accuracy_score(classifier.predict(X_test), y_test) * 100

"""
Running PCA experiments
"""
def _single_PCA(n, classifier, dataframe, batch):
    if batch:
        p = 0
        for _ in range(100):
            data = get_random_batch(dataframe)
            pca = PCA(n_components=n).fit(data[0], data[2])
            p += estimate_classifier_performance(classifier.fit(pca.transform(data[0]), data[2]), pca.transform(data[1]), data[3])
        p = p/100.0
    else:
        data = get_full_data(dataframe)
        pca = PCA(n_components=n).fit(data[0], data[2])
        p = estimate_classifier_performance(classifier.fit(pca.transform(data[0]), data[2]), pca.transform(data[1]), data[3])
    return (n, p)

def _run_PCA_auto(classifier, dataframe, batch):
    if batch:
        data = get_random_batch(dataframe)
    else:
        data = get_full_data(dataframe)
    pca = PCA().fit(data[0])
    optimal_components = np.argwhere(np.cumsum(pca.explained_variance_ratio_) > 0.9).flatten()[0]
    return _single_PCA(optimal_components, classifier, dataframe, batch)

def run_PCA_experiment(classifier, data_file, max_components = 40, batch=False, save_to_file=False, show_results=False):
    dataframe = pd.read_csv(data_file)
    if max_components is 'auto':
        return _run_PCA_auto(classifier, dataframe, batch)
    single_run = partial(_single_PCA, classifier=classifier, dataframe=dataframe, batch=batch)
    pool = Pool(mp.cpu_count())
    performance = dict(pool.map(single_run, range(1, max_components)))
    pool.close()
    pool.join()
    handle_plot(performance, show_results, save_to_file)
    return performance


"""
Code for running ICAs
"""
def _single_ICA(n, classifier, dataframe, batch):
    if batch:
        p = 0
        for _ in range(100):
            data = get_random_batch(dataframe)
            ica = FastICA(n_components=n, max_iter=1000).fit(data[0], data[2])
            p += estimate_classifier_performance(classifier.fit(ica.transform(data[0]), data[2]), ica.transform(data[1]), data[3])
        p = p/100.0
    else:
        data = get_full_data(dataframe)
        ica = FastICA(n_components=n, max_iter=1000).fit(data[0], data[2])
        p = estimate_classifier_performance(classifier.fit(ica.transform(data[0]), data[2]), ica.transform(data[1]), data[3])
    return (n, p)

def run_ICA_experiment(classifier, data_file, max_components = 20, batch=False,  show_results=False, save_to_file=False):
    dataframe = pd.read_csv(data_file)
    pool = Pool(mp.cpu_count())
    single_run = partial(_single_ICA, classifier=classifier, dataframe=dataframe, batch=batch)
    performance = dict(pool.map(single_run, range(1, max_components)))
    pool.close()
    pool.join()
    handle_plot(performance, show_results, save_to_file)
    return performance

"""
Code for running KernelPCA
"""
def _single_KPCA(n, classifier, dataframe, batch):
    if batch:
        p = 0
        for _ in range(100):
            data = get_random_batch(dataframe)
            kpca = KernelPCA(n_components=n).fit(data[0], data[2])
            p += estimate_classifier_performance(classifier.fit(kpca.transform(data[0]), data[2]), kpca.transform(data[1]), data[3])
        p = p/100.0
    else:
        data = get_full_data(dataframe)
        kpca = KernelPCA(n_components=n).fit(data[0], data[2])
        p = estimate_classifier_performance(classifier.fit(kpca.transform(data[0]), data[2]), kpca.transform(data[1]), data[3])
    return (n, p)

def run_KPCA_experiment(classifier, data_file, max_components = 20, batch=False,  show_results=False, save_to_file=False):
    dataframe = pd.read_csv(data_file)
    performance = []
    single_run = partial(_single_KPCA, classifier=classifier, dataframe=dataframe, batch=batch)
    for c in range(1, max_components):
        performance.append(single_run(c))
    performance = dict(performance)
    handle_plot(performance, show_results, save_to_file)
    return performance

"""
Handling experiment results
"""
def handle_plot(performance, show_results, save_to_file):
    fig = plt.figure()
    plt.title('Number of Components Retained vs Performance')
    plt.xlabel('Number of Components')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)

    plt.plot(performance.keys(), performance.values())
    if show_results:
        plt.show()
    if save_to_file:
        pp = PdfPages("experiment-results/" + save_to_file + ".pdf")
        pp.savefig(fig)
        pp.close()
