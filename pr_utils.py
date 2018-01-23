import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def get_full_data(dataframe):
    if type(dataframe) is not pd.core.frame.DataFrame:
        dataframe = pd.read_csv(dataframe)
    df = dataframe.as_matrix()
    return train_test_split(df[:, 1:], df[:, 0], test_size=0.2)

def get_random_batch(dataframe, frac=0.01):
    if type(dataframe) is not pd.core.frame.DataFrame:
        dataframe = pd.read_csv(dataframe)
    dataframe = dataframe.groupby("label")
    trainset = dataframe.apply(lambda x: x.sample(frac=frac)).as_matrix()
    validateset = dataframe.apply(lambda x: x.sample(frac=0.1)).as_matrix()

    X_train, y_train = trainset[:, 1:], trainset[:, 0]
    X_validate, y_validate = validateset[:, 1:], validateset[:, 0]
    return X_train, X_validate, y_train, y_validate


def estimate_classifier_performance(classifier, X_test, y_test):
    return accuracy_score(classifier.predict(X_test), y_test) * 100
#
# def _experimentPCA_batch(classifier, data_file):
#     dataframe = pd.read_csv(data_file)
#     p = []
#     for _ in range(100):
#         pca = PCA()
#         train_set, test_set = get_random_batch(dataframe)
#         pca.fit(train_set[:, 1:])
#         optimal_components = np.argwhere(np.cumsum(pca.explained_variance_ratio_) > 0.9).flatten()[0]
#
# def _experimentPCA_full(classifier, data_file):
#     dataframe = pd.read_csv(data_file)
#     X_train, y_train, X_validate, y_validate = get_full_data(dataframe)
#     pca = PCA().fit(X_train)
#     optimal_components = np.argwhere(np.cumsum(pca.explained_variance_ratio_) > 0.9).flatten()[0]
#
#
# def run_PCA_experiments(classifier, data_file, batch=False, n_components='auto')
#     dataframe = pd.read_csv(data_file)
#     if batch:
#         X_train, X_validate, y_train, y_validate = get_full_data(dataframe)
#     else:
#         X_train, X_validate, y_train, y_validate = get_random_batch(dataframe)
#
#     performance = {}
#     if n_components is 'auto':
#         pca = PCA().fit(X_train)
#         optimal_components = np.argwhere(np.cumsum(pca.explained_variance_ratio_) > 0.9).flatten()[0]
#         classifier.fit(pca.fit_transform(X_train), y_train)
#         performance[optimal_components] =
#     else:
#         for i in range(1, n_components)
#
#
#
#
# def experimentPCA_fulldata(classifier, full_data=True, filename=None, show_results=False, n_comp_auto=False):
#     performance = {}
#
#
#     if not n_comp_auto:
#         for n_comp in range(1, 30):
#             print("processing c=", n_comp)
#             pca = PCA(n_components=n_comp)
#             classifier.fit(pca.fit_transform(X_train), y_train)
#             performance[n_comp] = accuracy_score(y_validate, classifier.predict(pca.transform(X_validate))) * 100
#         handle_plot(performance, show_results, filename)
#     else:
#         pca = PCA()
#         pca.fit(X_train)
#         variance = pca.explained_variance_
#         n_comp = max(np.argwhere(variance > 0.9))[0]
#         pca.n_components = n_comp
#         classifier.fit(pca.fit_transform(X_train), y_train)
#         performance[0] = accuracy_score(y_validate, classifier.predict(pca.transform(X_validate))) * 100
#
#     return performance, n_comp
#
#
# def run_experiment_PCA(classifier, data_file, max_components = 20, batch=False, save_to_file=False, show_results=False):
#     dataframe = pd.read_csv(data_file)
#     performance = {}
#     for c in range(1, max_components):
#         if batch:
#             for _ in range(100):
#                 X_train, X_validate, y_train, y_validate = get_random_batch(dataframe)
#         else:
#             X_train, X_validate, y_train, y_validate = get_full_data(dataframe)
#             _run_PCA(data)
#         performance
#
#     performance = {}
#

def run_ICA_experiment(classifier, data_file, max_components = 20, batch=False,  show_results=False, save_to_file=False):
    dataframe = pd.read_csv(data_file)
    performance = {}
    for n in range(1, max_components):
        if batch:
            p = 0
            for _ in range(100):
                data = get_random_batch(dataframe)
                ica = FastICA(n_components=n, max_iter=1000).fit(data[0], data[2])
                p += estimate_classifier_performance(classifier.fit(ica.transform(data[0]), data[2]), ica.transform(data[1]), data[3])
            performance[n] = p/100.0
        else:
            data = get_full_data(dataframe)
            ica = FastICA(n_components=n, max_iter=1000).fit(data[0], data[2])
            performance[n]= estimate_classifier_performance(classifier.fit(ica.transform(data[0]), data[2]), ica.transform(data[1]), data[3])
    handle_plot(performance, show_results, save_to_file)
    return performance




def handle_plot(performance, show_results, save_to_file):
    fig = plt.figure()
    plt.title('Number of Components Retained vs Performance')
    plt.xlabel('Number of Components')
    plt.ylabel('Accuracy (%)')

    plt.plot(performance.keys(), performance.values())
    if show_results:
        plt.show()
    if save_to_file:
        pp = PdfPages("experiment-results/" + save_to_file + ".pdf")
        pp.savefig(fig)
        pp.close()
