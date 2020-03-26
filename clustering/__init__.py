from arff import Arff
import matplotlib.pyplot as plt
import numpy as np
from HAC import HACClustering
from Kmeans import KMEANSClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sys
import copy
import math


def plot_bar(data, labels=None, xlabel="", ylabel="", title=""):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if labels != None:
        plt.xticks(range(len(labels)), labels)
    if type(data) == dict:
        width = 1 / (len(data) + 1)
        for i, key in enumerate(data):
            value = data[key]
            offset = width * (i - len(data) // 2)
            plt.bar(np.arange(len(value)) + offset,
                    value, width=width, label=key)
        plt.legend()
    else:
        plt.bar(range(len(data)), data)
    plt.show()


 ## label_count = 0 because clustering is unsupervised.
def setup_data(file, label_count=0, exclude_output=False):
    print("\nFile =", file)
    mat = Arff(file, label_count=label_count)
    raw_data = mat.data
    data = raw_data[:,:-1] if exclude_output else raw_data
    data = raw_data
    scaler = MinMaxScaler()
    scaler.fit(data)
    norm_data = scaler.transform(data)
    return norm_data


def split_data(X, y, test_split=.25):
    # create test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split)
    return X_train, X_test, y_train, y_test


def normalize(data):
    maxes = data.max(axis=1)
    mins = data.min(axis=1)
    normal = (data - mins[:, None]) / (maxes - mins)[:, None]
    return normal


def split_data(X, y, test_split=.25):
    # create test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split)
    return X_train, X_test, y_train, y_test

def kmeans(norm_data, k=5, debug=False, output_file=None):
    KMEANS = KMEANSClustering(k=k,debug=debug)
    KMEANS.fit(norm_data)
    if output_file: KMEANS.save_clusters(output_file)
    return KMEANS.getSSE().sum()

### HAC SINGLE LINK ###
# link_type = 'single' or 'complete'
# output_file = debug_hac_complete.txt or debug_hac_single.txt or whatever
def hac(norm_data, k=5, link_type='single', output_file=None):
    HAC = HACClustering(k=k,link_type=link_type)
    HAC.fit(norm_data)
    if output_file: HAC.save_clusters(output_file)
    return HAC.getSSE().sum()

def two():
    data = setup_data('iris.arff', exclude_output=True)
    tests = [2,3,4,5,6,7]
    sse_results = {"kmeans": [], "hac_single": [], "hac_complete": []} 
    for k in tests:
        sse_results["kmeans"].append(kmeans(data, k=k, debug=False))
        sse_results["hac_single"].append(hac(data, k=k, link_type='single'))
        sse_results["hac_complete"].append(hac(data, k=k, link_type='complete'))
    plot_bar(sse_results, labels=tests, xlabel="K value", ylabel="SSE", title="SSE by K value (output label excluded)")

    data = setup_data('iris.arff', exclude_output=False)
    sse_results_2 = {"kmeans": [], "hac_single": [], "hac_complete": []} 
    for k in tests:
        sse_results_2["kmeans"].append(kmeans(data, k=k, debug=False))
        sse_results_2["hac_single"].append(hac(data, k=k, link_type='single'))
        sse_results_2["hac_complete"].append(hac(data, k=k, link_type='complete'))
    plot_bar(sse_results_2, labels=tests, xlabel="K value", ylabel="SSE", title="SSE by K value (output label included)")

    sse_results_3 = {"kmeans": [], "hac_single": [], "hac_complete": []}
    k=4
    for i in range(5):
        sse_results_3["kmeans"].append(kmeans(data, k=k, debug=False))
        sse_results_3["hac_single"].append(hac(data, k=k, link_type='single'))
        sse_results_3["hac_complete"].append(hac(data, k=k, link_type='complete'))
    plot_bar(sse_results_3, labels=[4,4,4,4,4], xlabel="K value", ylabel="SSE", title="SSE by K value (output label included)")
        
def one():
    data = setup_data('seismic-bumps_train.arff')
    kmeans(data, debug=True, output_file="evaluation_kmeans.txt")
    hac(data, link_type='single', output_file="evaluation_hac_single.txt")
    hac(data, link_type='complete', output_file="evaluation_hac_complete.txt")

def debug():
    data = setup_data('abalone.arff')
    kmeans(data, debug=True, output_file="debug_kmeans.txt")
    hac(data, link_type='single', output_file="debug_hac_single.txt")
    hac(data, link_type='complete', output_file="debug_hac_complete.txt")


def main():
    print("Starting tests")
    # debug()
    one()
    # two()
    # three()
    # four()
    # five()
    # six()
    # seven()
    return


if __name__ == '__main__':
    main()
