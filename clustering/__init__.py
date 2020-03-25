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
def setup_data(file, label_count=0):
    print("\nFile =", file)
    mat = Arff(file, label_count=label_count)
    raw_data = mat.data
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

def kmeans(norm_data, k=5, debug=False, output_file="debug_kmeans.txt"):
    KMEANS = KMEANSClustering(k=k,debug=debug)
    KMEANS.fit(norm_data)
    KMEANS.save_clusters(output_file)

### HAC SINGLE LINK ###
# link_type = 'single' or 'complete'
# output_file = debug_hac_complete.txt or debug_hac_single.txt or whatever
def hac(norm_data, link_type='single', output_file="debug_hac_single.txt"):
    HAC_single = HACClustering(k=5,link_type=link_type)
    HAC_single.fit(norm_data)
    # HAC_single.save_clusters(output_file)

def debug():
    data = setup_data('abalone.arff')
    # kmeans(data, debug=True)
    hac(data, link_type='single', output_file="debug_hac_single.txt")


def main():
    print("Starting tests")
    debug()
    # one()
    # two()
    # three()
    # four()
    # five()
    # six()
    # seven()
    return


if __name__ == '__main__':
    main()
