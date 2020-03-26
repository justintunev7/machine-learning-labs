from arff import Arff
import matplotlib.pyplot as plt
import numpy as np
from HAC import HACClustering
from Kmeans import KMEANSClustering
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.cluster import silhouette_score, davies_bouldin_score, calinski_harabasz_score
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

def kmeans(norm_data, k=5, debug=False, output_file=None, scikit=False, score=silhouette_score):
    KMEANS = KMeans(n_clusters=k, n_init=12-k, max_iter=400-(50*k)) if scikit else KMEANSClustering(k=k,debug=debug)
    KMEANS.fit(norm_data)
    if output_file: KMEANS.save_clusters(output_file)
    if scikit:
        # print(KMEANS.inertia_)
        return score(norm_data, KMEANS.labels_)
        # return KMEANS.inertia_
    else:
        return KMEANS.getSSE().sum()

### HAC SINGLE LINK ###
# link_type = 'single' or 'complete'
# output_file = debug_hac_complete.txt or debug_hac_single.txt or whatever
def hac(norm_data, k=5, link_type='single', output_file=None, scikit=False, score=silhouette_score):
    HAC = AgglomerativeClustering(n_clusters=k, linkage=link_type) if scikit else HACClustering(k=k,link_type=link_type)
    HAC.fit(norm_data)
    if output_file: HAC.save_clusters(output_file)
    if scikit:
        # print(HAC.labels_)
        return score(norm_data, HAC.labels_)
    else:
        return HAC.getSSE().sum()

def three():
    data = setup_data('iris.arff', exclude_output=False)
    tests = [2,3,4,5,6,7]
    sse_results = {"kmean(scikit)": [], "hac_single(scikit)": [],  "hac_complete(scikit)": [], "hac_ward(scikit)": []}
    for k in tests:
        sse_results["kmean(scikit)"].append(kmeans(data, k=k, scikit=True))
        sse_results["hac_single(scikit)"].append(hac(data, k=k, link_type='single', scikit=True))
        sse_results["hac_complete(scikit)"].append(hac(data, k=k, link_type='complete', scikit=True))
        sse_results["hac_ward(scikit)"].append(hac(data, k=k, link_type='ward', scikit=True))
    plot_bar(sse_results, labels=tests, xlabel="K value", ylabel="Silhouette Score", title="SSE by K value (SKLearn)")

    data = setup_data('diabetes.arff', exclude_output=False)
    tests = [2,3,4,5,6,7]
    sse_results = {"kmean(scikit)": [], "hac_single(scikit)": [],  "hac_complete(scikit)": [], "hac_ward(scikit)": []}
    for k in tests:
        sse_results["kmean(scikit)"].append(kmeans(data, k=k, scikit=True))
        sse_results["hac_single(scikit)"].append(hac(data, k=k, link_type='single', scikit=True))
        sse_results["hac_complete(scikit)"].append(hac(data, k=k, link_type='complete', scikit=True))
        sse_results["hac_ward(scikit)"].append(hac(data, k=k, link_type='ward', scikit=True))
    plot_bar(sse_results, labels=tests, xlabel="K value", ylabel="Silhouette Score", title="SSE by K value (SKLearn)")



    sse_results = {"kmean(scikit)": [], "hac_single(scikit)": [],  "hac_complete(scikit)": [], "hac_ward(scikit)": []}
    for k in tests:
        sse_results["kmean(scikit)"].append(kmeans(data, k=k, scikit=True, score=davies_bouldin_score))
        sse_results["hac_single(scikit)"].append(hac(data, k=k, link_type='single', scikit=True, score=davies_bouldin_score))
        sse_results["hac_complete(scikit)"].append(hac(data, k=k, link_type='complete', scikit=True, score=davies_bouldin_score))
        sse_results["hac_ward(scikit)"].append(hac(data, k=k, link_type='ward', scikit=True, score=davies_bouldin_score))
    plot_bar(sse_results, labels=tests, xlabel="K value", ylabel="Davies Bouldin Score", title="SSE by K value (SKLearn)")

    sse_results = {"kmean(scikit)": [], "hac_single(scikit)": [],  "hac_complete(scikit)": [], "hac_ward(scikit)": []}
    for k in tests:
        sse_results["kmean(scikit)"].append(kmeans(data, k=k, scikit=True, score=calinski_harabasz_score))
        sse_results["hac_single(scikit)"].append(hac(data, k=k, link_type='single', scikit=True, score=calinski_harabasz_score))
        sse_results["hac_complete(scikit)"].append(hac(data, k=k, link_type='complete', scikit=True, score=calinski_harabasz_score))
        sse_results["hac_ward(scikit)"].append(hac(data, k=k, link_type='ward', scikit=True, score=calinski_harabasz_score))
    plot_bar(sse_results, labels=tests, xlabel="K value", ylabel="Calinski Harabasz Score", title="SSE by K value (SKLearn)")



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

    sse_results_3 = {"kmeans": []}
    k=4
    for i in range(5):
        sse_results_3["kmeans"].append(kmeans(data, k=k, debug=False))
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

def test():
    data = setup_data('iris.arff', exclude_output=True)
    tests = [2,3,4,5,6,7]
    for i in range(50):
        for k in tests:
            kmeans(data, k=k, debug=False)

def main():
    print("Starting tests")
    # test()
    # debug()
    # one()
    # two()
    three()
    return


if __name__ == '__main__':
    main()
