import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
import math


class Cluster():
    def __init__(self, indices=set(()), distance=0):
        self.indices = indices
        self.distance = distance

    def mergeCluster(self, new_indices, distance):
        self.distance = distance
        self.indices = self.indices.union(new_indices)

    def getIndices(self):
        return self.indices

    def getDistance(self):
        return self.distance


class HACClustering(BaseEstimator, ClusterMixin):

    def __init__(self, k=3, link_type='single'):  # add parameters here
        """
        Args:
            k = how many final clusters to have
            link_type = single or complete. when combining two clusters use complete link or single link
        """
        self.link_type = link_type
        self.k = k

    def getDistancesFromInstance(self, instance):
        return np.linalg.norm(self.data-instance, axis=-1)

    def getDistances(self, data):
        distances = np.empty((0, len(self.data)))
        for instance in data:
            distances = np.vstack(
                (distances, self.getDistancesFromInstance(instance)))
        np.fill_diagonal(distances, math.inf)
        return distances

    def join_clusters(self, index1, index2):
        # for every merge of clusters, our number of clusters goes down by one
        self.num_clusters -= 1
        cluster = (self.clusters[index1][0].union(self.clusters[index1][1]), self.clusters[index2][0].union(
            self.clusters[index2][1]))
        self.cluster_dist[index1], self.cluster_dist[index2] = self.distances[index1,
                                                                              index2], self.distances[index1, index2]
        self.clusters[index1], self.clusters[index2] = cluster, cluster
        # self.clusters[index2] = cluster
        # self.heirarchy.append((self.distances[index1, index2], cluster))
        self.distances[index1, index2], self.distances[index2,
                                                       index1] = math.inf, math.inf
        # self.distances[index2, index1] = math.inf
        return cluster

    def fit(self, X, y=None):
        self.data = np.array(X)
        # self.heirarchy = []
        # indices in set, HAC distance to get set (initially 0)
        # set of arrays
        self.clusters = [({i}, set(())) for i in range(len(self.data))]
        self.cluster_dist = [0] * len(self.data)
        self.num_clusters = len(self.data)
        self.distances = self.getDistances(self.data)
        print(self.distances, "\n")
        min_index = np.argmin(self.distances)
        min_x = math.floor(min_index / len(self.distances[0]))
        min_y = math.floor(min_index % len(self.distances[0]))
        while self.num_clusters > 1:
            min_index = np.argmin(self.distances)
            min_x = math.floor(min_index / len(self.distances[0]))
            min_y = math.floor(min_index % len(self.distances[0]))
            self.join_clusters(min_x, min_y)

        # test = set([({1,2},2),({3},4),({1},2)])
        # print(test)
        # for i in range(len(self.clusters)):

        # final_clusters = set(self.clusters)
        final_clusters = []
        for i in range(self.k):
            # final_clusters.append(self.clusters.pop(np.argmax(self.clusters, key = lambda x: x[-1])))
            final_clusters.append(self.clusters.pop(
                np.argmax(self.cluster_dist)))
        # final_clusters = self.clusters[np.argpartition(self.clusters[:][-1], kth=len(self.clusters)-self.k)][-self.k:]
        print(final_clusters)

        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        return self

    def getClusters(self, distances):
        clusters = [[] for i in range(self.k)]
        clusters_indices = np.argmin(distances, axis=0)
        for i, cluster in enumerate(clusters_indices):
            clusters[cluster].append(self.data[i])
        return np.array(clusters)

    # get SSE of clusters
    def sse(self):
        self.sse = []
        for i, centroid in enumerate(self.centroids):
            self.sse.append(self.clusterSSE(centroid, self.clusters[i]))
        return np.array(self.sse)

    def clusterSSE(self, centroid, cluster):
        return np.square(cluster - centroid).sum(axis=None)

    def save_clusters(self, filename):
        sse = self.sse()
        f = open(filename, "w+")
        f.write("{:d}\n".format(self.k))
        f.write("{:.4f}\n\n".format(sse.sum()))
        for i, centroid in enumerate(self.centroids):
            f.write(np.array2string(centroid, precision=4, separator=","))
            f.write("\n")
            f.write("{:d}\n".format(len(self.clusters[i])))
            f.write("{:.4f}\n\n".format(sse[i]))
        f.close()
        """
            f = open(filename,"w+") 
            Used for grading.
            write("{:d}\n".format(k))
            write("{:.4f}\n\n".format(total SSE))
            for each cluster and centroid:
                write(np.array2string(centroid,precision=4,separator=","))
                write("\n")
                write("{:d}\n".format(size of cluster))
                write("{:.4f}\n\n".format(SSE of cluster))
            f.close()
        """
