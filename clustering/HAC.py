import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
import math


class TreeNode():
    def __init__(self, indices=set(()), left_child=None, right_child=None, distance=0):
        self.indices = indices
        self.left_child = left_child
        self.right_child = right_child
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

    def getUpdatedDistances(self, index1, index2):
        if self.link_type is 'single':
            new_row = np.min([self.distances[index1], self.distances[index2]], axis=0)
            new_row = np.delete(new_row, [index1, index2])
            mask = np.ones(len(self.distances), dtype=bool)
            mask[[index1,index2]] = False
            self.distances = self.distances[mask]
            self.distances = self.distances[:,mask]
            self.distances = np.vstack((self.distances, new_row))
            self.distances = np.column_stack((self.distances, np.append(new_row, [math.inf])))
        return self.distances

    def join_clusters(self, index1, index2):
        clusters_to_join=[]
        for index in [index1, index2]:
            clusters_to_join.append(self.clusters.pop(index))
        cluster=clusters_to_join[0].union(clusters_to_join[1])
        self.clusters.append(cluster)
        return cluster
    
    def testResult(self, clusters):
        final_set=set(())
        for cluster in self.clusters:
            temp=final_set.intersection(cluster)
            if len(temp) > 0:
                print("ERROR: Overlapping clusters")
            final_set=final_set.union(cluster)
        if len(final_set) < len(self.data): print("ERROR: Clusters do not cover dataset")


    def fit(self, X, y=None):
        self.data=np.array(X)
        self.clusters=[{i} for i in range(len(self.data))]
        self.distances=self.getDistances(self.data)
        while len(self.clusters) > self.k:
            min_index=np.argmin(self.distances)
            min_x=math.floor(min_index / len(self.distances[0]))
            min_y=math.floor(min_index % len(self.distances[0]))
            # sort the indices to avoid problems later with deleting stuff out of order
            indices = sorted([min_x, min_y], reverse=True)
            self.join_clusters(indices[0], indices[1])
            self.distances=self.getUpdatedDistances(indices[0], indices[1])
        self.testResult(self.clusters)
        return self

    def getClusters(self, distances):
        clusters=[[] for i in range(self.k)]
        clusters_indices=np.argmin(distances, axis=0)
        for i, cluster in enumerate(clusters_indices):
            clusters[cluster].append(self.data[i])
        return np.array(clusters)

    # get SSE of clusters
    def sse(self):
        self.sse=[]
        for i, centroid in enumerate(self.centroids):
            self.sse.append(self.clusterSSE(centroid, self.data[list(self.clusters[i])]))
        return np.array(self.sse)

    def clusterSSE(self, centroid, cluster):
        return np.square(cluster - centroid).sum(axis=None)
    
    def centroids(self):
        centroids = []
        for cluster in self.clusters:
            centroids.append(self.data[list(cluster)].mean(axis=0))
        return centroids


    def save_clusters(self, filename):
        self.centroids = self.centroids()
        sse=self.sse()
        f=open(filename, "w+")
        f.write("{:d}\n".format(self.k))
        f.write("{:.4f}\n\n".format(sse.sum()))
        for i, centroid in enumerate(self.centroids):
            f.write(np.array2string(centroid, precision=4, separator=","))
            f.write("\n")
            f.write("{:d}\n".format(len(self.clusters[i])))
            f.write("{:.4f}\n\n".format(sse[i]))
        f.close()
