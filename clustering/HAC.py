import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
import math

class HACClustering(BaseEstimator, ClusterMixin):

    def __init__(self, k=3, link_type='single'):  # add parameters here
        """
        Args:
            k = how many final clusters to have
            link_type = single or complete. when combining two clusters use complete link or single link
        """
        self.link_type = link_type
        self.k = k
        self.sse = []

    # Gets distances from each instance to the current instance
    def getDistancesFromInstance(self, instance):
        return np.linalg.norm(self.data-instance, axis=-1)

    # Gets initial distances
    def getDistances(self, data):
        distances = np.empty((0, len(self.data)))
        for instance in data:
            distances = np.vstack(
                (distances, self.getDistancesFromInstance(instance)))
        np.fill_diagonal(distances, math.inf)
        return distances

    # Update the distance matrix with combined clusters
    def getUpdatedDistances(self, index1, index2):
        if self.link_type is 'single':
            new_row = np.min([self.distances[index1], self.distances[index2]], axis=0)
        else:
            new_row = np.max([self.distances[index1], self.distances[index2]], axis=0)
        new_row = np.delete(new_row, [index1, index2])
        mask = np.ones(len(self.distances), dtype=bool)
        mask[[index1,index2]] = False
        self.distances = self.distances[mask]
        self.distances = self.distances[:,mask]
        self.distances = np.vstack((self.distances, new_row))
        self.distances = np.column_stack((self.distances, np.append(new_row, [math.inf])))
        return self.distances

    # Merge two clusters together
    def join_clusters(self, index1, index2):
        clusters_to_join=[]
        for index in [index1, index2]:
            clusters_to_join.append(self.clusters.pop(index))
        cluster=clusters_to_join[0].union(clusters_to_join[1])
        self.clusters.append(cluster)
        return cluster
    
    # A simple error check to make sure that all clusters are disjoint and when
    # combined, they include all data instances
    def testResult(self, clusters):
        final_set=set(())
        for cluster in self.clusters:
            temp=final_set.intersection(cluster)
            if len(temp) > 0:
                print("ERROR: Overlapping clusters")
            final_set=final_set.union(cluster)
        if len(final_set) < len(self.data): print("ERROR: Clusters do not cover dataset")

    # Gets indices of the next closest pair of clusters
    def getIndicesOfMinDistance(self):
        min_index=np.argmin(self.distances)
        min_x=math.floor(min_index / len(self.distances[0]))
        min_y=math.floor(min_index % len(self.distances[0]))
        # sort the indices to avoid problems later with deleting stuff out of order
        return sorted([min_x, min_y], reverse=True)

    def fit(self, X, y=None):
        self.data=np.array(X)
        self.clusters=[{i} for i in range(len(self.data))]
        self.distances=self.getDistances(self.data)
        while len(self.clusters) > self.k:
            indices = self.getIndicesOfMinDistance()
            self.join_clusters(indices[0], indices[1])
            self.distances=self.getUpdatedDistances(indices[0], indices[1])
        # print(self.clusters)
        self.testResult(self.clusters)
        return self

    def getClusters(self, distances):
        clusters=[[] for i in range(self.k)]
        clusters_indices=np.argmin(distances, axis=0)
        for i, cluster in enumerate(clusters_indices):
            clusters[cluster].append(self.data[i])
        return np.array(clusters)

    # get SSE of clusters
    def getSSE(self):
        if len(self.sse) > 0: return self.sse
        self.centroids = self.centroids()
        # self.sse=[]
        for i, centroid in enumerate(self.centroids):
            self.sse.append(self.clusterSSE(centroid, self.data[list(self.clusters[i])]))
        self.sse = np.array(self.sse)
        return self.sse

    def clusterSSE(self, centroid, cluster):
        return np.square(cluster - centroid).sum(axis=None)
    
    def centroids(self):
        centroids = []
        for cluster in self.clusters:
            centroids.append(self.data[list(cluster)].mean(axis=0))
        return centroids


    def save_clusters(self, filename):
        sse=self.getSSE()
        f=open(filename, "w+")
        f.write("{:d}\n".format(self.k))
        f.write("{:.4f}\n\n".format(sse.sum()))
        for i, centroid in enumerate(self.centroids):
            f.write(np.array2string(centroid, precision=4, separator=","))
            f.write("\n")
            f.write("{:d}\n".format(len(self.clusters[i])))
            f.write("{:.4f}\n\n".format(sse[i]))
        f.close()
