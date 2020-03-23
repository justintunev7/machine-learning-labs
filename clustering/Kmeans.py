import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from random import randint
from copy import copy

class KMEANSClustering(BaseEstimator,ClusterMixin):

    def __init__(self,k=3,debug=False): ## add parameters here
        """
        Args:
            k = how many final clusters to have
            debug = if debug is true use the first k instances as the initial centroids otherwise choose random points as the initial centroids.
        """
        self.k = k
        self.debug = debug
    
    def init_centroids(self, X):
        centroids = []
        if self.debug:
            centroids = X[:self.k]
        else:
            selected = set(())
            for i in range(self.k):
                rand_index = randint(0, len(X))
                while rand_index in selected: rand_index = randint(0, len(X))
                selected.add(rand_index)
                centroids.append(X[rand_index])
        return centroids
    
    def getDistanceFromCentroid(self, centroid):
        return np.linalg.norm(self.data-centroid, axis=-1)

    def getDistances(self, centroids):
        distances = np.empty((0,len(self.data)))
        for centroid in centroids:
            distances = np.vstack((distances, self.getDistanceFromCentroid(centroid)))
        return distances

    def getClusters(self, distances):
        clusters = [[] for i in range(self.k)]
        clusters_indices = np.argmin(distances, axis=0)
        for i, cluster in enumerate(clusters_indices):
            clusters[cluster].append(self.data[i])
        return np.array(clusters)
    
    def getNewCentroids(self, distances):
        self.clusters = self.getClusters(distances)
        centroids = []
        for cluster_val in self.clusters:
            centroids.append(np.mean(cluster_val, axis=0))
        return np.array(centroids)


    def fit(self,X,y=None):
        self.data = np.array(X)
        self.centroids = self.init_centroids(X)
        prev_centroids = []

        while not np.array_equal(prev_centroids, self.centroids):
            prev_centroids = copy(self.centroids)
            distances = self.getDistances(self.centroids)
            self.centroids = self.getNewCentroids(distances)
        
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        return self
    
    # get SSE of clusters
    def sse(self):
        self.sse = []
        for i, centroid in enumerate(self.centroids):
            self.sse.append(self.clusterSSE(centroid, self.clusters[i]))
        return np.array(self.sse)
    
    def clusterSSE(self, centroid, cluster):
        return np.square(cluster - centroid).sum(axis=None)

    def save_clusters(self,filename):
        sse = self.sse()
        f = open(filename,"w+")
        f.write("{:d}\n".format(self.k))
        f.write("{:.4f}\n\n".format(sse.sum()))
        for i, centroid in enumerate(self.centroids):
            f.write(np.array2string(centroid,precision=4,separator=","))
            f.write("\n")
            f.write("{:d}\n".format(len(self.clusters[i])))
            f.write("{:.4f}\n\n".format(sse[i]))
        f.close()
