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

    def join_clusters_fail(self, index1, index2):
        # for every merge of clusters, our number of clusters goes down by one
        cluster=(self.clusters[index1][0].union(self.clusters[index1][1]), self.clusters[index2][0].union(
            self.clusters[index2][1]))
        self.distances[index1, index2], self.distances[index2,
                                                       index1]=math.inf, math.inf
        print(self.clusters[index1], self.clusters[index2])
        if len(cluster[0].intersection(cluster[1])) > 0: return
        # cluster = self.clusters[index1].union(self.clusters[index2])
        # if len(cluster[0].intersection(cluster[1])) > 0: print(self.data[index1], self.data[index2])
        self.cluster_dist[index1], self.cluster_dist[index2]=self.distances[index1,
                                                                              index2], self.distances[index1, index2]
        self.clusters[index1], self.clusters[index2]=cluster, cluster
        # self.clusters[index2] = cluster
        # self.heirarchy.append((self.distances[index1, index2], cluster))
        # self.distances[index2, index1] = math.inf
        self.num_clusters -= 1
        return cluster

    def join_clusters(self, index1, index2):
        clusters_to_join=[]
        print(index1, index2)
        print(len(self.clusters))
        for index in [index1, index2]:
            clusters_to_join.append(self.clusters.pop(index))
        cluster=clusters_to_join[0].union(clusters_to_join[1])
        self.clusters.append(cluster)
        return cluster


    def fit(self, X, y=None):
        self.data=np.array(X)
        # self.heirarchy = []
        # indices in set, HAC distance to get set (initially 0)
        # set of arrays
        # self.clusters = [({i}, {}) for i in range(len(self.data))]
        # print(self.clusters)

        self.clusters=[{i} for i in range(len(self.data))]
        self.cluster_dist=[0] * len(self.data)
        self.num_clusters=len(self.data)
        self.distances=self.getDistances(self.data)
        # min_index=np.argmin(self.distances)
        # min_x=math.floor(min_index / len(self.distances[0]))
        # min_y=math.floor(min_index % len(self.distances[0]))
        # while self.distances[min_x, min_y] < math.inf:
        while len(self.clusters) > self.k:
            min_index=np.argmin(self.distances)
            min_x=math.floor(min_index / len(self.distances[0]))
            min_y=math.floor(min_index % len(self.distances[0]))
            indices = sorted([min_x, min_y], reverse=True)
            self.join_clusters(indices[0], indices[1])
            self.distances=self.getUpdatedDistances(indices[0], indices[1])

        # test = set([({1,2},2),({3},4),({1},2)])
        # print(test)
        # for i in range(len(self.clusters)):
        print(self.clusters)

        # final_clusters = set(self.clusters)
        # final_clusters=[]
        # for i in range(self.k - 1):
        #     # final_clusters.append(self.clusters.pop(np.argmax(self.clusters, key = lambda x: x[-1])))
        #     cluster=self.clusters.pop(np.argmax(self.cluster_dist))
        #     final_clusters.append(cluster[0])
        #     final_clusters.append(cluster[1])
        # final_clusters = self.clusters[np.argpartition(self.clusters[:][-1], kth=len(self.clusters)-self.k)][-self.k:]


        # flattened = [item for sublist in final_clusters for item in sublist]
        # final_clusters=sorted(final_clusters, key=lambda x: len(x))
        # # final_clusters = sorted(list(sum(final_clusters, ())), key=len)
        # print(final_clusters[0])
        # results=[]
        # for i in range(self.k):
        #     results.append(final_clusters[i])
        # unioned_clusters = set(frozenset(cluster[0].union(cluster[1])) for cluster in final_clusters)

        # print(unioned_clusters)
        final_set=set(())
        for cluster in self.clusters:
            # joint_cluster = cluster[0].union(cluster[1])
            temp=final_set.intersection(cluster)
            if len(temp) > 0:
                print("FAIL test")
                # final_clusters.remove(cluster)
            final_set=final_set.union(cluster)
        if len(final_set) < len(self.data): print("FAIL test 2")
        # print(final_clusters)

        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
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
            self.sse.append(self.clusterSSE(centroid, self.clusters[i]))
        return np.array(self.sse)

    def clusterSSE(self, centroid, cluster):
        return np.square(cluster - centroid).sum(axis=None)

    def save_clusters(self, filename):
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
