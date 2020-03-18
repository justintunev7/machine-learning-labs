import numpy as np
import math
from sklearn.base import BaseEstimator, ClassifierMixin

class KNNClassifier(BaseEstimator,ClassifierMixin):


    def __init__(self,labeltype=[],weight_type='inverse_distance',columntype="nominal",k=3, use_distance_weighting=False, regression=False): ## add parameters here
        """
        Args:
            columntype for each column tells you if continues[real] or if nominal.
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
        """
        self.columntype = columntype
        self.use_distance_weighting = use_distance_weighting
        self.weight_type = weight_type
        self.k = k
        self.regression = regression



    def fit(self,data,labels):
        """ Fit the data; run the algorithm (for this lab really just saves the data :D)
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.data = np.array(data)
        self.labels = labels
        return self
    
    # Gets the sorted distances between the row and all other datapoints in self.data
    # returns: sorted array of tuples of (distance, index, label)
    def getDistances(self, row):
        distances = np.column_stack((np.linalg.norm(self.data-row, axis=-1), self.labels))
        # distances = np.column_stack(((np.sqrt(np.sum(np.power(self.data-row, 2), axis=-1))), self.labels))
        return distances[distances[:,0].argpartition(kth=self.k)][0:self.k]

    # Computes regression labels
    def predict_regression(self, neighbors):
        if self.use_distance_weighting:
            # If distance is zero, return that label
            if 0.0 in neighbors[:,0]:
                return neighbors[np.where(neighbors == 0.0)[0][0], -1]

            # Regression formula
            return np.sum((neighbors[:,-1] / (neighbors[:,0] ** 2))) / np.sum((1 / (neighbors[:,0] ** 2)))
        elif self.regression:
            # if we don't care about the distance weighting, then just return the average label
            return np.mean(neighbors[:,-1])

    # Predicts using the distances (1/d^2) for each neighbor
    def predict_distance_weighted(self, neighbors):
        self.classes = int(max(self.labels) + 1)
        # Initialize labels (this helps with edge cases when two labels have equal distances)
        labels = {}
        for i in range(self.classes):
            labels[i] = 0
        for i, neighbor in enumerate(neighbors):
            # If distance is zero, just return the label
            if neighbor[0] == 0.0:
                return int(neighbor[-1])
            # Error check
            if neighbor[-1] not in labels: print(labels)
            # inverse distance = 1/d^2
            labels[neighbor[-1]] += (1 / (neighbor[0] ** 2))
        return max(labels,key=labels.get)

    # Generic predict method
    def predict_label(self, neighbors):
        if self.regression:
            return self.predict_regression(neighbors)
        elif self.use_distance_weighting:
            return self.predict_distance_weighted(neighbors)
        else:
            # get max counted class
            counts = np.bincount(neighbors.astype(int)[:,-1])
            return np.argmax(counts)

    def predict(self,X):
        y_hat = []
        for i, row in enumerate(X):
            # if i % 50 == 0:
            #     print(i)
            neighbors = self.getDistances(row)
            # print(neighbors)
            y_hat.append(self.predict_label(neighbors))
        # print(y_hat)
        return y_hat


        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
    
    # returns the mean squared error on the data set
    def mse(self, y_hat, y_truth):
        # print(y_hat)
        # print(y_truth)
        # print(y_hat - y_truth)
        return np.square(y_hat - y_truth).mean(axis=None)

    # Accuracy based on percent correct
    def accuracy(self, y_hat, y_truth):
        # print(y_hat, y_truth)
        sum = 0
        for i, result in enumerate(y_truth):
            sum += 0 if result == y_hat[i] else 1
        return 1 - sum / len(y_hat)

    #Returns the Mean score given input data and labels
    def score(self, X, y):
        y_hat = self.predict(X)
        error = self.mse(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        return error, accuracy 
        
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
                X (array-like): A 2D numpy array with data, excluding targets
                y (array-like): A 2D numpy array with targets
        Returns:
                score : float
                        Mean accuracy of self.predict(X) wrt. y.
        """
