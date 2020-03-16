import numpy as np
import math
from sklearn.base import BaseEstimator, ClassifierMixin

class KNNClassifier(BaseEstimator,ClassifierMixin):


    def __init__(self,labeltype=[],weight_type='inverse_distance',columntype="nominal",k=3, use_distance_weighting=False): ## add parameters here
        """
        Args:
            columntype for each column tells you if continues[real] or if nominal.
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
        """
        self.columntype = columntype
        self.use_distance_weighting = use_distance_weighting
        self.weight_type = weight_type
        self.k = k



    def fit(self,data,labels):
        """ Fit the data; run the algorithm (for this lab really just saves the data :D)
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.data = np.array(data)
        self.labels = labels.astype(int)
        self.classes = max(self.labels) + 1
        return self
    
    # Gets the sorted distances between the row and all other datapoints in self.data
    # returns: sorted array of tuples of (distance, index, label)
    def getDistances(self, row):
        distances = np.column_stack((np.linalg.norm(self.data-row, axis=-1), self.labels))
        return distances[distances[:,0].argpartition(kth=self.k)][0:self.k]

    def predict_label(self, neighbors):
        if self.use_distance_weighting:
            labels = {}
            for i in range(self.classes):
                print(i)
                labels[i] = 0
            for i, neighbor in enumerate(neighbors):
                if neighbor[0] == 0.0: return int(neighbor[-1])
                if neighbor[-1] not in labels: print(labels)
                labels[int(neighbor[-1])] += (1 / (neighbor[0]))         
            return max(labels,key=labels.get)
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
            y_hat.append(self.predict_label(neighbors))
        return y_hat


        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        pass
    
    # returns the mean squared error on the data set
    def mse(self, y_hat, y_truth):
        return np.square(y_hat - y_truth).mean(axis=None)

    # Accuracy based on percent correct
    def accuracy(self, y_hat, y_truth):
        sum = 0
        for i, result in enumerate(y_truth):
            sum += 0 if result == y_hat[i] else 1
        return 1 - sum / len(y_hat)

    #Returns the Mean score given input data and labels
    def score(self, X, y):
        y_hat = self.predict(X)
        error = self.mse(y_hat, y.astype(int))
        accuracy = self.accuracy(y_hat, y.astype(int))
        return error, accuracy 
        
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
                X (array-like): A 2D numpy array with data, excluding targets
                y (array-like): A 2D numpy array with targets
        Returns:
                score : float
                        Mean accuracy of self.predict(X) wrt. y.
        """
