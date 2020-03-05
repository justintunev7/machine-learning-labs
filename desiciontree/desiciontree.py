import numpy as np
import math
from sklearn.base import BaseEstimator, ClassifierMixin


class Node():
    def __init__(self, feature=None, value=None, majority_class=None):
        self.feature = feature
        self.value = value
        self.children = []
        # print(data)
        self.majority_class = majority_class
        # self.data = data

    # feature is index of feature, classes is list of class indices for feature
    def split(self, feature, classes, data):
        print("SPLIT TREE", feature)
        for i in range(classes):
            majority_class = self.majority(data[i]) if len(data[i]) > 0 else self.majority_class
            self.children.append(Node(int(feature), int(i), int(majority_class)))
        
    def get_feature_value(self):
        return self.feature, self.value
    
    def get_children(self):
        return self.children
    
    def get_majority_class(self):
        return self.majority_class
    
    def majority(self, data):
        a = list(data[:,-1])
        # print(a)
        return max(map(lambda val: (a.count(val), val), set(a)))[-1]
    
    def predict(self, instance):
        # print(len(self.children), len(self.data))
        if len(self.children) == 0:
            return self.majority_class
        child_feature, child_val = self.children[0].get_feature_value()
        return self.children[int(instance[child_feature])].predict(instance)
    
    def print_tree(self, pre=""):
        # print(pre, self.feature_class)
        for child in self.children:
            print(pre, child.get_feature_value(), child.get_majority_class(), "\n")
        for child in self.children:
            child.print_tree(pre + "\t")

# NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
class DTClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, counts=None):
        self.counts = counts
        print("COUNTS: ", counts)
        self.weights = None
        self.root = Node()
        self.visited = []
        """ Initialize class with chosen hyperparameters.

        Args:
            counts = how many types for each attribute
        Example:
            DT  = DTClassifier()
        """
    
    def init_weights(self):
        if self.weights: return [w * 0 for w in self.weights] # map(lambda w: w * 0, self.weights)
        self.weights = []
        for i in range(len(self.counts) - 1):
            self.weights.append(np.zeros((self.counts[i], self.counts[-1])))
        return self.weights
    
    def feature_entropy(self, data, total):
        entropy = 0
        for i in range(len(data)):
            feature_sum = data[i].sum()
            if feature_sum == 0: continue
            feature_prob = (feature_sum / total)
            feature_entropy = 0
            for j in range(len(data[i])):
                prob = (data[i][j]) / feature_sum
                if prob > 0: feature_entropy += prob * math.log2(prob)
            entropy += feature_entropy * (-1) * (feature_prob)
        return entropy

    # Get the entropy of a given attribute split
    # data is a 2D matrix that represents the count of each
    # feature class to target class. Targets is an array of counts of target classes
    def entropy(self, targets, data=None):
        total = targets.sum()
        if data is None:
            entropy = 0
            for i in range(len(targets)):
                prob = (targets[i] / total)
                entropy += prob * math.log2(prob)
            return entropy * -1
        else:
            return self.feature_entropy(data, total)
    
    def split_data(self, X, y, feature):
        data = []
        combined = np.column_stack((X, y))
        for i in range(self.counts[feature]):
            data.append(combined[combined[:,feature] == i])
        return data

    def get_min_entropy(self, targets, visited=[]):
        min_entropy = math.inf
        min_entropy_index = -1
        for i, matrix in enumerate(self.weights):
            # if i in visited: continue
            temp_entropy = self.entropy(targets, matrix)
            if temp_entropy < min_entropy: 
                min_entropy = temp_entropy
                min_entropy_index = i
        return min_entropy_index


        """ Fit the data; Make the Desicion tree
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
    def fit(self, X, y, node=None, visited=[]):
        print("IN FIT", visited)
        if node is None: node = self.root
        self.weights = self.init_weights()
        targets = np.zeros((self.counts[-1]))
        # get feature-output counts
        for i, instance in enumerate(X):
            for j, feature in enumerate(instance):
                self.weights[j][int(feature), int(y[i])] += 1
            targets[int(y[i])] += 1
        
        min_entropy_index = self.get_min_entropy(targets, visited)

        if min_entropy_index in visited: return
        
        data = self.split_data(X, y, min_entropy_index)
        node.split(min_entropy_index, self.counts[min_entropy_index], data)


        # data is X separated by classes from chosen feature
        for i, matrix in enumerate(data):
            remaining = list(range(len(matrix[0][:-1])))
            self.fit(matrix[:,remaining], matrix[:,-1], node.get_children()[i], visited + [min_entropy_index])        
        return self
    
    def predict_with_tree(self, instance):
        self.root.predict(instance)

    def predict(self, X):
        y_hat = []
        for i, instance in enumerate(X):
            # print("\n\n", i)
            y_hat.append(self.root.predict(instance))
        print(y_hat)
        return y_hat
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        pass

    def score(self, X, y):
        y_hat = self.predict(X)
        sum = 0
        for i, result in enumerate(y):
            sum += 0 if result == y_hat[i] else 1
        return 1 - sum / len(y_hat)
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-li    def _shuffle_data(self, X, y):
        """

    def get_root(self):
        return self.root