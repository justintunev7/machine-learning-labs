import arff
import math
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.linear_model import Perceptron

class PerceptronClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, lr=.1, shuffle=True, deterministic=10000, stopping_count=30):
        self.lr = lr
        self.shuffle = shuffle
        self.deterministic = deterministic
        self.stopping_count = stopping_count
        self.epochs = 0
        self.misclassifications = []

    def split_data_sets(self, data, labels):
        data, labels = self._shuffle_data(data, labels)
        train_size = round(len(data) * .7)
        train_data = data[:train_size]
        train_labels = labels[:train_size]
        test_data = data[train_size:]
        test_labels = labels[train_size:]
        return (train_data, train_labels, test_data, test_labels)

    # X is a list of input data, y is a list of results
    def fit(self, X, y, initial_weights=None):
        self.weights = initial_weights = self.initialize_weights(len(X[0]) + 1) if initial_weights is None else initial_weights
        count = 0
        prev_score = self.score(X, y)
        # If deterministic is set, then just run n times, if not, continue until n epochs result in no change
        while (self.epochs < self.deterministic and count < self.stopping_count):
            self.epochs += 1
            initial_weights = self.weights
            # shuffle data if specified
            if self.shuffle: X, y = self._shuffle_data(X, y)
            for i in range(len(X)):
                net = initial_weights.dot(np.append(X[i], 1))
                z = 1 if net > 0 else 0 # where z is the output found with current model
                a = (y[i] - z)
                if (a != 0):
                    change_weights = (np.append(X[i], 1) * (self.lr * a))
                    initial_weights = initial_weights + change_weights
            temp = self.weights
            self.weights = initial_weights
            current_score = self.score(X,y)
            # only update if the current_score is better than the prev_score
            if (current_score <= prev_score+.01) and self.deterministic==10000:
                self.misclassifications.append(1-prev_score)
                count += 1
                self.weights = temp
            else:
                self.misclassifications.append(1-current_score)
                prev_score = current_score
                count = 0
        return self

    def predict(self, X):
        results = []
        for i in range(len(X)):
            net = self.weights.dot(np.append(X[i], 1))
            z = 1 if net > 0 else 0 # where z is the output found with current model
            results.append(z)
        return results


    def initialize_weights(self, size = 2):
        return np.zeros(size)


    def score(self, X, y):
        predicted_results = self.predict(X)
        loss = 0
        for i, result in enumerate(predicted_results):
            loss += abs((y[i][0] - result))
        return 1 - loss/len(predicted_results)


    def _shuffle_data(self, X, y):
        x_len = len(X[0])
        combined = np.column_stack((X, y))
        np.random.shuffle(combined)
        return (combined[:, 0:x_len], combined[:,x_len:])

    def plot_descision_line(self, X, y):
        a = np.empty((0,len(X[0])))
        b = np.empty((0,len(X[0])))
        for i in range(len(X)):
            if y[i][0] == 1: a = np.vstack((a, X[i]))
            else: b = np.vstack((b, X[i]))
        plt.scatter(a[:,0], a[:,1], label='1')
        plt.scatter(b[:,0], b[:,1], label='0')

        slope = -self.weights[0]/self.weights[1]
        x = [-1, 1]
        plt.plot(x, [x[0]*slope - self.weights[2], x[1]*slope - self.weights[2]])
        plt.xlabel("var 1")
        plt.ylabel("var 2")
        plt.title("Instances and Decision Line")
        time.sleep(2)
        plt.show()

    def plot_misclassification_rate(self, misclass):
        max_epoch = max(len(x) for x in misclass)
        y = []
        for i in range(max_epoch):
            temp = []
            # get average value between all lists
            for j in range(len(misclass)):
                if len(misclass[j]) > i: temp.append(misclass[j][i])
            y.append(sum(temp)/(len(temp)))
        plt.plot(range(len(y)), y)
        plt.xlabel("Epochs completed")
        plt.ylabel("Miscallification Rate")
        plt.title("Miscallification rate vs epochs completed")
        plt.show()

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights
    
    # Returns the number of epochs done during training.
    def get_epochs(self):
        return self.epochs
    
    def get_missclassifications(self):
        return self.misclassifications