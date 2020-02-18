import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import math
import copy

class MLPClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, hidden_layer_widths, lr=.1, momentum=.5, shuffle=True, deterministic=10, debug=False, analytics=False):
        self.hidden_layer_widths = np.array(hidden_layer_widths)
        self.lr = lr
        self.momentum = momentum
        self.deterministic = deterministic
        self.shuffle = shuffle
        self.debug = debug
        self.lb = preprocessing.LabelBinarizer()
        self.validation_stop = False
        self.analytics = analytics

    def split_data(self, X, y, validation_split=.1, test_split=.25):
        # create test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
        # create validation split
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_split)
        self.X_validation = X_validation
        self.y_validation = self._one_hot_encode(y_validation)
        self.validation_stop = True
        return X_train, X_test, y_train, y_test
    

    def backward(self, results, y_truth):
        deltas = [None] * len(self.weights)
        delta_weights = [None] * len(self.weights)
        for i, layer in reversed(list(enumerate(results))):
            if i < len(results)-1:
                delta_weights[i] = np.transpose((np.multiply(results[i], deltas[i][:,None]) * self.lr)) + (self.momentum * self.prev_delta_weights[i])
                if i > 0: deltas[i-1] = (np.dot(self.weights[i][0:-1], deltas[i])) * self.sigmoid_derivative(results[i][:-1])
            if i == (len(results) - 1):
                deltas[i-1] = (y_truth - layer) * self.sigmoid_derivative(layer)

        if self.debug: print("DELTAS:\n", deltas) # debug info

        self.prev_delta_weights = delta_weights
        return delta_weights

    def single_forward(self, input_values, layer):
        return input_values.dot(self.weights[layer])

    def forward(self, x):
        results = []
        for i in range(len(self.weights)): # for each layer in network
            x = np.append(x, 1)
            results.append(x)
            x = self.single_forward(x, i)
            if i < len(self.weights): 
                x = self.sigmoid(x)
        results.append(x)
        return results
    
    def predict(self, X):
        y_hat = []
        for i in range(len(X)):
            y_hat.append(self.forward(X[i])[-1][0])
        return y_hat

    # Trains the model based on the 2D input and target arrays.
    # Performs forward steps and backpropagation
    # TODO: separate stopping criterias (validation set or not)
    def fit(self, X, y, initial_weights=None):
        y = self._one_hot_encode(y)
        self.weights = initial_weights = self.initialize_weights(len(X[0]), len(y[0])) if initial_weights is None else initial_weights
        current_solution, bssf = (self._validation_error(), copy.copy(self.weights)), None
        self.training_error = []
        self.validation_error = []
        self.validation_accuracy = []
        self.prev_delta_weights = [0] * len(self.weights)
        while current_solution != bssf:
            bssf = current_solution
            for a in range(self.deterministic):
                y_hat = []
                # print("--Epoch {}--".format(a+1))
                # shuffle the data each epoch
                X, y = self._shuffle_data(X, y)
                for i in range(len(X)):
                    results = self.forward(X[i])
                    y_hat.append(results[-1])
                    delta_weights = self.backward(results, y[i])
                    # if self.debug: self.print_debug_info(X[i], y[i], results) # debug info
                    self.weights = np.add(self.weights, delta_weights)
                self.validation_error.append(self._validation_error())
                if self.analytics: self.validation_accuracy.append(self._validation_accuracy())
                temp_solution = (self.validation_error[-1], copy.copy(self.weights))
                if self.analytics: self.training_error.append(self.mse(y_hat, y))
                if temp_solution[0] < bssf[0]: current_solution = temp_solution
        if self.validation_stop:
            self.weights = bssf[1]
        return self

    def print_debug_info(self, x, y, results):
        print("Weights:\n", self.weights)
        print("Input vector:", x)
        print("Target ouput:", y)
        print("Predicted outputs:", results)


    def initialize_weights_to_zero(self, input_size):
        self.network_size = [input_size, *self.hidden_layer_widths, 1]
        self.weights = []
        for i in range(len(self.network_size)):
            # create 2D matrix for each layer of weights
            if i < len(self.network_size)-1: self.weights.append(np.zeros((self.network_size[i]+1, self.network_size[i+1])))
        return self.weights


    def initialize_weights(self, input_size=2, output_size=1):
        self.network_size = [input_size, *self.hidden_layer_widths, output_size]
        self.weights = []
        for i in range(len(self.network_size)):
            # create 2D matrix for each layer of weights
            if i < len(self.network_size)-1: self.weights.append(np.random.normal(0, .1, [self.network_size[i]+1, self.network_size[i+1]]))
        return self.weights

    def shape(self, weights):
        shape = []
        for weight in weights:
            shape.append(weight.shape)
        return shape
    
    def _one_hot_encode(self, data):
        return self.lb.fit(data).transform(data)

    # takes a 2D input array and predicts an output based on current MLP
    def predict(self, X):
        results = []
        for i in range(len(X)):
            result = self.forward(X[i])[-1]
            if len(result) > 0: results.append(np.argmax(result))
        return results

    # Takes 2D array of input and target data and returns the accuracy or mse of the model on the input
    def score(self, X, y, error=False):
        y = self._one_hot_encode(y)
        results = self.lb.transform(self.predict(X))
        if error: return self.mse(results, y)
        return self.accuracy(results, y)
    
    def _validation_error(self):
        if not self.validation_stop: return 0
        return self.mse(self.lb.transform(self.predict(self.X_validation)), self.y_validation)
    
    def _validation_accuracy(self):
        if not self.validation_stop: return 0
        return self.accuracy(self.lb.transform(self.predict(self.X_validation)), self.y_validation)

    # returns the mean squared error on the data set
    def mse(self, y_hat, y_truth):
        return np.max(np.square(y_hat - y_truth), axis=1).mean(axis=None)

    # Accuracy based on percent correct
    def accuracy(self, y_hat, y_truth):
        return 1 - np.max(np.abs(y_hat-y_truth), axis=1).mean(axis=None)

    # shuffles the data so that each epoch presents the data in a different order
    def _shuffle_data(self, X, y):
        if self.shuffle:
            x_len = len(X[0])
            combined = np.column_stack((X, y))
            np.random.shuffle(combined)
            return (combined[:, 0:x_len], combined[:,x_len:])
        return X, y

    def get_weights(self):
        return self.weights

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1.0 - x)
    
    def error_results(self, x_test, y_test):
        return self.training_error[-1], self.validation_error[-1], self.score(x_test, y_test, error=True)
    
    def error_analytics(self):
        return {
            "Training error": self.training_error,
            "Validation error": self.validation_error,
            "Validation accuracy": self.validation_accuracy
        }