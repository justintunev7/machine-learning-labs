import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class Node():
    def __init__(self, num_weights):
        self.weights = np.random.normal(0, .5, self.network_size)

    def get_weights(self):
        return self.weights

class MLPClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, hidden_layer_widths, lr=.1, momentum=0, shuffle=True, deterministic=1):
        self.hidden_layer_widths = np.array(hidden_layer_widths)
        self.lr = lr
        self.momentum = momentum
        self.deterministic = deterministic
        self.shuffle = shuffle
    

    def backward(self, results, y_truth):
        deltas = [None] * len(results)
        delta_weights = [None] * len(self.weights)
        for i, layer in reversed(list(enumerate(results))):
            # print(i, len(results)-1)
            if i < len(results)-1:
                # layer = layer[0:-1]
                deltas[i] = (np.dot(self.weights[i][0:-1], deltas[i+1]))
                print([*results[i], 1], deltas[i+1][:,None])
                delta_weights[i] = np.transpose((np.multiply([*results[i], 1], deltas[i+1][:,None]) * self.lr))
                # results[i] = np.append(results[i],1)
            if i == (len(results) - 1):
                deltas[i] = ((y_truth - layer) * self.sigmoid_derivative(layer))
            # else: 
            #     deltas[i] = (np.dot(self.weights[i+1], deltas[i+1]))
        # delta_weights[0] = (np.multiply(deltas[1], [*results[0], 1]) * self.lr)
        return delta_weights

    # def backward(self, results, y_truth):
    #     deltas = [None] * len(results)
    #     delta_weights = [None] * len(self.weights)
    #     print("RESULTS",results)
    #     print("YTRUTH", y_truth)
    #     for i, layer in reversed(list(enumerate(results))):
    #         deltas[i] = []
    #         # remove bias
    #         # if i == 0: break
    #         if i < len(results)-1: layer = layer[0:-1]
    #         delta_weights[i-1] = np.zeros((len(layer), len(results[i-1])))
    #         for j, node in enumerate(layer):
    #             print(i,j)
    #             if i == (len(results) - 1): d = (y_truth - node) * self.sigmoid_derivative(node)
    #             # for each node in next layer (k), get SUM(d(k) * W(current to k))
    #             else:
    #                 # print("WEIGHTS",self.weights[i][j])
    #                 # print("DELTAS",deltas)
    #                 d = np.sum(deltas[i+1] * self.weights[i][j]) * self.sigmoid_derivative(node)
    #             deltas[i].append(d)
    #             # print("DELTAS",deltas)
    #             delta_weights[i-1][j] = np.multiply(results[i-1], d) * self.lr
    #         delta_weights[i-1] = np.transpose(delta_weights[i-1])
    #     return delta_weights

    def single_forward(self, input_values, weights):
        return input_values.dot(weights)

    def forward(self, x):
        results = []
        results.append(x)
        for i in range(len(self.weights)): # for each layer in network
            x = np.append(x, 1)
            x = self.single_forward(x, self.weights[i])
            if i < len(self.weights): 
                x = self.sigmoid(x)
            results.append(x)
        return results

    def fit(self, X, y, initial_weights=None):
        self.weights = initial_weights = self.initialize_weights(len(X[0])) if initial_weights is None else initial_weights
        # for a in range(10):
        y_hat = []
        for a in range(self.deterministic):
            print("--Epoch {}--".format(a+1))
            for i in range(1):
                print("Weights:\n", self.weights)
                print("Input vector:", X[i])
                print("Target ouput:", y[i])
                print("Forward propagating...")
                results = self.forward(X[i])
                print("Predicted outputs:", results)
                y_hat.append(results[-1])
                print("Back propagating...")
                delta_weights = self.backward(results, y[i])
                print("Deltas:", delta_weights)
                print("WEIGHTS:", self.weights)
                print("Descending gradient...")
                self.weights = np.add(self.weights, delta_weights)
                print("Weights after:\n", self.weights)
            print(self.accuracy(y_hat, y))
        return self

    def initialize_weights_to_zero(self, input_size):
        self.network_size = [input_size, *self.hidden_layer_widths, 1]
        self.weights = []
        for i in range(len(self.network_size)):
            # create 2D matrix for each layer of weights
            # if i < len(self.network_size)-1: self.weights.append(np.random.normal(0, .1, [self.network_size[i]+1, self.network_size[i+1]]))
            if i < len(self.network_size)-1: self.weights.append(np.zeros((self.network_size[i]+1, self.network_size[i+1])))
        print(self.shape(self.weights))
        return self.weights


    def initialize_weights(self, input_size=2):
        self.network_size = [input_size, *self.hidden_layer_widths, 1]
        self.weights = []
        for i in range(len(self.network_size)):
            # create 2D matrix for each layer of weights
            if i < len(self.network_size)-1: self.weights.append(np.random.normal(0, .1, [self.network_size[i]+1, self.network_size[i+1]]))
            # if i < len(self.network_size)-1: self.weights.append(np.zeros((self.network_size[i]+1, self.network_size[i+1])))
        print(self.shape(self.weights))
        return self.weights

    def shape(self, weights):
        shape = []
        for weight in weights:
            shape.append(weight.shape)
        return shape
# Weights:                                                                 
    # 0.1, 0.2, -0.1,                                                      
    # -0.2, 0.3, -0.3,                                                     

    # 0.1, -0.2, -0.3, 
    # 0.2, -0.1, 0.3,  

    # 0.2, -0.1, 0.3, 
    # 0.1, -0.2, -0.3 


    def predict(self, X):
        results = []
        for i in range(len(X)):
            net = self.weights.dot(np.append(X[i], 1))
            z = 1 if net > 0 else 0 # where z is the output found with current model
            results.append(z)
        return results
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
    
    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        return np.square(np.subtract(X, y)).mean()

    def accuracy(self, y_hat, y_truth):
        loss = 0
        # print(y_hat, y_truth)
        for i, result in enumerate(y_hat):
            # print(result, y_truth[i][0])
            if abs(y_truth[i][0] - result) != 0: loss += 1
        # print(loss, len(y_hat))
        return 1 - loss/len(y_hat)

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
                shuffling X and y exactly the same way, independently.
        """
        x_len = len(X[0])
        combined = np.column_stack((X, y))
        np.random.shuffle(combined)
        return (combined[:, 0:x_len], combined[:,x_len:])

    def get_weights(self):
        return self.weights

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1.0 - x)