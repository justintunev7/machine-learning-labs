import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class Node():
    def __init__(self, num_weights):
        self.weights = np.random.normal(0, .5, self.network_size)

    def get_weights(self):
        return self.weights

class MLPClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, hidden_layer_widths, lr=.1, momentum=0, shuffle=True):
        self.hidden_layer_widths = np.array(hidden_layer_widths)
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle

    # prev is input, curr is weights of current layer
    def single_layer_forward_prop(self, input_values, weights):
        return input_values.dot(weights)

    def forward(self, x):
        for i in range(len(self.weights)):
            x = self.single_layer_forward_prop(np.append(x, 1), self.weights[i])
        return x

    def fit(self, X, y, initial_weights=None):
        self.weights = initial_weights = self.initialize_weights(len(X[0])) if initial_weights is None else initial_weights
        for i in range(len(X)):
            y_hat = self.forward(X[i])
            print(y_hat, y[i])
        return self


    def initialize_weights(self, input_size=2):
        self.network_size = [input_size, *self.hidden_layer_widths, 1]
        self.weights = []
        for i in range(len(self.network_size)):
            # create 2D matrix for each layer of weights
            if i < len(self.network_size)-1: self.weights.append(np.random.normal(0, .5, [self.network_size[i]+1, self.network_size[i+1]]))
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
        predicted_results = self.predict(X)
        loss = 0
        for i, result in enumerate(predicted_results):
            loss += abs((y[i][0] - result))
        return 1 - loss/len(predicted_results)

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