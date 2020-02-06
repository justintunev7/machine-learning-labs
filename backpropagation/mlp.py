import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

# class Perceptron():
#     def __init__(self, size=3):
#         self.weights = self.initialize_weights(size + 1)

#      def initialize_weights(self, size=2):
#         self.weights = np.random.normal(0, .5, size)
#         return self.weights
    
#     def get_weights(self):
#         return self.weights

class MLPClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, hidden_layer_widths, lr=.1, momentum=0, shuffle=True):
        """ Initialize class with chosen hyperparameters.

        Args:
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.

        Example:
            mlp = MLPClassifier([3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        self.hidden_layer_widths = np.array(hidden_layer_widths)
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle
        
        # self.weights = self.initialize_weights()
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1.0 - x)
    # prev is input, curr is weights of current layer
    def single_layer_forward_prop(self, prev, curr):
        # print("PREV=",prev)
        # print("CURR=", curr)
        return np.matmul(prev, curr)
    
    def forward(self, x):
        # print(self.weights)
        # # 2 array
        # print(self.weights[0])
        # # 3 array
        # print(self.weights[:,0])

    def fit(self, X, y, initial_weights=None):
        self.weights = initial_weights = self.initialize_weights(len(X[0])) if initial_weights is None else initial_weights
        # epochs
        # for i in range(10):
        
        # for each input vector
        for j in range(len(X)):
            in_val = np.append(X[j], 1)

            out = self.forward(in_val)
            # go through layers
            # for weight in self.weights:
            #     print(weight)
                # for m in range(len(self.weights[k])):   
                # print(self.weights)
                # print(self.weights[:,0], self.weights[:,1])
                # in_val = self.single_layer_forward_prop(in_val, self.weights[:,k])

                # for m in range(len(self.weights[k])):
                #     # single perceptron
                #     print(self.single_layer_forward_prop())

        return self
        # count = 0
        # prev_score = self.score(X, y)
        # # If deterministic is set, then just run n times, if not, continue until n epochs result in no change
        # while (self.epochs < self.deterministic and count < self.stopping_count):
        #     self.epochs += 1
        #     initial_weights = self.weights
        #     # shuffle data if specified
        #     if self.shuffle: X, y = self._shuffle_data(X, y)
        #     for i in range(len(X)):
        #         net = initial_weights.dot(np.append(X[i], 1))
        #         z = 1 if net > 0 else 0 # where z is the output found with current model
        #         a = (y[i] - z)
        #         if (a != 0):
        #             change_weights = (np.append(X[i], 1) * (self.lr * a))
        #             initial_weights = initial_weights + change_weights
        #     temp = self.weights
        #     self.weights = initial_weights
        #     current_score = self.score(X,y)
        #     # only update if the current_score is better than the prev_score
        #     if (current_score <= prev_score+.01) and self.deterministic==10000:
        #         self.misclassifications.append(1-prev_score)
        #         count += 1
        #         self.weights = temp
        #     else:
        #         self.misclassifications.append(1-current_score)
        #         prev_score = current_score
        #         count = 0
        # return self

        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        # self.initial_weights = self.initialize_weights() if not initial_weights else initial_weights

        # return self

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

    def initialize_weights(self, input_size=2):
        self.network_size = [input_size, *self.hidden_layer_widths, 0]
        # add bias
        self.network_size = [x+1 for x in self.network_size]
        print("NETWORK SIZE", self.network_size)
        self.weights = np.random.normal(0, .5, self.network_size)
        print("WEIGHTS=",self.weights.shape)
        return self.weights

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

        return 0

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
                shuffling X and y exactly the same way, independently.
        """
        x_len = len(X[0])
        combined = np.column_stack((X, y))
        np.random.shuffle(combined)
        return (combined[:, 0:x_len], combined[:,x_len:])

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights
