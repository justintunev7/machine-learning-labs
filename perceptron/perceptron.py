import arff
import numpy as np
import sys
from sklearn.base import BaseEstimator, ClassifierMixin

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

from sklearn.linear_model import Perceptron

class PerceptronClassifier(BaseEstimator,ClassifierMixin):

        # """ Initialize class with chosen hyperparameters.
        # Args:
        #     lr (float): A learning rate / step size.
        #     shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        # """
    def __init__(self, lr=.1, shuffle=True, deterministic=10):
        self.lr = lr
        self.shuffle = shuffle
        self.deterministic = deterministic


        # """ Fit the data; run the algorithm and adjust the weights to find a good solution
        # Args:
        #     X (array-like): A 2D numpy array with the training data, excluding targets
        #     y (array-like): A 2D (1D?) numpy array with the training targets
        #     initial_weights (array-like): allows the user to provide initial weights
        # Returns:
        #     self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        # """
    def fit(self, X, y, initial_weights=None):
        initial_weights = self.initialize_weights(len(X[0]) + 1) if initial_weights is None else initial_weights
        for j in range(self.deterministic):
            for i in range(len(X)):
                net = initial_weights.dot(np.append(X[i], 1))
                z = 1 if net > 0 else 0 # where z is the output found with current model
                change_weights = (np.append(X[i], 1) * (self.lr * (y[i] - z)))
                initial_weights = initial_weights + change_weights
        self.weights = initial_weights
        return self


        # """ Predict all classes for a dataset X
        # Args:
        #     X (array-like): A 2D numpy array with the training data, excluding targets
        # Returns:
        #     array, shape (n_samples,)
        #         Predicted target values per element in X.
        # """
    def predict(self, X):
        pass


        # """ Initialize weights for perceptron. Don't forget the bias!
        # Returns:
        # """
    def initialize_weights(self, size = 2):
        return np.zeros(size)


        # """ Return accuracy of model on a given dataset. Must implement own score function.
        # Args:
        #     X (array-like): A 2D numpy array with data, excluding targets
        #     y (array-like): A 2D numpy array with targets
        # Returns:
        #     score : float
        #         Mean accuracy of self.predict(X) wrt. y.
        # """
    def score(self, X, y):
    
        return 0


        # """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
        #     It might be easier to concatenate X & y and shuffle a single 2D array, rather than
        #      shuffling X and y exactly the same way, independently.
        # """
    def _shuffle_data(self, X, y):
        pass

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights

# argv = [perceptron.py, import_arff_file]
# if len(sys.argv) > 1:
mat = arff.Arff("../data/perceptron/debug/linsep2nonorigin.arff",label_count=1)
data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)
PClass = PerceptronClassifier(lr=0.1,shuffle=False,deterministic=10)
PClass.fit(data,labels)
Accuracy = PClass.score(data,labels)
print("Accuray = [{:.2f}]".format(Accuracy))
print("Final Weights =",PClass.get_weights())







    # arff_obj = arff.Arff(arff=sys.argv[1], label_count=3)
    # perceptron = PerceptronClassifier(.1, True)
    # y = np.empty(0)
    # for i, row in enumerate(arff_obj):
    #     if i==0:
    #         X = np.array(row[0:-1])
    #     else:
    #         X = np.vstack((X, row[0:-1]))
    #     y = np.append(y, row[-1])
    # results = perceptron.fit(X, y)
    # for i in range(10):
    #     results = perceptron.fit(X, y, results.get_weights())
    # print(results.get_weights())