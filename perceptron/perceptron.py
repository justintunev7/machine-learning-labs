import arff
import math
import time
import matplotlib.pyplot as plt
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

    def __init__(self, lr=.1, shuffle=True, deterministic=10000):
        self.lr = lr
        self.shuffle = shuffle
        self.deterministic = deterministic
        self.stopping_count = 30
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
            if (current_score <= prev_score+.01):
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
        return results#, len(reuslts)


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
            for j in range(len(misclass)):
                if len(misclass[j]) > i: temp.append(misclass[j][i])
            # print(len(temp))
            y.append(sum(temp)/(len(temp)))
        plt.plot(range(len(y)), y)
        plt.show()

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights
    
    # Returns the number of epochs done during training.
    def get_epochs(self):
        return self.epochs
    
    def get_missclassifications(self):
        return self.misclassifications

# create training/testing sets
# mat = arff.Arff("../data/perceptron/debug/linsep2nonorigin.arff",label_count=1)
# mat = arff.Arff("../data/perceptron/evaluation/data_banknote_authentication.arff",label_count=1)
# mat = arff.Arff("./test2_data.arff",label_count=1)
# data = mat.data[:,0:-1]
# data = mat.data[:,0:-1]
# labels = mat.data[:,-1].reshape(-1,1)
# PClass = PerceptronClassifier(lr=0.1,shuffle=False,deterministic=10)
# train_data, train_labels, test_data, test_labels = PClass.split_data_sets(data,labels)
# PClass.fit(train_data,train_labels)
# Accuracy = PClass.score(test_data,test_labels)
# print("Accuracy = [{:.2f}]".format(Accuracy))
# print("Final Weights =",PClass.get_weights())

# argv = [perceptron.py, import_arff_file]
# mat = arff.Arff("../data/perceptron/debug/linsep2nonorigin.arff",label_count=1)
# mat = arff.Arff("../data/perceptron/evaluation/data_banknote_authentication.arff",label_count=1) "./test2_data.arff"
if len(sys.argv) == 4:
    file, lr, deterministic = sys.argv[1:]
    PClass = PerceptronClassifier(lr=float(lr),deterministic=int(deterministic))
elif len(sys.argv) == 3:
    file, lr = sys.argv[1:]
    PClass = PerceptronClassifier(lr=float(lr))
else:
    file, lr, deterministic = "./linsep2nonorigin.arff", .1, 10
    PClass = PerceptronClassifier(lr=float(lr),shuffle=False,deterministic=int(deterministic))

mat = arff.Arff(file,label_count=1)
data = mat.data[:,0:-1]
data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)
print("File =",file)
print("Learning rate =",lr)
misclassifications = []
for i in range(5):
    PClass = PerceptronClassifier(lr=float(lr))
    train_data, train_labels, test_data, test_labels = PClass.split_data_sets(data,labels)
    PClass.fit(train_data,train_labels)
    test_accuracy = PClass.score(test_data,test_labels)
    train_accuracy = PClass.score(train_data,train_labels)
    print("\n")
    print("Number of Epochs =",PClass.get_epochs())
    print("Training Accuracy = [{:.2f}]".format(train_accuracy))
    print("Test Accuracy = [{:.2f}]".format(test_accuracy))
    print("Final Weights =",['{:.2f}'.format(weight) for weight in PClass.get_weights()])
    misclassifications.append(PClass.get_missclassifications())
# print(misclassifications)
PClass.plot_misclassification_rate(misclassifications)
# for i in misclassifications:
#     print(['{:.2f}'.format(a) for a in i])

# mat = arff.Arff(file,label_count=1)
# data = mat.data[:,0:-1]
# labels = mat.data[:,-1].reshape(-1,1)

# PClass.fit(data,labels)
# Accuracy = PClass.score(data,labels)

# PClass.plot_descision_line(data, labels)
