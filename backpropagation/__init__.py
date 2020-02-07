import arff
import matplotlib.pyplot as plt
import numpy as np
from mlp import MLPClassifier
from sklearn.linear_model import Perceptron
import sys

# def create_weight_matrix(hidden_)
def setup_data(file):
    # print("File =",file)
    mat = arff.Arff(file,label_count=1)
    data = mat.data[:,0:-1]
    labels = mat.data[:,-1].reshape(-1,1)
    return data, labels

def scikit_setup_data(file):
    # print("File =",file)
    mat = arff.Arff(file,label_count=1)
    data = mat.data[:,0:-1]
    labels = mat.data[:,-1]
    return data, labels

def plot_descision_line(X, y, weights):
    a = np.empty((0,len(X[0])))
    b = np.empty((0,len(X[0])))
    for i in range(len(X)):
        if y[i][0] == 1: a = np.vstack((a, X[i]))
        else: b = np.vstack((b, X[i]))
    plt.scatter(a[:,0], a[:,1], label='1')
    plt.scatter(b[:,0], b[:,1], label='0')

    slope = -weights[0]/weights[1]
    x = [-1, 1]
    plt.plot(x, [x[0]*slope - weights[2], x[1]*slope - weights[2]])
    plt.xlabel("var 1")
    plt.ylabel("var 2")
    plt.title("Instances and Decision Line")
    plt.show()


def part_one():
    # print("\n\npart_one_a")
    # data, labels = setup_data("./linsep2nonorigin.arff")
    data, labels = setup_data("./test2.arff")
    MLPClass = MLPClassifier(lr=0.1,shuffle=False, hidden_layer_widths=[3])
    MLPClass.fit(data,labels)
    # Accuracy = MLPClass.score(data,labels)
    # print("Accuracy = [{:.2f}]".format(Accuracy))
    # print("Final Weights =",MLPClass.get_weights())

def part_two():
    pass

def part_three():
    pass

def part_four():
    pass

def part_five():
    pass

def main():
    # mlp = MLPClassifier(hidden_layer_widths=[3,4,5])
    # weights = mlp.initialize_weights(input_size=3)
    # print(weights.shape)
    # print(weights)
    part_one()
    return
    # part_one()
    # part_one()
    # part_three()
    # part_four()
    # part_five()


if __name__ == '__main__':
    main()