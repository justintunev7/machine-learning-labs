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


def debug():
    # print("\n\npart_one_a")
    # data, labels = setup_data("./linsep2nonorigin.arff")
    data, labels = setup_data("linsep2nonorigin.arff")
    MLPClass = MLPClassifier(lr=0.1,shuffle=False, hidden_layer_widths=[4])
    MLPClass.fit(data,labels, initial_weights=MLPClass.initialize_weights_to_zero(2))
    weights = MLPClass.get_weights()
    print(weights)
    expected = [-1.0608880e-02,
                -1.0608880e-02, 
                -1.0608880e-02, 
                -1.0608880e-02, 
                -2.1454953e-02, 
                -8.8177979e-05, 
                7.8275773e-04, 
                -3.9435364e-03, 
                -8.8177979e-05, 
                7.8275773e-04, 
                -3.9435364e-03, 
                -8.8177979e-05, 
                7.8275773e-04, 
                -3.9435364e-03, 
                -8.8177979e-05, 
                7.8275773e-04, 
                -3.9435364e-03]
    allowed_error = .000000001
    print(abs(weights[0][0][0] - expected[5]) < allowed_error)
    print(abs(weights[0][1][0] - expected[6]) < allowed_error)
    print(abs(weights[0][2][0] - expected[7]) < allowed_error)
    print(abs(weights[1][0][0] - expected[0]) < allowed_error)
    print(abs(weights[1][-1][0] - expected[4]) < allowed_error)

def part_two():
    data, labels = setup_data("vowel.arff")
    MLPClass = MLPClassifier(lr=0.1,shuffle=False, hidden_layer_widths=[3])
    MLPClass.fit(data,labels)

def part_three():
    pass

def part_four():
    pass

def part_five():
    pass

def main():
    debug()
    return



if __name__ == '__main__':
    main()