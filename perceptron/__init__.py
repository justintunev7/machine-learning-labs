import arff
import matplotlib.pyplot as plt
import numpy as np
from perceptron import PerceptronClassifier
from sklearn.linear_model import Perceptron
import sys


def setup_data(file):
    print("File =",file)
    mat = arff.Arff(file,label_count=1)
    data = mat.data[:,0:-1]
    labels = mat.data[:,-1].reshape(-1,1)
    return data, labels

def scikit_setup_data(file):
    print("File =",file)
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

def plot_misclassification_rate(misclass):
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

def part_one_a():
    print("\n\npart_one_a")
    data, labels = setup_data("./linsep2nonorigin.arff")
    PClass = PerceptronClassifier(lr=0.1,shuffle=False,deterministic=10)
    PClass.fit(data,labels)
    Accuracy = PClass.score(data,labels)
    print("Accuracy = [{:.2f}]".format(Accuracy))
    print("Final Weights =",PClass.get_weights())

def part_one_b():
    print("\n\npart_one_b")
    data, labels = setup_data("./data_banknote_authentication.arff")
    PClass = PerceptronClassifier(lr=0.1,shuffle=False,deterministic=10)
    PClass.fit(data,labels)
    Accuracy = PClass.score(data,labels)
    print("Accuracy = [{:.2f}]".format(Accuracy))
    print("Final Weights =",PClass.get_weights())



def part_three_a():
    print("\n\npart_three_a")
    data, labels = setup_data("./test1_data.arff")
    for lr in [.001, .01, .1]:
        PClass = PerceptronClassifier(lr=lr,shuffle=False,deterministic=10)
        PClass.fit(data,labels)
        Accuracy = PClass.score(data,labels)    
        print("Learning Rate =",lr,"Accuracy = [{:.2f}]".format(Accuracy),"Final Weights =",PClass.get_weights())
    plot_descision_line(data, labels, PClass.get_weights())

def part_three_b():
    print("\n\npart_one_b")
    data, labels = setup_data("./test2_data.arff")
    for lr in [.001, .01, .1]:
        PClass = PerceptronClassifier(lr=lr,shuffle=False,deterministic=10)
        PClass.fit(data,labels)
        Accuracy = PClass.score(data,labels)    
        print("Learning Rate =",lr,"Accuracy = [{:.2f}]".format(Accuracy),"Final Weights =",PClass.get_weights())
    plot_descision_line(data, labels, PClass.get_weights())


def part_four():
    print("\n\npart_four")
    data, labels = setup_data("./voting_data.arff")
    misclassifications = []
    for i in range(5):
        PClass = PerceptronClassifier()
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
    plot_misclassification_rate(misclassifications)


def part_five_a():
    print("\n\npart_five_a")
    data, labels = scikit_setup_data("./voting_data.arff")
    PClass = Perceptron(max_iter=100, tol=.01, validation_fraction=.1)
    PClass.fit(data, labels)
    print("SCIKIT-LEARN Testing accuracy",PClass.score(data, labels))

    data, labels = setup_data("./voting_data.arff")
    PClass = PerceptronClassifier()
    PClass.fit(data, labels)
    print("PerceptronClassifier Testing accuracy",PClass.score(data, labels))

def part_five_b():
    print("\n\npart_five_b")
    data, labels = scikit_setup_data( "./diabetes.arff")
    PClass = Perceptron(max_iter=100, tol=.01, validation_fraction=.1)
    PClass.fit(data, labels)
    print("SCIKIT-LEARN Testing accuracy",PClass.score(data, labels))

    data, labels = setup_data( "./diabetes.arff")
    PClass = PerceptronClassifier()
    PClass.fit(data, labels)
    print("PerceptronClassifier Testing accuracy",PClass.score(data, labels))


def main():
    part_one_a()
    part_one_b()
    part_three_a()
    part_three_b()
    part_four()
    part_five_a()
    part_five_b()


if __name__ == '__main__':
    main()