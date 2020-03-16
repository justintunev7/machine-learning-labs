from arff import Arff
import matplotlib.pyplot as plt
import numpy as np
from KNN import KNNClassifier
from sklearn.model_selection import train_test_split
import sys
import copy
import math

def setup_data(file, label_count=1):
    print("\nFile =",file)
    mat = Arff(file,label_count=label_count)
    raw_data = mat.data
    h,w = raw_data.shape
    data = raw_data[:,:-1]
    labels = raw_data[:,-1]
    return data, labels

def normalize(data):
    maxes = data.max(axis=1)
    mins = data.min(axis=1)
    normal = (data - mins[:,None]) / (maxes - mins)[:,None]
    return normal

def split_data(X, y, test_split=.25):
    # create test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
    return X_train, X_test, y_train, y_test

def two():
    train_data, train_labels = setup_data("magic_telescope_train.arff")
    test_data, test_labels = setup_data("magic_telescope_test.arff")
    # without normalizing
    # KNN = KNNClassifier(labeltype ='classification',weight_type='inverse_distance', k=3)
    # KNN.fit(train_data,train_labels)
    # pred = KNN.predict(test_data)
    # error, accuracy = KNN.score(test_data,test_labels)
    # print("ERROR:", error)
    # print("ACCURACY:", accuracy)

    # normalize
    norm_train_data = normalize(train_data)
    norm_test_data = normalize(test_data)
    KNN = KNNClassifier(labeltype ='classification',weight_type='inverse_distance', k=3)
    KNN.fit(norm_train_data,train_labels)
    pred = KNN.predict(norm_test_data)
    error, accuracy = KNN.score(norm_test_data,test_labels)
    print("NORMALIZED ERROR:", error)
    print("NORMALIZED ACCURACY:", accuracy)


    k = 3
    while k <= 15:
        KNN = KNNClassifier(labeltype ='classification',weight_type='inverse_distance', k=k)
        KNN.fit(norm_train_data,train_labels)
        pred = KNN.predict(norm_test_data)
        error, accuracy = KNN.score(norm_test_data,test_labels)
        print("NORMALIZED ERROR for k =", k, ":", error)
        print("NORMALIZED ACCURACY for k =", k, ":", accuracy)
        k += 2

def one():
    train_data, train_labels = setup_data("diabetes.arff")
    test_data, test_labels = setup_data("diabetes_test.arff")
    KNN = KNNClassifier(labeltype ='classification',weight_type='inverse_distance', k=3)
    KNN.fit(train_data,train_labels)
    pred = KNN.predict(test_data)
    error, accuracy = KNN.score(test_data,test_labels)
    print("ERROR:", error)
    print("ACCURACY:", accuracy)
    # print("PREDICTION",pred)
    np.savetxt("seismic-bump-prediction.csv",pred,delimiter=',',fmt="%i")

def debug():
    train_data, train_labels = setup_data("seismic-bumps_train.arff")
    test_data, test_labels = setup_data("seismic-bumps_test.arff")
    KNN = KNNClassifier(labeltype ='classification',weight_type='inverse_distance', k=3, use_distance_weighting=True)
    KNN.fit(train_data,train_labels)
    pred = KNN.predict(test_data)
    error, accuracy = KNN.score(test_data,test_labels)
    print("ERROR:", error)
    print("ACCURACY:", accuracy)
    # print("PREDICTION",pred)
    np.savetxt("seismic-bump-prediction.csv",pred,delimiter=',',fmt="%i")


def main():
    print("Starting tests")
    debug()
    # two()
    # part_two()
    # part_three()
    # part_five()
    # part_six()
    return



if __name__ == '__main__':
    main()