from arff import Arff
import matplotlib.pyplot as plt
import numpy as np
from desiciontree import DTClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
import sys
import copy
import math

def sklearn_cross_validation_tests(data, labels, counts, max_depth=None, min_samples_split=2):
    accuracies = []
    for train_index, test_index in cross_validation(data, n_splits=10):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        DTClass = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
        DTClass.fit(X_train,y_train)
        accuracies.append(DTClass.score(X_test, y_test))
        print(DTClass.feature_importances_)
    # if display_tree: DTClass.display_tree()
    del DTClass
    return accuracies

def part_five():
    print("PART FIVE (sklearn)")
    data, labels, counts = setup_data("./cars.arff", clean=True)
    accuracies = sklearn_cross_validation_tests(data, labels, counts)
    print("Average Accuracy CARS (sklearn) max_depth=", None, "and min_samples_split=", 2, ":", sum(accuracies) / len(accuracies))
    for i in range(5):
        accuracies = sklearn_cross_validation_tests(data, labels, counts, max_depth=i+2, min_samples_split=i+2)
        # print("ACCURACIES CARS (sklearn):", accuracies)
        print("Average Accuracy CARS (sklearn) max_depth=", i+2, "and min_samples_split=", i+2, ":", sum(accuracies) / len(accuracies))

    data, labels, counts = setup_data("./voting.arff", clean=True)
    accuracies = sklearn_cross_validation_tests(data, labels, counts)
    print("Average Accuracy VOTING (sklearn) max_depth=", None, "and min_samples_split=", 2, ":", sum(accuracies) / len(accuracies))
    for i in range(5):
        accuracies = sklearn_cross_validation_tests(data, labels, counts, max_depth=i+2, min_samples_split=i+2)
        # print("ACCURACIES CARS (sklearn):", accuracies)
        print("Average Accuracy VOTING (sklearn) max_depth=", i+2, "and min_samples_split=", i+2, ":", sum(accuracies) / len(accuracies))


def setup_data(file, clean=False):
    print("\nFile =",file)
    mat = Arff(file,label_count=1)
    counts = []
    for i in range(mat.data.shape[1]):
        counts += [mat.unique_value_count(i)]
    data = mat.data[:,0:-1]
    labels = mat.data[:,-1]
    if clean: data, labels, counts = clean_data(data, labels, counts)
    return data, labels, counts

def clean_data(data, labels, counts):
    cleaned_counts = copy.copy(counts)
    for i, instance in enumerate(data):
        if math.isnan(labels[i]):
            cleaned_counts[-1] = counts[-1] + 1
            labels[i] = counts[-1]
        for j, feature in enumerate(instance):
            if math.isnan(feature):
                cleaned_counts[j] = counts[j] + 1
                data[i][j] = counts[j]
    return data, labels, cleaned_counts

def split_data(X, y, test_split=.25):
    # create test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
    return X_train, X_test, y_train, y_test

def cross_validation(X, n_splits=10):
    return KFold(n_splits=n_splits, shuffle=True).split(X)

def cross_validation_tests(data, labels, counts, display_tree=False):
    accuracies = []
    for train_index, test_index in cross_validation(data, n_splits=10):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        DTClass = DTClassifier(counts)
        DTClass.fit(X_train,y_train)
        accuracies.append(DTClass.score(X_test, y_test))
    if display_tree: DTClass.display_tree()
    del DTClass
    return accuracies

def part_three():
    part_two(display_tree=True)

def part_two(display_tree=False):
    print("\nPART THREE:") if display_tree else print("\nPART TWO:")
    data, labels, counts = setup_data("./cars.arff")
    accuracies = cross_validation_tests(data, labels, counts, display_tree=display_tree)
    print("ACCURACIES CARS:", accuracies)
    print("Average Accuracy CARS:", sum(accuracies) / len(accuracies))
    # for train_index, test_index in cross_validation(data, n_splits=10):
    #     X_train, X_test = data[train_index], data[test_index]
    #     y_train, y_test = labels[train_index], labels[test_index]
    #     DTClass = DTClassifier(counts)
    #     DTClass.fit(X_train,y_train)
    #     accuracies.append(DTClass.score(X_test, y_test))
    #     del DTClass
    # print("ACCURACIES CARS:", accuracies)
    # print("Average Accuracy CARS:", sum(accuracies) / len(accuracies))

    accuracies = []
    data, labels, counts = setup_data("./voting.arff", clean=True)
    accuracies = cross_validation_tests(data, labels, counts, display_tree=display_tree)
    print("ACCURACIES CARS:", accuracies)
    print("Average Accuracy VOTING:", sum(accuracies) / len(accuracies))
    # for i in range(5):
    #     print(i)
    #     X_train, X_test, y_train, y_test = split_data(data, labels)
    #     DTClass = DTClassifier(counts)
    #     DTClass.fit(X_train,y_train)
    #     accuracies.append(DTClass.score(X_test, y_test))
    #     del DTClass
    # print("Average Accuracy VOTING:", sum(accuracies) / len(accuracies))


def part_one():
    accuracies = []
    data, labels, counts = setup_data("./lenses.arff")
    for i in range(5):
        print(i)
        X_train, X_test, y_train, y_test = split_data(data, labels)
        DTClass = DTClassifier(counts)
        DTClass.fit(X_train,y_train)
        accuracies.append(DTClass.score(X_test, y_test))
        del DTClass
    print("Average Accuracy:", sum(accuracies) / len(accuracies))


def debug():
    mat = Arff("./lenses.arff",label_count=1)
    counts = [] ## this is so you know how many types for each column

    for i in range(mat.data.shape[1]):
        counts += [mat.unique_value_count(i)]
    data = mat.data[:,0:-1]
    labels = mat.data[:,-1]
    DTClass = DTClassifier(counts)
    DTClass.fit(data,labels).get_root().print_tree()
    mat2 = Arff("./all_lenses.arff", label_count=1)
    data2 = mat2.data[:,0:-1]
    labels2 = mat2.data[:,-1]
    pred = DTClass.predict(data2)
    print("FINISHED!")
    Acc = DTClass.score(data2,labels2)

    # write to CSV file
    np.savetxt("pred_lenses.csv",pred,delimiter=",")
    print("Accuracy = [{:.2f}]".format(Acc))

def main():
    print("Starting tests")
    # debug()
    # part_one()
    # part_two()
    # part_three()
    # part_four()
    part_five()
    # part_six()
    return



if __name__ == '__main__':
    main()