from arff import Arff
import matplotlib.pyplot as plt
import numpy as np
from KNN import KNNClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import sys
import copy
import math


def plot_bar(data, labels=None, xlabel="", ylabel="", title=""):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if labels != None:
        plt.xticks(range(len(labels)), labels)
    if type(data) == dict:
        width = 1 / (len(data) + 1)
        for i, key in enumerate(data):
            value = data[key]
            offset = width * (i - len(data) // 2)
            plt.bar(np.arange(len(value)) + offset,
                    value, width=width, label=key)
        plt.legend()
    else:
        plt.bar(range(len(data)), data)
    plt.show()


def setup_data(file, label_count=1):
    print("\nFile =", file)
    mat = Arff(file, label_count=label_count)
    raw_data = mat.data
    h, w = raw_data.shape
    data = raw_data[:, :-1]
    labels = raw_data[:, -1]
    return data, labels


def split_data(X, y, test_split=.25):
    # create test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split)
    return X_train, X_test, y_train, y_test


def normalize(data):
    maxes = data.max(axis=1)
    mins = data.min(axis=1)
    normal = (data - mins[:, None]) / (maxes - mins)[:, None]
    return normal


def split_data(X, y, test_split=.25):
    # create test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split)
    return X_train, X_test, y_train, y_test

def seven():
    train_data, train_labels = setup_data("magic_telescope_train.arff")
    test_data, test_labels = setup_data("magic_telescope_test.arff")
    # normalize
    norm_train_data = normalize(train_data)
    norm_test_data = normalize(test_data)
    KNN = KNeighborsClassifier(n_neighbors=3, p=1)
    KNN.fit(norm_train_data, train_labels)
    accuracy = KNN.score(norm_test_data, test_labels)
    print("k:", k, "p: 1")
    print("SKLearn magic telescope ACCURACY:", accuracy)

def six():
    train_data, train_labels = setup_data("magic_telescope_train.arff")
    test_data, test_labels = setup_data("magic_telescope_test.arff")
    # normalize
    norm_train_data = normalize(train_data)
    norm_test_data = normalize(test_data)
    print(len(test_labels))
    print(len(test_data))
    accuracies_k = []
    for i in range(4):
        k = 1 + (i * 2)
        KNN = KNeighborsClassifier(n_neighbors=k, p=1)
        KNN.fit(norm_train_data, train_labels)
        accuracy = KNN.score(norm_test_data, test_labels)
        print("k:", k, "p: 1")
        print("SKLearn magic telescope ACCURACY:", accuracy)
        accuracies_k.append(accuracy)

        KNN = KNeighborsClassifier(n_neighbors=k, p=2)
        KNN.fit(norm_train_data, train_labels)
        pred = KNN.predict(norm_test_data)
        accuracy = KNN.score(norm_test_data, test_labels.astype(int))
        print("k:", k, "p: 2")
        print("SKLearn magic telescope ACCURACY:", accuracy)

    train_data, train_labels = setup_data("house_price_train.arff")
    test_data, test_labels = setup_data("house_price_test.arff")
    # normalize
    norm_train_data = normalize(train_data)
    norm_test_data = normalize(test_data)
    for i in range(5):
        k = 1 + (i * 2)
        KNN = KNeighborsRegressor(n_neighbors=k, p=1,weights='distance', algorithm='auto')
        KNN.fit(norm_train_data, train_labels)
        accuracy = KNN.score(norm_test_data, test_labels)
        print("k:", k, "p: 1")
        print("SKLearn housing ACCURACY:", accuracy)

        KNN = KNeighborsRegressor(n_neighbors=k, p=2,weights='distance', algorithm='auto')
        KNN.fit(norm_train_data, train_labels)
        accuracy = KNN.score(norm_test_data, test_labels)
        print("k:", k, "p: 2")
        print("SKLearn housing ACCURACY:", accuracy)

def five():
    data, labels = setup_data("credit_approval.arff")
    X_train, X_test, y_train, y_test = split_data(data, labels, test_split=.20)
    norm_train_data = normalize(X_train)
    norm_test_data = normalize(X_test)
    columntype = ["nominal", "real", "real", "nominal", "nominal",
                  "nominal", "nominal", "real", "nominal", "nominal", "real", "nominal", "nominal", "real", "real"]
    KNN = KNNClassifier(labeltype='classification', columntype=columntype,
                        weight_type='inverse_distance', k=3, use_distance_weighting=True, regression=False)
    KNN.fit(norm_train_data, y_train, hoem=True)
    pred = KNN.predict(norm_test_data)
    error, accuracy = KNN.score(norm_test_data, y_test)
    print("ERROR:", error)
    print("ACCURACY:", accuracy)


def four():
    # house_price
    train_data, train_labels = setup_data("house_price_train.arff")
    test_data, test_labels = setup_data("house_price_test.arff")
    norm_train_data = normalize(train_data)
    norm_test_data = normalize(test_data)

    _, mse, labels = try_k(norm_train_data, norm_test_data, train_labels,
                           test_labels, reg=True, use_distance_weighting=True)
    plot_bar(mse, labels, xlabel="k", ylabel="MSE",
             title="House Price MSE by K value")

    # magic_telescope
    train_data, train_labels = setup_data("magic_telescope_train.arff")
    test_data, test_labels = setup_data("magic_telescope_test.arff")
    norm_train_data = normalize(train_data)
    norm_test_data = normalize(test_data)

    _, mse, labels = try_k(norm_train_data, norm_test_data, train_labels,
                           test_labels, reg=True, use_distance_weighting=True)
    plot_bar(mse, labels, xlabel="k", ylabel="MSE",
             title="Magic Telescope MSE by K value")


def three():
    train_data, train_labels = setup_data("house_price_train.arff")
    test_data, test_labels = setup_data("house_price_test.arff")
    # normalize
    norm_train_data = normalize(train_data)
    norm_test_data = normalize(test_data)

    KNN = KNNClassifier(labeltype='classification', k=3,
                        use_distance_weighting=False, regression=True)
    KNN.fit(norm_train_data, train_labels)
    pred = KNN.predict(norm_test_data)
    error, accuracy = KNN.score(norm_test_data, test_labels)
    print("NORMALIZED ERROR:", error)
    print("NORMALIZED ACCURACY:", accuracy)

    _, mse, labels = try_k(norm_train_data, norm_test_data,
                           train_labels, test_labels, reg=True)
    plot_bar(mse, labels, xlabel="k", ylabel="MSE",
             title="Housing Price KNN MSE by K value")


def two():
    train_data, train_labels = setup_data("magic_telescope_train.arff")
    test_data, test_labels = setup_data("magic_telescope_test.arff")
    # # without normalizing
    KNN = KNNClassifier(labeltype='classification',
                        weight_type='inverse_distance', k=3, use_distance_weighting=False)
    KNN.fit(train_data, train_labels)
    pred = KNN.predict(test_data)
    error, accuracy = KNN.score(test_data, test_labels)
    print("ERROR:", error)
    print("ACCURACY:", accuracy)

    # normalize
    norm_train_data = normalize(train_data)
    norm_test_data = normalize(test_data)
    KNN = KNNClassifier(labeltype='classification',
                        weight_type='inverse_distance', k=3, use_distance_weighting=False)
    KNN.fit(norm_train_data, train_labels)
    pred = KNN.predict(norm_test_data)
    error, accuracy = KNN.score(norm_test_data, test_labels)
    print("\nNORMALIZED ERROR:", error)
    print("NORMALIZED ACCURACY:", accuracy)

    accuracies, _, labels = try_k(
        norm_train_data, norm_test_data, train_labels, test_labels)
    plot_bar(accuracies, labels, xlabel="k", ylabel="Accuracy",
             title="Magic Telescope KNN Accuracy by K value")


def try_k(norm_train_data, norm_test_data, train_labels, test_labels, reg=False, use_distance_weighting=False):
    k = 1
    accuracies = []
    mse = []
    labels = []
    while k <= 15:
        KNN = KNNClassifier(labeltype='classification', weight_type='inverse_distance',
                            k=k, use_distance_weighting=use_distance_weighting, regression=reg)
        KNN.fit(norm_train_data, train_labels)
        pred = KNN.predict(norm_test_data)
        error, accuracy = KNN.score(norm_test_data, test_labels)
        print("\nNORMALIZED ERROR for k =", k, ":", error)
        print("NORMALIZED ACCURACY for k =", k, ":", accuracy)
        accuracies.append(accuracy)
        mse.append(error)
        labels.append(k)
        k += 2
    return accuracies, mse, labels


def one():
    train_data, train_labels = setup_data("diabetes.arff")
    test_data, test_labels = setup_data("diabetes_test.arff")
    KNN = KNNClassifier(labeltype='classification',
                        weight_type='inverse_distance', k=3, use_distance_weighting=True)
    KNN.fit(train_data, train_labels)
    pred = KNN.predict(test_data)
    error, accuracy = KNN.score(test_data, test_labels)
    print("ERROR:", error)
    print("ACCURACY:", accuracy)
    np.savetxt("diabetes-prediction.csv", pred, delimiter=',', fmt="%i")


def debug():
    train_data, train_labels = setup_data("seismic-bumps_train.arff")
    test_data, test_labels = setup_data("seismic-bumps_test.arff")
    KNN = KNNClassifier(labeltype='classification',
                        weight_type='inverse_distance', k=3, use_distance_weighting=False)
    KNN.fit(train_data, train_labels)
    pred = KNN.predict(test_data)
    error, accuracy = KNN.score(test_data, test_labels)
    print("(without weighting) ERROR:", error)
    print("(without weighting) ACCURACY:", accuracy)

    train_data, train_labels = setup_data("seismic-bumps_train.arff")
    test_data, test_labels = setup_data("seismic-bumps_test.arff")
    KNN = KNNClassifier(labeltype='classification',
                        weight_type='inverse_distance', k=3, use_distance_weighting=True)
    KNN.fit(train_data, train_labels)
    pred = KNN.predict(test_data)
    error, accuracy = KNN.score(test_data, test_labels)
    print("ERROR:", error)
    print("ACCURACY:", accuracy)
    # print("PREDICTION",pred)
    np.savetxt("seismic-bump-prediction.csv", pred, delimiter=',', fmt="%i")


def main():
    print("Starting tests")
    # debug()
    # one()
    # two()
    # three()
    # four()
    # five()
    six()
    return


if __name__ == '__main__':
    main()
