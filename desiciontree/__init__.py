from arff import Arff
import matplotlib.pyplot as plt
import numpy as np
from desiciontree import DTClassifier
from sklearn.model_selection import train_test_split
import sys
import math

def split_data(X, y, test_split=.25):
    # create test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
    return X_train, X_test, y_train, y_test

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
    debug()
    # part_one()
    # part_two()
    # part_three()
    # part_four()
    # part_five()
    # part_six()
    return



if __name__ == '__main__':
    main()