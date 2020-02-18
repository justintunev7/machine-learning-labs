import arff
import matplotlib.pyplot as plt
import numpy as np
from mlp import MLPClassifier
from sklearn.linear_model import Perceptron
import sys
import math

# def create_weight_matrix(hidden_)
def setup_data(file):
    # print("File =",file)
    mat = arff.Arff(file,label_count=1)
    data = mat.data[:,0:-1]
    labels = mat.data[:,-1].reshape(-1,1)
    return data, labels

def plot_bar(data, labels=None, xlabel="", ylabel="", title=""):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if labels != None: plt.xticks(range(len(labels)), labels)
    if type(data) == dict:
        width = 1 / (len(data) + 1)
        for i, key in enumerate(data):
            value = data[key]
            offset = width * (i - len(data) // 2)
            plt.bar(np.arange(len(value)) + offset, value, width=width, label=key) 
        plt.legend()
    else:
        plt.bar(range(len(data)), data) 
    plt.show()

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

def to_csv_file(filename, weights):
    temp_array = []
    for weight in weights:
        temp_array += weight.flat
    np.savetxt(filename, np.array(temp_array), delimiter=",")

def evaluation():
    print("\n\nEVALUATION")
    data, labels = setup_data("data_banknote_authentication.arff")
    MLPClass = MLPClassifier(lr=0.1,momentum=0.5,shuffle=False,deterministic=10, hidden_layer_widths=[((4*2))])
    MLPClass.fit(data,labels)
    weights = MLPClass.get_weights()
    to_csv_file("evaluation.csv", weights)
    accuracy = MLPClass.score(data, labels)
    print("ACCURACY",accuracy)

def part_one():
    debug()
    # evaluation()

def print_score_data(mse, accuracy):
    print("MSE",mse)
    print("ACCURACY",accuracy)

def debug():
    print("\nDEBUG")
    data, labels = setup_data("linsep2nonorigin.arff")
    # MLPClass = MLPClassifier(lr=0.1,shuffle=False, hidden_layer_widths=[4], deterministic=10, debug=True)
    MLPClass = MLPClassifier(lr=0.1,shuffle=False, hidden_layer_widths=[4], deterministic=10, debug=False)
    MLPClass.fit(data,labels, initial_weights=MLPClass.initialize_weights_to_zero(2))
    weights = MLPClass.get_weights()
    # print(weights)
    expected = [-1.0608880e-02,
                -2.1454953e-02, 
                -8.8177979e-05, 
                7.8275773e-04, 
                -3.9435364e-03, ]
    allowed_error = .000000001
    print(abs(weights[0][0][0] - expected[2]) < allowed_error)
    print(abs(weights[0][1][0] - expected[3]) < allowed_error)
    print(abs(weights[0][2][0] - expected[4]) < allowed_error)
    print(abs(weights[1][0][0] - expected[0]) < allowed_error)
    print(abs(weights[1][-1][0] - expected[1]) < allowed_error)
    accuracy = MLPClass.score(data, labels)
    print("ACCURACY",accuracy)

def part_two():
    print("\nPART TWO")
    data, labels = setup_data("iris.arff")
    MLPClass = MLPClassifier(lr=0.1,shuffle=True, hidden_layer_widths=[8], analytics=True)
    X_train, X_test, y_train, y_test = MLPClass.split_data(data, labels)
    MLPClass = MLPClass.fit(X_train,y_train)
    accuracy = MLPClass.score(X_test, y_test)
    print("ACCURACY",accuracy)
    analytics = MLPClass.error_analytics()
    length = range(len(analytics["Training error"]))
    plot_bar(analytics, labels=None, xlabel="Epochs", ylabel="Error/Accuracy", title="Error/Accuracy by Epoch")
    # t_error, v_error, v_accuracy = MLPClass.error_analytics()
    # plot_bar(t_error, range(len(t_error)), "Epochs", "Training Error", "Training Error by Epoch")
    # plot_bar(v_error, range(len(v_error)), "Epochs", "Validation Error", "Validation Error by Epoch")
    # plot_bar(v_accuracy, range(len(v_accuracy)), "Epochs", "Validation Accuracy", "Validation Accuracy by Epoch")

def plot_number_of_epochs(analytics, label):
    results = {
        "Number of Epochs": [len(a["Training error"]) for a in analytics],
        label: [a[label] for a in analytics]
    }
    plot_bar(results["Number of Epochs"], results[label], label, "Number of Epochs", "Number of Epochs to optimization by " + label)



def part_three():
    print("\nPART Three")
    data, labels = setup_data("vowel.arff")
    # remove train/test column
    data = data[:,1:]
    hidden_nodes = len(data[0]) * 2
    print(hidden_nodes)
    lr = 1.5
    MLPClass = MLPClassifier(lr=lr,shuffle=True, hidden_layer_widths=[hidden_nodes], analytics=True)
    X_train, X_test, y_train, y_test = MLPClass.split_data(data, labels)
    MLPClass = MLPClass.fit(X_train,y_train)
    # print("WEIGHTS", MLPClass.get_weights())
    accuracy = MLPClass.score(X_test, y_test)
    print("BASELINE ACCURACY",accuracy)
    errors = {"Training error": [], "Validation error": [], "Test error": []}
    analytics = []
    analytics.append(MLPClass.error_analytics())
    current_errors = MLPClass.error_results(X_test, y_test)
    errors["Training error"].append(current_errors[0])
    errors["Validation error"].append(current_errors[1])
    errors["Test error"].append(current_errors[2])
    analytics[-1]["Learning rate"] = lr
    bssf_num_epochs = math.inf
    current_num_epochs = len(analytics[-1]["Training error"])

    # we want to optimize on number of epochs, but don't ignore accuracy
    while current_num_epochs < bssf_num_epochs or (analytics[-1]["Validation accuracy"][-1] < .7):
        print(analytics[-1]["Validation accuracy"][-1])
        bssf_num_epochs = current_num_epochs
        lr = round((lr/1.5), 3) # decrease the learning rate each time
        del MLPClass
        MLPClass = MLPClassifier(lr=lr,shuffle=True, hidden_layer_widths=[hidden_nodes], analytics=True)
        X_train, X_test, y_train, y_test = MLPClass.split_data(data, labels)
        MLPClass = MLPClass.fit(X_train,y_train)

        current_errors = MLPClass.error_results(X_test, y_test)
        errors["Training error"].append(current_errors[0])
        errors["Validation error"].append(current_errors[1])
        errors["Test error"].append(current_errors[2])

        analytics.append(MLPClass.error_analytics())

        analytics[-1]["Learning rate"] = lr
        current_num_epochs = len(analytics[-1]["Training error"])
    learning_rates = [a["Learning rate"] for a in analytics]
    plot_number_of_epochs(analytics, "Learning rate")
    del analytics[-2]["Learning rate"]
    plot_bar(analytics[-2], xlabel="Epochs", ylabel="Error/Accuracy", title="Error/Accuracy by Epoch")
    plot_bar(errors, labels=learning_rates, xlabel="Learning rate", ylabel="MSE", title="Error by Learning Rate")
    accuracy = MLPClass.score(X_test, y_test)
    print("ACCURACY",accuracy)

def part_four():
    print("\nPART Four")
    data, labels = setup_data("vowel.arff")
    # remove train/test column
    data = data[:,1:]
    hidden_nodes = 1
    # what we found in part 3
    lr = .2
    MLPClass = MLPClassifier(lr=lr,shuffle=True, hidden_layer_widths=[hidden_nodes], analytics=True)
    X_train, X_test, y_train, y_test = MLPClass.split_data(data, labels)
    MLPClass = MLPClass.fit(X_train,y_train)
    # print("WEIGHTS", MLPClass.get_weights())
    accuracy = MLPClass.score(X_test, y_test)
    print("BASELINE ACCURACY",accuracy)
    errors = {"Training error": [], "Validation error": [], "Test error": []}
    analytics = []
    analytics.append(MLPClass.error_analytics())
    current_errors = MLPClass.error_results(X_test, y_test)
    errors["Training error"].append(current_errors[0])
    errors["Validation error"].append(current_errors[1])
    errors["Test error"].append(current_errors[2])
    analytics[-1]["Hidden nodes"] = hidden_nodes
    bssf_accuracy = 0
    current_accuracy = len(analytics[-1]["Validation accuracy"])

    # we want to optimize on number of epochs, but don't ignore accuracy
    while current_accuracy > bssf_accuracy or (analytics[-1]["Validation accuracy"][-1] < .7):
        print(analytics[-1]["Validation accuracy"][-1])
        bssf_accuracy = current_accuracy
        hidden_nodes *= 2 # double the learning rate each time
        del MLPClass
        MLPClass = MLPClassifier(lr=lr,shuffle=True, hidden_layer_widths=[hidden_nodes], analytics=True)
        X_train, X_test, y_train, y_test = MLPClass.split_data(data, labels)
        MLPClass = MLPClass.fit(X_train,y_train)

        current_errors = MLPClass.error_results(X_test, y_test)
        errors["Training error"].append(current_errors[0])
        errors["Validation error"].append(current_errors[1])
        errors["Test error"].append(current_errors[2])

        analytics.append(MLPClass.error_analytics())

        analytics[-1]["Hidden nodes"] = hidden_nodes
        current_accuracy = len(analytics[-1]["Validation accuracy"])
    hidden_node_labels = [a["Hidden nodes"] for a in analytics]
    # plot_different_learning_rates(analytics)
    # del analytics[-2]["Learning rate"]
    # plot_bar(analytics[-2], xlabel="Epochs", ylabel="Error/Accuracy", title="Error/Accuracy by Epoch")
    plot_bar(errors, labels=hidden_node_labels, xlabel="Hidden nodes", ylabel="MSE", title="Error by number of hidden nodes")
    accuracy = MLPClass.score(X_test, y_test)
    print("ACCURACY",accuracy)

def part_five():
    print("\nPART Five")
    data, labels = setup_data("vowel.arff")
    # remove train/test column
    data = data[:,1:]

    lr = .2
    hidden_nodes = 128
    momentum = .6

    MLPClass = MLPClassifier(lr=lr,shuffle=True, momentum=momentum, hidden_layer_widths=[hidden_nodes], analytics=True)
    X_train, X_test, y_train, y_test = MLPClass.split_data(data, labels)
    MLPClass = MLPClass.fit(X_train,y_train)
    # print("WEIGHTS", MLPClass.get_weights())
    accuracy = MLPClass.score(X_test, y_test)
    print("BASELINE ACCURACY",accuracy)
    errors = {"Training error": [], "Validation error": [], "Test error": []}
    analytics = []
    analytics.append(MLPClass.error_analytics())
    current_errors = MLPClass.error_results(X_test, y_test)
    errors["Training error"].append(current_errors[0])
    errors["Validation error"].append(current_errors[1])
    errors["Test error"].append(current_errors[2])
    analytics[-1]["Momentum"] = momentum
    bssf_num_epochs = math.inf
    current_num_epochs = len(analytics[-1]["Training error"])

    # we want to optimize on number of epochs, but don't ignore accuracy
    while current_num_epochs < bssf_num_epochs or (analytics[-1]["Validation accuracy"][-1] < .7):
        print(analytics[-1]["Validation accuracy"][-1])
        bssf_num_epochs = current_num_epochs
        momentum /= 2 # increase momentum each time
        del MLPClass
        MLPClass = MLPClassifier(lr=lr,shuffle=True, momentum=momentum, hidden_layer_widths=[hidden_nodes], analytics=True)
        X_train, X_test, y_train, y_test = MLPClass.split_data(data, labels)
        MLPClass = MLPClass.fit(X_train,y_train)

        current_errors = MLPClass.error_results(X_test, y_test)
        errors["Training error"].append(current_errors[0])
        errors["Validation error"].append(current_errors[1])
        errors["Test error"].append(current_errors[2])

        analytics.append(MLPClass.error_analytics())

        analytics[-1]["Momentum"] = momentum
        current_num_epochs = len(analytics[-1]["Training error"])
    momentum_labels = [a["Momentum"] for a in analytics]
    plot_number_of_epochs(analytics, "Momentum")
    # plot_different_learning_rates(analytics)
    # del analytics[-2]["Learning rate"]
    # plot_bar(analytics[-2], xlabel="Epochs", ylabel="Error/Accuracy", title="Error/Accuracy by Epoch")
    # plot_bar(errors, labels=momentum_labels, xlabel="Momentum", ylabel="MSE", title="MSE as a result of momentum multiplier")
    accuracy = MLPClass.score(X_test, y_test)
    print("ACCURACY",accuracy)

def main():
    # part_one()
    part_two()
    # part_three()
    # part_four()
    # part_five()
    return



if __name__ == '__main__':
    main()