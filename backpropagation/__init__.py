import arff
import matplotlib.pyplot as plt
import numpy as np
from mlp import MLPClassifier
from sklearn.neural_network import MLPClassifier as SKMLPClassifier
from sklearn.model_selection import train_test_split
import sys
import math

def setup_data(file):
    print("File =",file)
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
    evaluation()

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
        momentum /= 2 # decrease momentum each time
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
    accuracy = MLPClass.score(X_test, y_test)
    print("ACCURACY",accuracy)

def split_data(X, y, test_split=.25):
        # create test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
        return X_train, X_test, y_train, y_test

def part_six_run_models(data, labels):
    mlp1 = SKMLPClassifier(hidden_layer_sizes=(10,10), 
                           activation='relu',
                           learning_rate_init=.01,
                           momentum=.9,
                           max_iter=1000,
                           nesterovs_momentum=False,
                           early_stopping=False,
                           alpha=.001)
    mlp2 = SKMLPClassifier(hidden_layer_sizes=(20,20,20), 
                           activation='tanh',
                           learning_rate_init=.1,
                           momentum=.5,
                           max_iter=1000,
                           nesterovs_momentum=True,
                           early_stopping=True,
                           alpha=.01)
    my_mlp = MLPClassifier(hidden_layer_widths=[20], lr=.2, deterministic=20, momentum=.5)
    X_train, X_test, y_train, y_test = my_mlp.split_data(data, labels)
    # print(y_train[:,-1])
    mlp1.fit(X_train,y_train[:,-1])
    mlp2.fit(X_train,y_train[:,-1])
    my_mlp.fit(X_train, y_train)
    print("\nMLP1 SCORE:", mlp1.score(X_test, y_test))
    print("MLP2 SCORE:", mlp2.score(X_test, y_test))
    print("MY_MLP SCORE:", my_mlp.score(X_test, y_test, error=False))

def grid_search(X_train, X_test, y_train, y_test):
    results = []
    for activation in ['relu', 'tanh', 'logistic']:
        for lr in [.001, .01, .1]:
            for hidden_layers in [(10,10), (20, 20), (30)]:
                for momentum in [.1, .5, .9]:
                    mlp = SKMLPClassifier(hidden_layer_sizes=hidden_layers,
                                          activation=activation,
                                          momentum=momentum,
                                          early_stopping=False,
                                          max_iter=1000,
                                          learning_rate_init=lr)
                    print("\nHidden layer sizes:", hidden_layers,
                          "Activation fun:", activation,
                          "Momentum:", momentum,
                          "Learning rate", lr)
                    mlp.fit(X_train, y_train[:,-1])
                    score = mlp.score(X_test, y_test)
                    print("Score:", score)
                    results.append((activation, lr, hidden_layers, momentum, score))
    return results

def part_six():
    print("\nPART SIX")
    data, labels = setup_data("vowel.arff")
    # remove train/test column
    data = data[:,1:]
    part_six_run_models(data, labels)

    data, labels = setup_data("iris.arff")
    part_six_run_models(data, labels)

    data, labels = setup_data("diabetes.arff")
    X_train, X_test, y_train, y_test = split_data(data, labels)
    results = grid_search(X_train, X_test, y_train, y_test)
    print(max(results, key=lambda r: r[-1] ))

def main():
    print("Starting tests")
    part_one()
    part_two()
    part_three()
    part_four()
    part_five()
    part_six()
    return



if __name__ == '__main__':
    main()