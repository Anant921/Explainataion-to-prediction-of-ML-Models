#!/usr/bin/env python3

import getopt
import os
import sys

import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from explaining.explain import encoding_model, explaining_procedures
from keras.datasets import mnist
from neural_functions.neural_net_functions import build_model, dataset_preprocessing

from tensorflow import keras


def parse_options():
    """
        Parses command-line options:
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   'm:n:d:e:s:h',
                                   ['mnist'
                                    'hidden_node=',
                                    'dataframe=',
                                    'explanation=',
                                    'solver=',
                                    'help'])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err).capitalize())
        usage()
        sys.exit(1)

    number_of_nodes_in_hidden_layer = 20
    dataframe_name = None
    explanation_type = "minimal"
    solver = "cplex"

    for opt, arg in opts:
        if opt in ('-n', '--hidden_node'):
            number_of_nodes_in_hidden_layer = int(arg)
        elif opt in ('-d', '--dataframe'):
            dataframe_name = str(arg)
        elif opt in ('-e', '--explanation'):
            explanation_type = str(arg)
        elif opt in ('-s', '--solver'):
            solver = str(arg)
        elif opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        else:
            assert False, 'Unhandled option: {0} {1}'.format(opt, arg)

    return number_of_nodes_in_hidden_layer, dataframe_name, explanation_type, solver


def usage():
    """
        Prints usage message.
    """

    print('Usage:', os.path.basename(sys.argv[0]), '[options] lp-file')
    print('Options:')
    print('        -n, --hidden_node=<int>  Number of hidden node for the NN')
    print('                                 Available values: [10, 15, 20] (default: 20)')
    print('        -d, --dataframe=<string> Name of dataframe to explain')
    print('        -e, --explanation=<string>      Approach to get explanation')
    print('                                 Available values: [minimal, smallest] (default: minimal)')
    print('        -s, --solver=<string>        Type of solver to use')
    print('                                 Available values: [cplex, smt] (default: cplex)')
    print('        -h, --help')


def get_datas(datas):
    data = np.hstack(datas[0])
    dataframe = pd.DataFrame(data, index=["p" + str(i) for i in range(len(data))]).transpose()
    dataframe['target'] = datas[1]

    return dataframe


def create_unique_df(datas):
    dataframe = pd.DataFrame()

    for couple in datas:
        row = get_datas(couple)
        dataframe = dataframe.append(row)

    dataframe.reset_index()

    return dataframe


def get_MNIST_datas():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    X = np.append(train_X, test_X, axis=0)
    y = np.append(train_y, test_y, axis=0)

    to_add = []
    for i, value in enumerate(y):
        if value == 3 or value == 8:
            to_add.append((X[i], value))

    dataframe = create_unique_df(to_add)

    path_sep = os.sep

    dataframe.to_csv("." + path_sep + "datasets_files" + path_sep + "MNIST_datas.csv", index=False)

    from matplotlib import pyplot as plt

    num_row = 2
    num_col = 5

    num = 10
    images = train_X[:num]
    labels = train_y[:num]
    # plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
    for i in range(num):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(images[i], cmap='gray')
        ax.set_title('Label: {}'.format(labels[i]))
    plt.tight_layout()
    plt.show()


def import_dataframe(dataframe_name):

    sep = os.sep

    if dataframe_name == "mnist":
        dataframe = pd.read_csv("." + sep + "datasets_files" + sep + "MNIST_datas.csv").iloc[:100, :]
    else:
        dataframe = pd.read_csv("." + sep + "datasets_files" + sep + dataframe_name, index_col=[0], sep=',', na_values=[''])

    return dataframe


def print_vars(variables):
    for v in variables:
        print(v.df_name + ": " + str(v.varValue))


def print_estimates(values, measure):
    print("Min ", measure, " size: ", np.min(values))
    print("Avg ", measure, " size: ", np.average(values))
    print("Max ", measure, " size: ", np.max(values))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    hidden_nodes, df_name, expl_type, solver_name = parse_options()

    df = import_dataframe(df_name)
    df = dataset_preprocessing(df, df_name)

    explanations = []
    computational_times = []

    path_separator = os.sep
    model_path = "." + path_separator + "models" + path_separator + df_name + "_hidden_" + str(hidden_nodes)

    if os.path.isdir(model_path):
        model = keras.models.load_model(model_path)
    else:
        model = build_model(df, hidden_nodes)
        model.save(model_path)

    encoded_model, input_vars = encoding_model(model, df_name, solver_name)

    for pattern_number in range(df.shape[0]):
        print("Pattern number: ", pattern_number)
        x_values = df.drop(columns="target").iloc[pattern_number, :]
        input_values = x_values.array.astype('float32').reshape(1, -1)
        print("Input values: ", input_values)

        correct_output = df['target']
        output_values = model.predict(input_values)[0]
        #
        output_values = [0 if i != output_values.tolist().index(max(output_values)) else 1 for i in range(len(output_values))]
        output_vars = ["X_{0}_layer_2_type_C".format(i) for i in range(len(output_values))]

        print("Prediction: ", output_values, " id predicted_class: ", output_values.index(1))

        len_expl, time = explaining_procedures(encoded_model, expl_type, solver_name, pattern_number, df_name,
                                               (input_vars, input_values[0]), (output_vars, output_values))
        explanations.append(len_expl)
        computational_times.append(time)
        print("Elapsed Time: ", np.sum(computational_times))

    print_estimates(explanations, "Explanation")
    print_estimates(computational_times, "Computational time")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
