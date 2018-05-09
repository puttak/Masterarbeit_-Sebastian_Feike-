########################################################################################################################
#This support.py file is divided into three sections
#1) Support functions for the predictive model
#2) Functions for loading the training and test data
#3) Functions for the visualization of the achieved performance
########################################################################################################################
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import itertools
from pylab import *

########################################################################################################################
# 1) Model functions
########################################################################################################################

def make_hparam_string(params):
    '''
    Conversion of all hyperparameters to strings for a unique identification in TensorBoard
    or cloud-based results accomplished.
    :param params: python dict which contains all hyperparameter of the ANNs
    :return:
    '''
    '''
    activation = ""
    if params["activation"] == tf.nn.relu:
        activation = "relu_act"
    elif params["activation"] == tf.nn.relu6:
        activation = "relu6_act"
    elif params["activation"] == tf.nn.selu:
        activation = "selu_act"
    elif params["activation"] == tf.nn.crelu:
        activation = "crelu_act"
    elif params["activation"] == tf.nn.elu:
        activation = "elu_act"
    elif params["activation"] == tf.nn.leaky_relu:
        activation = "leaky_act"
    else:
    '''
    activation = "NON_act"

    return 'steps%s, input%s, classes%s, rnn%s, layer%s, hidden%s, batch%s, epochs%s,' \
           'wdev%s, bdev%s, delta%s, opt%s, %s, learn%s, decay_r%s, ' \
           'drop%s' % (params["num_steps"], params["num_input"], params["num_classes"], params["rnn_cell"],
                       params["num_layer"], params["num_hidden"], params["batch_size"], params["num_epochs"],
                       params["weight_stddev"], params["bias_stddev"], params["delta"], params["optimizer"], activation,
                       params["learning_rate"], params["decay_rate"], params["dropout_rate"])

def iterator(train_temp, train_res, batch_size):
    '''
    In summary, this feature guarantees that each measurement of training data is always fed into the ANN
    in the same frequency and in a different order.
    :param train_temp: ndarray of train temperature sequences
    :param train_res: ndarray of train resistances
    :param batch_size:
    :return: randomly shuffled train batches
    '''
    # Shuffle training data
    shuffled_ix = np.random.permutation(np.arange(len(train_temp)))
    x_train = train_temp[shuffled_ix]
    y_train = train_res[shuffled_ix]
    # by small batch_sizes, --> last batch goes zero when its contains only 9 temperature measurement instead of 100
    # by static_rnn, all batch_sizes must have the same size
    # Since the number of training data is not divisible into batches of equal size,
    # the + 1 must be removed from num_batches. -> So there is always a small rest per epoch
    num_batches = int(len(x_train) / batch_size) #+ 1
    # split the train data into batches
    for i in range(num_batches):
        # Select train data
        min_ix = i * batch_size
        max_ix = np.min([len(x_train), ((i + 1) * batch_size)])
        x_train_batch = x_train[min_ix:max_ix]
        y_train_batch = y_train[min_ix:max_ix]
        yield (x_train_batch, y_train_batch)

########################################################################################################################
# 2) data load and preparation functions
########################################################################################################################

def one_hot(y_):
    '''
    Function to encode output labels from number indexes
    e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    :param y_: the labels
    :return: one hot encoded target classes
    '''
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

def load_train_test_data_for_RNN(prefix):
    '''
    The created Python function loads and reshape the data for a RNN
    :param prefix: The designation serves as a prefix.
    This ensures that the training and test data of the selected record is loaded
    :return: training and test sensor signals (x_) and the associated classes (y_) also the number of time steps
    (depending on the selected data set) and their inputs amount (per timestep --> standrad = 1)
    '''

    # check if the selected file exists
    if prefix == "2F5" or prefix == "2F22" or prefix == "2F48":
        print(str(prefix) + "Data loading")
    else:
        # if not print a message
        print("selected data doesn't exist")
        sys.exit()

    # load the train temperatures and resistances
    train_dataset = pd.read_csv('00_Data_and_prepare_files/' + str(prefix) + '_training_dataset(pump_1_3_4_5_6_7_9).csv')
    x_train = train_dataset.iloc[:, :101].values
    y_train = np.reshape(train_dataset.iloc[:, 101], (-1, 1))
    y_train = one_hot(y_train)
    # load the test temperatures and resistances
    test_dataset = pd.read_csv('00_Data_and_prepare_files/' + str(prefix) + '_test_dataset(pump_2_8).csv')
    x_test = test_dataset.iloc[:, :101].values
    y_test = np.reshape(test_dataset.iloc[:, 101], (-1, 1))
    y_test = one_hot(y_test)

    num_steps = len(x_train[0])  # 101 timesteps per series
    num_input = 1  # 1 input parameters per timestep
    # reshape the temperature sequence

    # divide the sequences into time steps and their inputs --> Transformation into suitable form for RNNs
    x_train = np.reshape(x_train, (len(x_train), num_steps, num_input))
    x_test = np.reshape(x_test, (len(x_test), num_steps, num_input))

    return x_train, y_train, x_test, y_test, num_steps, num_input

def load_train_test_data_for_FFNN(prefix):
    '''
    The function loads the training and test data according to the selected prefix
    (no data transformation is performed here in time steps as in the function of the same name for RNNs)
    :param prefix: The designation serves as a prefix.
    This ensures that the training and test data of the selected record is loaded
    :return: training and test sensor signals (x_) and the associated classes (y_)
    '''

    if prefix == "2F5" or prefix == "2F22" or prefix == "2F48":
        print(str(prefix) + "Data loading")
    else:
        print("no defined Data!")
        sys.exit()
    # load the train, eval and pred temperatures and resistances
    #train_dataset = pd.read_csv('Training_dataset(pump_1_to_7).csv')
    train_dataset = pd.read_csv('00_Data_and_prepare_files/' + str(prefix) + '_training_dataset(pump_1_3_4_5_6_7_9).csv')
    x_train = train_dataset.iloc[:, :101].values
    y_train = np.reshape(train_dataset.iloc[:, 101], (-1, 1))
    y_train = one_hot(y_train)

    test_dataset = pd.read_csv('00_Data_and_prepare_files/' + str(prefix) + '_test_dataset(pump_2_8).csv')
    x_test = test_dataset.iloc[:, :101].values
    y_test = np.reshape(test_dataset.iloc[:, 101], (-1, 1))
    y_test = one_hot(y_test)

    # determine the length of the measured sensor signals
    num_features = len(x_train[0])

    # transform the X and Y data into the desired shape
    x_train = np.reshape(x_train, (len(x_train), num_features))
    x_test = np.reshape(x_test, (len(x_test), num_features))

    return x_train, y_train, x_test, y_test

########################################################################################################################
# 3) visualise the model performance
########################################################################################################################

def acc_process_plot(y1, y2):
    '''
    plot the process ot the computed accuracy (accuracy = comparision between true value and predicted value)
    additional, can be shown over- or underfitting
    :param y1: test_accuracy_list depends on test data
    :param y2: train_accuracy_list depends on train data
    :return:
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(y1, '-', linewidth=2, label='val_acc', color=FAPS_Colours.green)
    ax.plot(y2, '--', linewidth=2, label='train_acc', color=FAPS_Colours.blue)
    ax.set_title('loss process over all epochs')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()
    plt.show()

def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix'):
    '''
    This function prints and plots the confusion matrix.
    :param cm: ndarray of the predicted and
    :param normalize: normalization can be applied by setting `normalize=True`.
    :param title:
    :return:
    '''
    classes = ["Bewegung_BSD_X (BBX)", "Fräsoperation_Hexagon_1 (FH1)", "Fräsoperation_Hexagon_2 (FH2)",
               "Bewegung_BSD_Y (BBY)", "Positionierung_Inaktiv (PI)"]
    classes_x = ["BBX", "FH1", "FH2", "BBY", "PI"]
    figure()
    #classes_x = ["MBX", "OHM1", "OHM2", "MBY", "PI"]
    #classes = ["Movement_BSD_X (MBX)", "Operation_Hexagon_Milling_1 (OHM1)", "Operation_Hexagon_Milling_2 (OHM2)",
    #           "Movement_BSD_Y (MBY)", "Positioning_Inactivity (PI)"]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes_x)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Tatsächliche Klassen')
    plt.xlabel('Vorhergesagte Klassen')

    plt.show()

def plot_time_measurements(y1, y2, batch_size):
    '''
    plot the computation-time of all prediction-time-measurements and their average
    :param y1: python list of all measurements
    :param y2: average prediction value
    :param batch_size: amount of predictions per measurement
    :return:
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # determine the max prediction time and the corresponding y value
    ymax = max(y1)
    xpos = y1.index(ymax)

    binwidth = 0.0005
    # create and show the plot
    #ax.hist(y1, bins=bins, histtype='bar', ec='black', label='Vorhersagezeiten', color=FAPS_Colours.green)
    ax.hist(y1, bins=np.arange(min(y1), max(y1) + binwidth, binwidth), histtype='bar', ec='black',
            label='Vorhersagezeiten', color=FAPS_Colours.green)
    #ax.hist(xpos, ymax, label='max. Vorhersagezeit(s): %s' %np.round(ymax, decimals=6),
             #color=FAPS_Colours.dark_green)
    ax.set_xlabel('Zeit in s \n'
                  '(Klassenbreite: %s)' %binwidth)
    binwidth_x_label = 0.002
    ax.set_xticks(np.arange(0, max(y1) + binwidth_x_label, binwidth_x_label))
    ax.set_ylabel('Häufigkeitsverteilung der gemessenen Zeiten')
    ax.yaxis.grid()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #ax.set_title('Zeitmessungen für die Losgröße %s' %batch_size)
    ax.set_title('durchsch. Vorhersagezeit: ' + str(round(y2*1000, 2)) + 'ms', fontsize=11)
    ax.legend(loc=1)
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), fancybox=True, shadow=False)
    plt.tight_layout()
    plt.show()


def compare_computed_time(results):
    '''
    Compares the time measurements of two predictive models. To do this, the main () function must iterate
    over two iterations and store the measured times in the result list.
    This function can therefore only be called at the end of the main ()
    :param results: python list which contians the results (time measurements of two iterations)
    :return:
    '''
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    #binwidth = 0.0005
    binwidth=0.5
    y = results[0][0]
    y = [x * 1000 for x in y]
    # create and show the plot
    #ax.hist(y1, bins=bins, histtype='bar', ec='black', label='Vorhersagezeiten', color=FAPS_Colours.green)
    ax1.hist(y, bins=np.arange(min(y), max(y) + binwidth, binwidth), histtype='bar', ec='black',
            label='Vorhersagezeiten', color=FAPS_Colours.green)
    #ax.plot(np.repeat(y2, 1418), '--', linewidth=3, label='Durchschnittliche Vorhersagezeit(s): %s'
    #                                                      %np.round(y2, decimals=6), color=FAPS_Colours.blue)
    #ax.hist(xpos, ymax, label='max. Vorhersagezeit(s): %s' %np.round(ymax, decimals=6),
             #color=FAPS_Colours.dark_green)
    ax1.set_xlabel('Zeit in ms \n'
                  '(Klassenbreite: ' + str(binwidth) + 'ms)', fontsize=10)
    binwidth_x_label = 2
    ax1.set_xticks(np.arange(0, max(y) + binwidth_x_label, binwidth_x_label))
    ax1.set_ylabel('Häufigkeitsverteilung der gemessenen Zeiten', fontsize=10)
    #ax1.yaxis.grid()
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_title('Vorhersagezeit eines FCNN mit: \n 3-Sichten und ca. 95% Acc. \n durchsch. Vorhersagezeit: '
                  + str(round(results[0][1]*1000, 2)) + 'ms', fontsize=11)
    #ax1.legend(loc=1)

    y1 = results[1][0]
    y1 = [x * 1000 for x in y1]
    # create and show the plot
    #ax.hist(y1, bins=bins, histtype='bar', ec='black', label='Vorhersagezeiten', color=FAPS_Colours.green)
    ax2.hist(y1, bins=np.arange(min(y1), max(y1) + binwidth, binwidth), histtype='bar', ec='black',
            label='Vorhersagezeiten', color=FAPS_Colours.green)
    #ax.plot(np.repeat(y2, 1418), '--', linewidth=3, label='Durchschnittliche Vorhersagezeit(s): %s'
    #                                                      %np.round(y2, decimals=6), color=FAPS_Colours.blue)
    #ax.hist(xpos, ymax, label='max. Vorhersagezeit(s): %s' %np.round(ymax, decimals=6),
             #color=FAPS_Colours.dark_green)
    ax2.set_xlabel('Zeit in ms \n'
                  '(Klassenbreite: ' + str(binwidth) + 'ms)', fontsize=10)
    binwidth_x_label = 2
    ax2.set_xticks(np.arange(0, max(y1) + binwidth_x_label, binwidth_x_label))
    #ax2.yaxis.grid()
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_title('Vorhersagezeit eines FCNN mit: \n 4-Sichten und ca. 98% Acc. \n durchsch. Vorhersagezeit: '
                  + str(round(results[1][1]*1000, 2)) + 'ms', fontsize=11)
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), fancybox=True, shadow=False)
    plt.tight_layout()
    plt.show()

class FAPS_Colours():
        green = (151 / 255, 193 / 255, 57 / 255)
        dark_green = (0 / 255, 102 / 255, 0 / 255)
        blue = (41 / 255, 97 / 255, 147 / 255)
        grey = (95 / 255, 95 / 255, 95 / 255)
        red = (255 / 255, 153 / 255, 51 / 255)
