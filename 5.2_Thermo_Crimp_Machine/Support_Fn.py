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
import pylab
import scipy.stats as stats
from Peaks_detection_Fn import detect_peaks # contains the function for the peak detection
import itertools

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
        activation = "NON_act"

    return 'steps%s, input%s, rnn%s, layer%s, hidden%s, batch%s, epochs%s,' \
           'wdev%s, bdev%s, delta%s, opt%s, %s, learn%s, mom%s, decay_r%s ' \
           'drop%s' % (params["num_steps"], params["num_input"], params["rnn_cell"], params["num_layer"],
                       params["num_hidden"], params["batch_size"], params["num_epochs"],
                       params["weight_stddev"], params["bias_stddev"], params["delta"], params["optimizer"], activation,
                       params["learning_rate"], params["momentum"], params["decay_rate"], params["dropout_rate"])



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


def prepare_temp_data_for_TBPTT(data, num_input):
    '''
    # if num_inputs > 1 than split the x_data into corresponding num_steps
    # Note: for a true Truncated Backpropagation i will sample the x_data in fixed-width sliding windows
    # of/with 50% overlapping
    # else: we will feed the full sequence length of each attempt (80 num_steps)
    :param num_input: input amount per timestep
    :param data: embody the train or eval or pred temp data
    :return: overlapping temperature sequence and the corresponding number of num_steps
    '''
    # how many sliding windows have to be created to have 50% overlapping windows (=num_steps)
    num_steps = int(((80 - num_input) // (num_input / 2)) + 1)
    # k2 defines the 50% overlapping
    k2 = int(num_input / 2)
    sliding_windows_range = int(num_steps * k2)
    # split the data into "num_sliding_windows" with 50% overlapping
    data = [data[:, j: j + num_input] for j in range(0, sliding_windows_range, k2)]
    # convert the data{list} into a {nddarray} and reshape them to the shape(length, num_steps, num_input)
    data = np.transpose(np.reshape(data, (num_steps, -1, num_input)), (1, 0, 2))

    return data, num_steps


def load_train_and_test_data(num_steps, num_input, TBTT=False, all_data=True):
    '''
    The created Python function has several parameters which define the data to be loaded and their shape
    :param num_steps: determine the numbers of timestamps
    :param num_input: determine the amount ob inputs per timestamps
    :param TBTT: if True reshape the temp. sequences according the num_inputs --> 50% overlapping
    :param all_data: if False, only the measurements of attempt 42 and 44 will be loaded
    :return: trainings and test temperature sequences and the according resistances
    '''
    if all_data == False:
        suffix = "(42_44)"
    else:
        suffix = ""

    # load the train, eval and pred temperatures and resistances
    train_dataset = pd.read_csv('00_Data_and_prepare_files/Training_Dataset%s.csv' % suffix)
    train_temp = train_dataset.iloc[:, :80].values
    train_res = np.reshape(train_dataset.iloc[:, 80], (-1, 1))

    test_dataset = pd.read_csv('00_Data_and_prepare_files/Test_Dataset%s.csv' % suffix)
    test_temp = test_dataset.iloc[:, :80].values
    test_res = np.reshape(test_dataset.iloc[:, 80], (-1, 1))

    # define the back propagation through time steps
    # reshape the temperature sequence
    if TBTT == True:
        train_temp, num_steps = prepare_temp_data_for_TBPTT(train_temp, num_input)
        test_temp, num_steps = prepare_temp_data_for_TBPTT(test_temp, num_input)
    else:
        train_temp = np.reshape(train_temp, (len(train_temp), num_steps, num_input))
        test_temp = np.reshape(test_temp, (len(test_temp), num_steps, num_input))

    return train_temp, train_res, test_temp, test_res

def peak_detection(temp_data, res_data):
    '''
    select the 8 characteristic peaks of each temp. sequence
    Should a temperature sequence have no 8 peaks, this and the associated resistance will not be considered further
    called by the following load_peak_train_and_test_data()
    :param temp_data:
    :param res_data:
    :return: the last 8 peaks of each measurement and the corresponding resistances
    '''

    peak_list = []
    failures_list = []
    for i in range(len(temp_data)):
        # single out one row of the "temp_data" panda Series
        temp = temp_data[i, :]
        # detect all peaks of the selected temperature sequence
        indexes = detect_peaks(temp)

        # check if a attempt get more or equal 8 temperature peaks
        # (!!! it is not the case by attempt 42,44 !!!, but by all measurements!!!)
        if len(indexes) >= 8:
            # select the last 8 peaks
            indexes = indexes[-8:]
            # select the corresponding temp-values
            peaks = temp[indexes]
            # save them into the peak list
            peak_list.append(peaks)

        # get the attempt less then 8 peaks, it will be not considered further
        else:
            # save the indexes which get less then 8 peaks
            failures_list.append(i)

    # transform the detected peaks into an array where all peaks of one resistace (gets 2 measurements --> 16 peaks)
    # are in one row
    temp_data = np.reshape(peak_list, (-1, 8))
    # delete all resistances which temperature sequences get less the 8 peaks
    # and reshape them into a array with one column
    res_data = np.reshape(np.delete(res_data, failures_list, None), (-1, 1))
    # delete every second row (delete the duplicates)
    # res_data = np.delete(res_data, list(range(0, res_data.shape[0], 2)), axis=0)

    return temp_data, res_data

def load_peak_train_and_test_data(all_data):
    '''
    With this function, the last 8 peaks of the training and test temperature sequences are determined.
    To remember: the course of the 8 peaks is characteristic of a successful or a faulty crimp connection.
    (Provided the measurement data are correct)
    :param all_data: If all_data is False, then only the peaks from measurement trials 42 and 44 are determined.
    Otherwise from all measurements.
    :return: trainings and test temperature sequences and the according resistances
    '''

    if all_data == False:
        suffix = "(42_44)"
    else:
        suffix = ""

    # load only the last eight temperature peaks and resistances
    # train data
    train_dataset = pd.read_csv('00_Data_and_prepare_files/Training_Dataset%s.csv' % suffix)
    train_temp = train_dataset.iloc[:, :80].values
    train_res = np.reshape(train_dataset.iloc[:, 80], (-1, 1))
    train_temp, train_res = peak_detection(train_temp, train_res)
    # divided into good ( < 0.01) and faulty  (> 0.01) resistances
    train_res = np.reshape([0 if x < 0.01 else 1 for x in train_res], (-1,1))
    # apply one hot encoding
    train_res = one_hot(train_res)

    test_dataset = pd.read_csv('00_Data_and_prepare_files/Test_Dataset%s.csv' % suffix)
    test_temp = test_dataset.iloc[:, :80].values
    test_res = np.reshape(test_dataset.iloc[:, 80], (-1, 1))
    test_temp, test_res = peak_detection(test_temp, test_res)
    # divided into good ( < 0.01) and faulty  (> 0.01) resistances
    test_res = np.reshape([0 if x < 0.01 else 1 for x in test_res], (-1, 1))
    # apply one hot encoding
    test_res = one_hot(test_res)

    return train_temp, train_res, test_temp, test_res

def load_X(X_signals_paths):
    '''
    ++++++++++++++++++++++++++++ only for the HAR Dataset +++++++++++++++++++++++++++++
    prepare the sensor signals (accelerometer and gyroscope)  of the HAR dataset
    :param X_signals_paths: activity sequences
    :return:
    '''

    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))

def load_y(y_path):
    '''
    ++++++++++++++++++++++++++++ only for the HAR Dataset +++++++++++++++++++++++++++++
    load and prepare the labels of the HAR-Dataset
    :param y_path: # Load "y" (the neural network's training and testing outputs)
    :return:
    '''
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1

########################################################################################################################
# 3) visualise the model performance
########################################################################################################################

class FAPS_Colours():
    green = (151 / 255, 193 / 255, 57 / 255)
    light_green = (205 / 255, 226 / 255, 158 / 255)
    dark_green = (0 / 255, 102 / 255, 0 / 255)
    blue = (41 / 255, 97 / 255, 147 / 255)
    light_blue = (180 / 255, 208 / 255, 234 / 255)
    dark_blue = (30 / 255, 71 / 255, 108 / 255)
    grey4 = (95 / 255, 95 / 255, 95 / 255)
    light_grey = (178 / 255, 178 / 255, 178 / 255)
    red = (153 / 255, 0 / 255, 51 / 255)
    orange = (255 / 255, 153 / 255, 51 / 255)

def loss_process_plot(y1, y2):
    '''
    plot the process ot the calculated loss (loss = distance between true value - predicted value)
    additional, can be shown over- or underfitting
    :param y1: test_loss_list depends on test data
    :param y2: train_loss_list depends on train data
    :return:
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(y1, '-', linewidth=2, label='test_loss',  color=FAPS_Colours.green)
    ax.plot(y2, '--', linewidth=2, label='train_loss', color=FAPS_Colours.blue)
    ax.set_title('loss process over all epochs')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()
    plt.show()

def rmse_process_plot(y1, y2):
    '''
    plot the process ot the calculated loss (loss = distance between true value - predicted value)
    additional, can be shown over- or underfitting
    :param y1: test_rmse_list depends on test data
    :param y2: train_remse_list depends on train data
    :return:
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(y1, '-', linewidth=2, label='test_rmse',  color=FAPS_Colours.green)
    ax.plot(y2, '--', linewidth=2, label='train_rmse', color=FAPS_Colours.blue)
    ax.set_title('loss process over all epochs')
    ax.set_xlabel('epoch')
    ax.set_ylabel('rmse')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()
    plt.show()

def correlation_test(pred_y, y):
    '''
    create a quantile-quantile plot to control the trainprocess
    :param pred_y: predicted resistance
    :param y: actual resistances
    :return:
    '''

    # reshape and convert tensors into numpy ndarray
    y = np.reshape(y, (-1,))
    pred_y = np.reshape(pred_y, (-1,))

    corrcoef = np.correlate(y, pred_y)
    # create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(y, pred_y, 'o', color=FAPS_Colours.green)
    #ax.set_title('Zwischen den vorhergesagten und den tatsächlichen Widerständeb '
    #             '\n herrscht eine Korrelation von: %s ' + str(corrcoef/100) + '%', fontsize = 11)
    ax.set_title('Mit LSTM-Zellen wird eine Korrelation'
                 '\nvon: ' + str(corrcoef/100) + '% erreicht', fontsize = 11)
    ax.set_xlabel('Gemessene Widerstände', fontsize = 10)
    ax.set_ylabel('Vorhergesagte Widerstände', fontsize = 10)
    ax.grid(linestyle=':')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.show()

def q_q_plot(y_pred):
    '''
    plot eine Quantitil- Quantitl Abbidlung der Vorhergesagten Widerstände
    :param y_pred: predicted resistances
    :return:
    '''
    y_pred = np.reshape(y_pred, (-1,))
    stats.probplot(y_pred, dist="norm", plot=pylab)
    pylab.show()


def compare_Correlation(results):

    # reshape and convert tensors into numpy ndarray
    y1 = np.reshape(results[0][0], (-1,))
    pred_y1 = np.reshape(results[0][1], (-1,))
    corrcoef1 = np.correlate(y1, pred_y1)

    y2 = np.reshape(results[1][0], (-1,))
    pred_y2 = np.reshape(results[1][1], (-1,))
    corrcoef2 = np.correlate(y2, pred_y2)

    # create the plot
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    ax1.plot(y1, pred_y1, 'o', color=FAPS_Colours.green)
    ax1.set_title('Mit LSTM-Zellen wird eine Korrelation'
                 '\nvon: ' + str(corrcoef1/100) + '% erreicht', fontsize = 11)
    ax1.set_xlabel('Gemessene Widerstände [Ohm]', fontsize = 10)
    ax1.set_ylabel('Vorhergesagte Widerstände [Ohm]', fontsize = 10)
    ax1.grid(linestyle=':')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    ax2.plot(y2, pred_y2, 'o', color=FAPS_Colours.green)
    ax2.set_title('Mit GRU-Zellen wird eine Korrelation'
                 '\n von: ' + str(corrcoef2/100) + '% erreicht', fontsize = 11)
    ax2.set_xlabel('Gemessene Widerstände [Ohm]', fontsize = 10)
    #ax2.set_ylabel('Vorhergesagte Widerstände', fontsize = 10)
    ax2.grid(linestyle=':')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    #ax2.legend()
    plt.tight_layout()
    plt.show()


def compare_RNN_Cells(results):
    '''
    This function visualizes the course of the performance indicators MSE and RMSE from the
    different training iterations
    :param results: python listes witch contains the process of the train and test losses and RMSE's
    :return: two horizontal subplots
    '''
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(results[0][0], '-', color=FAPS_Colours.light_green)
    ax1.plot(results[0][1], '--', color=FAPS_Colours.green)
    ax1.plot(results[1][0], '-', color=FAPS_Colours.light_blue)
    ax1.plot(results[1][1], '--', color=FAPS_Colours.blue)
    ax1.plot(results[2][0], '-', color=FAPS_Colours.light_grey)
    ax1.plot(results[2][1], '--', color=FAPS_Colours.grey4)
    ax1.set_title('Performanz-Bewertung bzgl. \n der Entwicklung des Verlustes', fontsize=12)
    ax1.set_xlabel('Epochen', fontsize=10)
    ax1.set_ylabel('Verlust', fontsize=10)
    ax1.set_ylim([0, 0.1])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    ax2.plot(results[0][2], '-', label='LSTM train', color=FAPS_Colours.light_green)
    ax2.plot(results[0][3], '--', label='LSTM test', color=FAPS_Colours.green)
    ax2.plot(results[1][2], '-', label='GRU train', color=FAPS_Colours.light_blue)
    ax2.plot(results[1][3], '--', label='GRU test', color=FAPS_Colours.blue)
    ax2.plot(results[2][2], '-', label='LN_LSTM train', color=FAPS_Colours.light_grey)
    ax2.plot(results[2][3], '--', label='LN_LSTM test', color=FAPS_Colours.grey4)
    ax2.set_title('Performanz-Bewertung \n bzgl. der Entwicklung des RMSE', fontsize=12)
    ax2.set_xlabel('Epochen')
    ax2.set_ylabel('RMSE')
    ax2.set_ylim([0, 0.1])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.legend(loc='upper center', bbox_to_anchor=(-0.2, -0.2),
               fancybox=True, shadow=True, ncol=3, fontsize=10)
    plt.show()

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

    ax.plot(y1, '-', linewidth=2, label='test_acc', color=FAPS_Colours.green)
    ax.plot(y2, '--', linewidth=2, label='train_acc', color=FAPS_Colours.blue)
    ax.set_title('loss process over all epochs')
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()
    plt.show()


def plot_confusion_matrix(cm, labels, normalize=False, title='Konfusionsmatrix (oder Wahrheitsmatrix)'):
    """
    This function prints and plots the confusion matrix.
    :param cm: ndarray of the predicted and
    :param normalize: normalization can be applied by setting `normalize=True`.
    :param title:
    :return:
    """
    label_de = ["Gehen (GE)", "Hoch_Gehen (H_GE)", "Runter_Gehen (R_GE)", "Sitzen (SI)",
              "Stehen (ST)", "Liegen (LI)"]
    classes = label_de
    label_de = ["WALKING (WA)", "WALKING_UPSTAIRS (W_U)", "WALKING_DOWNSTAIRS (W_D)", "SITTING (SI)",
              "STANDING (ST)", "LAYING (LA)"]
    classes_x = ["GE","H_GE", "R_GE", "SI", "ST", "LI" ]
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
    plt.xticks(tick_marks, classes_x)#rotation=45
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

class FAPS_Colours():
    green = (151 / 255, 193 / 255, 57 / 255)
    light_green = (205 / 255, 226 / 255, 158 / 255)
    dark_green = (0 / 255, 102 / 255, 0 / 255)
    blue = (41 / 255, 97 / 255, 147 / 255)
    light_blue = (180 / 255, 208 / 255, 234 / 255)
    dark_blue = (30 / 255, 71 / 255, 108 / 255)
    grey4 = (95 / 255, 95 / 255, 95 / 255)
    light_grey = (178 / 255, 178 / 255, 178 / 255)
    red = (153 / 255, 0 / 255, 51 / 255)
    orange = (255 / 255, 153 / 255, 51 / 255)