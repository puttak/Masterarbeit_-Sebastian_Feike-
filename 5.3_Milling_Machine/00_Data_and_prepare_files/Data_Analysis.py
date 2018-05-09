import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
import csv
import time


########################################################################################################################
# visualize the acceleration sensor measurement platters
########################################################################################################################

def pcolor():
    '''
    - plot the vibration platters of one pump
    - use pcolor for it
    :return:
    '''

    # load the "labels_hex_plus_complete_specgram.pkl" file
    file = pd.read_pickle('revised_labels_hex_plus_complete_specgram.pkl')
    # convert it to a Dataframe
    df = pd.DataFrame(file)

    # load the vibration measurements for pump 1
    measurement_data = df[0]
    # KS = Körperschall
    KS = measurement_data[1][0]

    # create a pcolor plot
    fig, ax = plt.subplots()
    #ax.pcolor(KS, norm=colors.LogNorm(vmin=KS.min(), vmax=KS.max()), cmap='viridis')
    #ax.pcolor(KS, norm=colors.PowerNorm(gamma=1. / 20.), cmap='viridis')
    ax.pcolor(KS, norm=colors.SymLogNorm(linthresh=0.0001, linscale=3,
                                                  vmin=-0.5, vmax=0.5), cmap='viridis')
    # define the axex an titel
    ax.set_xlabel("Messungen über Pumpen-Fertigungszyklus \n (Messungen alle 62,5 ms)", fontsize = 10)
    ax.set_ylabel("Messwerte des Beschleunigungssensors", fontsize = 10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #ax.set_title("Gemessener Körperschall bei Pumpe 1", fontsize = 12)
    plt.tight_layout()
    plt.show()
pcolor()

########################################################################################################################
# visualize the measurements and the corresponding labels
########################################################################################################################

def plot_pump_labels_3_4_in_subplots():
    # load the "labels_hex_plus_complete_specgram.pkl" file
    file = pd.read_pickle('revised_labels_hex_plus_complete_specgram.pkl')
    # convert it to a Dataframe
    df = pd.DataFrame(file)
    # select column "Movement_BSD_X"
    measurement_data = df[0]
    # select column "Operation_Hexagon_Milling_1
    label_data = df[1]

    label_list = []
    # save for every pump the energy_sequence and the corresponding labels into the energy- and label_list
    plt.plot()
    plt.show()

    time_shift = 0
    x_steps = np.reshape(np.arange(0, 709, 1), (-1,1))
    for pump in range(1, len(df)):
        measurement = measurement_data[pump]
        time = np.reshape(measurement[2], (-1, 1))

        # select the energy sequence of one pump
        label = label_data[pump]
        # convert the shape

        labels = np.reshape(label, (-1, 1))
        index_3 = np.where(labels==3)
        index_4 = np.where(labels==4)
        index_3_4 = np.sort(np.concatenate((index_3[0], index_4[0])))

        label_list.append(labels[index_3_4])
    y_scale = [3, 4]
    faps_green = (151 / 255, 193 / 255, 57 / 255)

    f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(nrows=9, ncols=1, sharex=True, sharey=True)
    # subplot for pump1
    ax1.plot(label_list[0], 'o', label="Prozesse \n (Pumpe 1)", color=faps_green)
    ax1.set_xticks([])
    ax1.yaxis.grid()
    ax1.set_yticks(y_scale)
    ax1.set_ylabel("Label")
    #ax1.spines['right'].set_visible(False)
    #ax1.spines['top'].set_visible(False)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # subplot for pump2
    ax2.plot(label_list[1], 'o', label="Prozesse \n (Pumpe 2)", color=faps_green)
    ax2.set_xticks([])
    ax2.yaxis.grid()
    ax2.set_yticks(y_scale)
    ax2.set_ylabel("Label")
    #ax2.spines['right'].set_visible(False)
    #ax2.spines['top'].set_visible(False)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # subplot for pump3
    ax3.plot(label_list[2], 'o', label="Prozesse \n (Pumpe 3)", color=faps_green)
    ax3.set_xticks([])
    ax3.yaxis.grid()
    ax3.set_yticks(y_scale)
    ax3.set_ylabel("Label")
    #ax3.spines['right'].set_visible(False)
    #ax3.spines['top'].set_visible(False)
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # subplot for pump4
    ax4.plot(label_list[3], 'o', label="Prozesse \n (Pumpe 4)", color=faps_green)
    ax4.set_xticks([])
    ax4.yaxis.grid()
    ax4.set_yticks(y_scale)
    ax4.set_ylabel("Label")
    #ax4.spines['right'].set_visible(False)
    #ax4.spines['top'].set_visible(False)
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # subplot for pump5
    ax5.plot(label_list[4], 'o', label="Prozesse \n (Pumpe 5)", color=faps_green)
    ax5.set_xticks([])
    ax5.yaxis.grid()
    ax5.set_yticks(y_scale)
    ax5.set_ylabel("Label")
    #ax5.spines['right'].set_visible(False)
    #ax5.spines['top'].set_visible(False)
    ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # subplot for pump6
    ax6.plot(label_list[5], 'o', label="Prozesse \n (Pumpe 6)", color=faps_green)
    ax6.set_xticks([])
    ax6.yaxis.grid()
    ax6.set_yticks(y_scale)
    ax6.set_ylabel("Label")
    #ax6.spines['right'].set_visible(False)
    #ax6.spines['top'].set_visible(False)
    ax6.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # subplot for pump7
    ax7.plot(label_list[6], 'o', label="Prozesse \n (Pumpe 7)", color=faps_green)
    ax7.set_xticks([])
    ax7.yaxis.grid()
    ax7.set_yticks(y_scale)
    ax7.set_ylabel("Label")
    #ax7.spines['right'].set_visible(False)
    #ax7.spines['top'].set_visible(False)
    ax7.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # subplot for pump8
    ax8.plot(label_list[7], 'o', label="Prozesse \n (Pumpe 8)", color=faps_green)
    ax8.set_xticks([])
    ax8.yaxis.grid()
    ax8.set_yticks(y_scale)
    ax8.set_ylabel("Label")
    #ax8.spines['right'].set_visible(False)
    #ax8.spines['top'].set_visible(False)
    ax8.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # subplot for pump9
    ax9.plot(label_list[8], 'o', label="Prozesse \n (Pumpe 9)", color=faps_green)
    ax9.set_xticks([])
    ax9.yaxis.grid()
    ax9.set_yticks(y_scale)
    ax9.set_ylabel("Label")
    #ax9.spines['right'].set_visible(False)
    #ax9.spines['top'].set_visible(False)
    ax9.set_xlabel("Prozessfolge der spanenden Fertigung einer Pumpe")
    ax9.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    #plt.subplots_adjust(hspace=0.1)
    plt.show()

#plot_pump_labels_3_4_in_subplots()

def plot_pump_labels_in_subplots():
    # load the "labels_hex_plus_complete_specgram.pkl" file
    file = pd.read_pickle('revised_labels_hex_plus_complete_specgram.pkl')
    # convert it to a Dataframe
    df = pd.DataFrame(file)
    # select column "Movement_BSD_X"
    measurement_data = df[0]
    # select column "Operation_Hexagon_Milling_1
    label_data = df[1]
    x1 = np.reshape([1,2,3], (-1,1))
    x2 = 2
    z = x1+x2
    label_list = []
    # save for every pump the energy_sequence and the corresponding labels into the energy- and label_list
    plt.plot()
    plt.show()

    time_shift = 0
    x_steps = np.reshape(np.arange(0, 709, 1), (-1,1))
    for pump in range(1, len(df)):
        measurement = measurement_data[pump]
        time = np.reshape(measurement[2], (-1, 1))

        # select the energy sequence of one pump
        label = label_data[pump]
        # convert the shape
        labels = np.reshape(label, (-1, 1))
        label_list.append(labels)
    y_scale = [0, 1, 2, 3, 4]
    faps_green = (151 / 255, 193 / 255, 57 / 255)

    f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)


    # subplot for pump4
    ax1.plot(label_list[3], 'o', color=faps_green)
    ax1.set_xticks([])
    ax1.yaxis.grid()
    ax1.set_yticks(y_scale)
    ax1.set_ylabel("Zustand")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_title("Zustände während der Fertigung von Pumpe 4", fontsize=11)
    #ax1.legend(loc='center upper', bbox_to_anchor=(1, 0.5))
    '''
    # subplot for pump5
    ax2.plot(label_list[4], 'o', color=faps_green)
    ax2.set_xticks([])
    ax2.yaxis.grid()
    ax2.set_yticks(y_scale)
    ax2.set_ylabel("Zustand")
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_title("Zustände während der Fertigung von Pumpe 5", fontsize=11)
    #ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    '''
    # subplot for pump7
    ax2.plot(label_list[6], 'o', color=faps_green)
    ax2.set_xticks([])
    ax2.yaxis.grid()
    ax2.set_yticks(y_scale)
    ax2.set_ylabel("Zustand")
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_title("Zustände während der Fertigung von Pumpe 7", fontsize=11)
    #ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.set_xlabel("Zustandsfolge der spanenden Fertigung einer Pumpe")
    '''
    # subplot for pump8
    ax4.plot(label_list[7], 'o', color=faps_green)
    ax4.set_xticks([])
    ax4.yaxis.grid()
    ax4.set_yticks(y_scale)
    ax4.set_ylabel("Zustand")
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.set_xlabel("Zustandsfolge der spanenden Fertigung einer Pumpe")
    ax4.set_title("Zustände während der Fertigung von Pumpe 8", fontsize=11)
    #ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    '''

    plt.subplots_adjust(hspace=0.1)
    plt.show()


#plot_pump_labels_in_subplots()




def plot_milling_operation_of_all_pumps():
    # load the "labels_hex_plus_complete_specgram.pkl" file
    file = pd.read_pickle('labels_hex_plus_complete_specgram.pkl')
    # convert it to a Dataframe
    df = pd.DataFrame(file)
    # select column "Movement_BSD_X"
    energy_data = df[0]
    # select column "Operation_Hexagon_Milling_1
    operations = df[1]


    # save for every pump the energy_sequence and the corresponding labels into the energy- and label_list
    for pump in range(1, len(df)):
        # select the energy sequence of one pump
        energy = energy_data[pump]
        # convert the shape
        energy = np.transpose(np.reshape(energy[0], (101, 709)))

        # select the labels of one pump
        operation = operations[pump]
        labels = np.reshape(operation, (-1, 1))

        zero_index = np.where(labels == 0)
        one_index = np.where(labels == 1)
        two_index = np.where(labels == 2)
        three_index = np.where(labels == 3)
        four_index = np.where(labels == 4)


        plt.subplot(5, 1, 1)
        plt.plot(energy[zero_index[0], :].T, '-')
        plt.xticks([])
        plt.ylabel("Energie")
        plt.title('Movement_BSD_X')

        plt.subplot(5, 1, 2)
        plt.plot(energy[one_index[0], :].T, '-')
        plt.xticks([])
        plt.ylabel("Energie")
        plt.title('Operation_Hexagon_Milling_1')

        plt.subplot(5, 1, 3)
        plt.plot(energy[two_index[0], :].T, '-')
        plt.xticks([])
        plt.ylabel("Energie")
        plt.title('Operation_Hexagon_Milling_2')

        plt.subplot(5, 1, 4)
        plt.plot(energy[three_index[0], :].T, '-')
        plt.xticks([])
        plt.ylabel("Energie")
        plt.title('Movement_BSD_Y')

        plt.subplot(5, 1, 5)
        plt.plot(energy[four_index[0], :].T, '-')
        #plt.xticks([])
        plt.ylabel("Energie")
        plt.xlabel("Amplituden")
        plt.title('Positioning_Inactivity')


        plt.suptitle('Pump' + str(pump), fontsize=16)

        plt.show()

#plot_milling_operation_of_all_pumps()





########################################################################################################################
# label failure correction
########################################################################################################################


def revised_labels_of_2F48_Dataset():
    '''
    rectify the labeling failures of the 2F48 dataset
    :return: a revised pickle file
    '''
    pickle_in = open("labels_hex_plus_complete_specgram.pkl", "rb")
    data = pickle.load(pickle_in, encoding='latin1')

    for pump in np.arange(1, 10, 1):
        prepared_date = data[pump][1]
        if pump == 3 or pump == 5 or pump == 6 or pump == 8:
            prepared_date = np.reshape(prepared_date, (-1, 1))
            one_index = np.where(prepared_date == 1)
            two_index = np.where(prepared_date == 2)
            prepared_date[one_index] = 2
            prepared_date[two_index] = 1
            prepared_date = prepared_date.tolist()
        data[pump][1] = prepared_date


    pickle_out = open("revised_labels_hex_plus_complete_specgram.pkl", "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()
    print()


#revised_labels_of_2F48_Dataset()

def revised_labels_of_2F22_Dataset():
    '''
    rectify the labeling failures of the 2F22 dataset
    :return:
    '''
    pickle_in = open("2F22_fp_specgram.pkl", "rb")
    data = pickle.load(pickle_in, encoding='latin1')

    for pump in np.arange(1, 10, 1):
        prepared_date = data[pump][1]
        if pump == 2 or pump == 3 or pump == 7:
            prepared_date = np.reshape(prepared_date, (-1, 1))
            one_index = np.where(prepared_date == 1)
            zero_index = np.where(prepared_date == 0)
            prepared_date[one_index] = 0
            prepared_date[zero_index] = 1
            prepared_date = prepared_date.tolist()
        data[pump][1] = prepared_date

    pickle_out = open("revised_2F22_fp_specgram.pkl", "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()
    print()


#revised_labels_of_2F22_Dataset()

########################################################################################################################

########################################################################################################################

def convert_pickle_to_csv_file():
    '''
    this function converts a pickle file into a csv file
    :return:
    '''
    # load the pickle file
    pickle_in = open("revised_labels_hex_plus_complete_specgram.pkl", "rb")
    data = pickle.load(pickle_in, encoding='latin1')
    # save the data from the pickle file into a csv file
    with open("revised_labels_hex_plus_complete_specgram.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(data)
#convert_pickle_to_csv_file()