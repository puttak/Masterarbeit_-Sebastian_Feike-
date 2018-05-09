import pandas as pd
import numpy as np
import pickle

def load_and_save_data():
    '''
    1) load the "revised_labels_hex_plus_complete_specgram.pkl" file
    2) select the measurement vibration measurements (energy_amplitudes) and the corresponding labels of each pump (1-9)
    3) split the train and test dataset according complete pumps
    5) save the the train and test data into csv files
    :return:
    '''

    # load the "revised_labels_hex_plus_complete_specgram.pkl" file
    file = pd.read_pickle('revised_labels_hex_plus_complete_specgram.pkl')
    # convert it to a Dataframe
    df = pd.DataFrame(file)
    # select column vibration measurements (energy amplitudes)
    all_measurements = df[0]
    # select the labels --> milling-operation-processes
    operations = df[1]

    # define lists to save the train and test features and labels
    train_energy_list = []
    train_label_list = []
    test_energy_list = []
    test_label_list = []

    # save for every pump the energy_sequence and the corresponding labels into the energy- and label_list
    for pump in range(1, len(df)):
        # select the energy sequence of one pump
        measurements_per_pump  = all_measurements[pump]
        # convert the shape
        energy = np.transpose(np.reshape(measurements_per_pump[0], (101, 709)))

        # select the labels of one pump
        operation = operations[pump]
        # convert the shape
        labels = np.reshape(operation, (-1, 1))

        # save as test data if pump is 2 or 8 into a list
        if pump == 2 or pump == 8:
            test_energy_list.append(energy)
            test_label_list.append(labels)
        else:
            # save as train data into a list
            train_energy_list.append(energy)
            train_label_list.append(labels)

    def convert_lists(energy_list, label_list):
        '''
        convert the lists into two arrays and concatenate them
        :param energy_list: train or test energy data
        :param label_list: train or test labels
        :return: concatenated numpy ndarray
        '''
        # convert into a numpy ndarray
        x = np.reshape(energy_list, (-1, 101))
        y = np.reshape(label_list, (-1, 1))
        # concatenate features and labels
        energy_sequence_plus_label = np.concatenate((x, y), axis=1)
        return energy_sequence_plus_label

    # stacked features and labels into a pandas dataframe
    df_train = pd.DataFrame(convert_lists(train_energy_list, train_label_list))
    df_test = pd.DataFrame(convert_lists(test_energy_list, test_label_list))

    # save the datasets into csv files
    df_train.to_csv('2F48_training_dataset(pump_1_3_4_5_6_7_9).csv', index=False)
    df_test.to_csv('2F48_Test_dataset(pump_2_8).csv', index=False)



if __name__ == '__main__':
    '''
    '''
    #load_and_save_data()