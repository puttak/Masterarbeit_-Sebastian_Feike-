import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def load_padding_and_split_all_data(params, all_data=True):
    '''
    1) load all sequences and resistances from the pickle file
    2) padding all temp. sequences to the same length
    3) shuffle temp. sequences and resistances randomly
    :return: shuffled temperature and resistances data
    '''
    # load the "crimp_train_data.pkl" file
    file = pd.read_pickle('crimp_train_data.pkl')
    # convert it to a Dataframe
    df = pd.DataFrame(file)
    # select the column "data" and "resistance"
    # contains all resistances
    resistance = df['resistance']
    # contains measured data (timesteps, energy and termperature) for each measurement attempt
    data = df['data']
    attempt = df['name']

    res_list = []
    temp_seq_list = []
    for i in range(len(data)):
        if all_data == False:
            if attempt[i] == "Versuch42" or attempt[i] == "Versuch44":
                # select one attempt
                # contains:
                # timesteps of each measurements (column 0) and the corresponding
                # measured energy (column 1)
                # measured temperature (column 2)
                one_attempt = data[i]
                # determine their length
                attempt_length = len(one_attempt)
                # determine the number of required zeros
                num_zeros = 80 - attempt_length
                # select the temp. seq. of the selected attempt
                # which is included in the 3 column
                one_temp_seq = np.reshape(one_attempt[:, 2], (-1, 1))
                # use padding with zeros
                uniform_temp_seq = np.pad(one_temp_seq, [(0, num_zeros), (0, 0)],
                                          mode='constant', constant_values=0)
                # save the uniformed temp. seq  into the temp_seq_list
                temp_seq_list.extend(uniform_temp_seq)
                test = resistance[i]
                res_list.append(resistance[i])
        else:
            # select one attempt
            # contains:
            # timesteps of each measurements (column 0) and the corresponding
            # measured energy (column 1)
            # measured temperature (column 2)
            one_attempt = data[i]
            # determine their length
            attempt_length = len(one_attempt)
            # determine the number of required zeros
            num_zeros = 80 - attempt_length
            # select the temp. seq. of the selected attempt
            # which is included in the 3 column
            one_temp_seq = np.reshape(one_attempt[:, 2], (-1, 1))
            # use padding with zeros
            uniform_temp_seq = np.pad(one_temp_seq, [(0, num_zeros), (0, 0)],
                                      mode='constant', constant_values=0)
            # save the uniformed temp. seq  into the temp_seq_list
            temp_seq_list.extend(uniform_temp_seq)
    # transform all uniformed temp. sequences into a suitable shape
    temp_sequences = np.reshape(temp_seq_list, (-1, 80))

    if all_data == False:
        resistance = res_list
    resistance = np.reshape(resistance, (-1, 1))

    # select all indexes where resistances higher than > 0.01 Ohm
    failure_res_ix = np.where(resistance >= 0.01)
    # use reshape to convert the pandas series into an ndarray
    failure_res = np.reshape(resistance[failure_res_ix[0]], (-1, 1))
    failure_temp_seq = temp_sequences[failure_res_ix[0]]

    # select all indexes where resistances higher than > 0.01 Ohm
    good_res_ix = np.where(resistance < 0.01)
    good_res = np.reshape(resistance[good_res_ix[0]], (-1, 1))
    good_temp_seq = temp_sequences[good_res_ix[0]]

    def train_test_data_split(temp_seq, res, params):
        '''
        This function shuffles the resistance and temperature data randomly.
        In addition, the subdivision into training and test data takes place according to the defined test_size
        :param temp_seq: temperature sequences of the failure or good resistances
        :param res: good or defect resistances
        :param params: contains the test size
        :return: train and test temp. sequences and resistances of the good or failure crimp processes
        '''

        # define the trainings and test indices
        train_ix = np.random.choice(len(res), round(len(res)*params["test_size"]), replace = False)
        test_ix = np.array(list(set(range(len(res))) - set(train_ix)))
        # shuffle and split the temp and resistance data according the indices
        temp_seq_train = temp_seq[train_ix]
        temp_seq_test = temp_seq[test_ix]
        res_train = res[train_ix]
        res_test = res[test_ix]
        return temp_seq_train, temp_seq_test, res_train, res_test

    # split the failure data into train and test datasets
    train_failure_temp_seq, test_failure_temp_seq, train_failure_res, test_failure_res = train_test_data_split(
        failure_temp_seq, failure_res, params)

    # split the good data into train and test datasets
    train_good_temp_seq, test_good_temp_seq, train_good_res, test_good_res  = train_test_data_split(
        good_temp_seq, good_res, params)

    # create the train dataset and convert it to a pandas dataframe
    train_temp_seq = np.concatenate((train_good_temp_seq, train_failure_temp_seq), axis=0)
    train_res = np.reshape(np.concatenate((train_good_res, train_failure_res), axis=0), (-1,1))
    train_data = np.concatenate((train_temp_seq, train_res), axis=1)
    df_train = pd.DataFrame(data=train_data, dtype=np.float64)

    # create test dataset and convert it to a pandas dataframe
    test_temp_seq = np.concatenate((test_good_temp_seq, test_failure_temp_seq), axis=0)
    test_res = np.reshape(np.concatenate((test_good_res, test_failure_res), axis=0), (-1, 1))
    test_data = np.concatenate((test_temp_seq, test_res), axis=1)
    df_test = pd.DataFrame(data=test_data, dtype=np.float64)

    if all_data == False:
        # save the train and test dataframe as csv file
        df_train.to_csv('Training_Dataset(42_44).csv', index=False)
        df_test.to_csv('Test_Dataset(42_44).csv', index=False)
    else:
        # save the train and test dataframe as csv file
        df_train.to_csv('Training_Dataset.csv', index=False)
        df_test.to_csv('Test_Dataset.csv', index=False)




def main():
    '''

    This function can be used to set the proportion of test data.
    In addition, the parameter all_Data determines whether a training or test dataset should be created
    with all measurements or only with the measurements of trials 42 and 44
    :return:
    '''
    # choose the percentage of the test dataset
    test_size = 0.2
    params = {'test_size': test_size}
    # if all data = False then only measure attempt 42 and 44 are regarded
    #load_padding_and_split_all_data(params, all_data=False)



if __name__ == '__main__':
    main()