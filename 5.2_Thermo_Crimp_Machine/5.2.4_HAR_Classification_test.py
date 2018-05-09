########################################################################################################################
# structure of this pyhton file:
# it contains four functions. The main function declares all hyperparameters and starts the prediction model.
# The predictive model, in turn, calls two functions that define the ANN architecture.
# All support Functions like, data loading or results visualization are imported from the Support_FN.py
########################################################################################################################

import tensorflow as tf
import numpy as np
import sys
import Support_Fn as SFN
from sklearn import metrics

def RNN_Layer(params, dropout, x_input, seqlen):
    '''
    this function creats the RNN-Architecture and compute the predicted
    resistances
    :param params: python dict. which contains all hyperparameter, like num_hidden
    neurons of each RNN-layer all hyperparameters are defined in the model() function
    :param dropout: enables a dropout regularization during training
    :param x_input: correspond the input data (batches of temp. sequences )
    :param seqlen: characterize a ndarry which contains the exact length of
    the input temp. sequences
    :return: the predicted resistances, depends on the x_input
    '''
    # save the variable number of RNN layers into this list variable
    rnn_layers = []
    # create RNN-layers corresponding on the defined num_layer
    # hyperparamter
    for layer in range(params["num_layer"]):
        # define the selected rnn-cells for all layers
        # (corresponding on num_cell hyperparameter)
        if params["rnn_cell"] == "Vanilla":
            # create num_layer Vanilla RNN Cells
            rnn_layer = tf.contrib.rnn.BasicRNNCell(params["num_hidden"])
            # add dropout
            rnn_layer = tf.contrib.rnn.DropoutWrapper(rnn_layer, dropout)

        elif params["rnn_cell"] == "LSTM":
            # create num_layers LSTMCells
            rnn_layer = tf.contrib.rnn.BasicLSTMCell(params["num_hidden"],
                                                     state_is_tuple=True)
            # add dropout
            rnn_layer = tf.contrib.rnn.DropoutWrapper(rnn_layer, dropout)

        elif params["rnn_cell"] == "GRU":
            # create num_layer GRUCells
            rnn_layer = tf.contrib.rnn.GRUCell(params["num_hidden"])
            # add dropout
            rnn_layer = tf.contrib.rnn.DropoutWrapper(rnn_layer, dropout)

        elif params["rnn_cell"] == "LN_LSTM":
            # create LSTM cell with layer normalization
            # [dropout is included]
            rnn_layer = tf.contrib.rnn.LayerNormBasicLSTMCell(
                num_units=params["num_hidden"], dropout_keep_prob=dropout)
        else:
            print("Error: wrong definition of the parameter 'rnn_cell'!")
            sys.exit()

        rnn_layers.append(rnn_layer)

    # stack all RNN layers together
    multi_rnn_cell = tf.contrib.rnn.MultiRNNCell(rnn_layers,
                                                 state_is_tuple=True)
    # compute the hidden- and the output- states
    rnn_outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                           inputs=x_input,
                                           sequence_length=seqlen,
                                           dtype=tf.float32)
    return rnn_outputs

def fc_output_layer(rnn_outputs, seqlen, params, weights, biases):
    '''
    this function creates the full connected outputlayer
    When performing dynamic calculation, we must retrieve the last dynamically computed output.
    However TensorFlow doesn't support advanced indexing yet, so i build a custom op that for each sample in batch size,
    get its length and get the corresponding relevant output.
    :param rnn_outputs: tensor contains the predicted resistances of the RNN
    :param seqlen: seqlen: characterize a ndarry which contains the exact length of
    the input temp. sequences
    :param params: python dict. which contains all hyperparamter (hyperparameter defined in the main() function)
    :param weights: weights of the output layer
    :param biases: baises of the output layer
    :return: the final prediction of the resistances, depends on RNN output
    '''
    with tf.name_scope("full_connectes_output_layer"):
        # rnn_outputs shape = (batch_size, num_steps, num_neurons)
        # hack to build the indexing and retrieve the right output.
        batch_size = tf.shape(rnn_outputs)[0]
        # indices for each sample
        index = tf.range(0, batch_size) * params["num_steps"] + (seqlen - 1)
        # select last outputs
        last = tf.gather(tf.reshape(rnn_outputs, [-1, params["num_hidden"]]), index)
        # predict the output of the RNN Network
        prediction = tf.matmul(last, weights["out"]) + biases["out"]
        # weights["out] has shape (num_neuons, 1)
        # prediction shape = (batch_size, 1)
    return prediction


def model(params, train_temp, train_res, test_temp, test_res, hparam):
    '''
    The model () function consists of five areas:
    1) Definition of variables and placeholders
    2) Call the ANNs
    3) Implementation of the loss function, L2 regularization, performance metrics and the optimizer
    4) Initialization of the calculation graph and start of training and performance evaluation
    5) Visualization of the achieved performance
    :param params: python dictionary with all hyperparameters
    :param train_temp: ndarry of the trainings temp. sequences
    :param train_res: ndarray of the trainings resistances
    :param test_temp: ndarray of the test temp. sequences
    :param test_res: ndarray of the test resistances
    :param hparam: all hyperparameters as string (for unambiguous identification in TB, for example)
    :return: Depending on the application, for example, cloud training (normally nothing)
    '''

    # delete the old graph
    tf.reset_default_graph()
    # initialize the computational graph
    sess = tf.Session()

    ################################################################################################################
    # 1) Definition of placeholders and variables
    ################################################################################################################
    # declare the Placeholders
    x_input = tf.placeholder(tf.float32, [None, params["num_steps"], params["num_input"]], name="x_signals")
    y_output = tf.placeholder(tf.float32, [None, params["num_classes"]], name="HAR")
    # determine the neurons dropout ration
    dropout = tf.placeholder_with_default(1.0, shape=(), name="dropout")
    # a placeholder for indicating each sequence length
    seqlen = tf.placeholder(tf.int32, [None], name="seq_length")

    # Graph weights
    weights = {'out': tf.Variable(tf.truncated_normal(shape=[params["num_hidden"], params["num_classes"]],
                                                      stddev=params["weight_stddev"], mean=1.0), name="output_weights")}
    tf.summary.histogram("weights_out", weights["out"])

    # bias initialization to zero  reduce the exploding and vanishing gradient problem
    biases = {'out': tf.Variable(tf.truncated_normal(shape=[1], stddev=params["bias_stddev"]), name="output_bias")}
    tf.summary.histogram("biases_out", biases["out"])

    ################################################################################################################
    # 2) Call the ANNs
    ################################################################################################################
    # create deep stacked RNN
    rnn_outputs = RNN_Layer(params, dropout, x_input, seqlen)
    # call the fc_ouput_layer to convert the rnn_outputs to the predictions
    predictions = fc_output_layer(rnn_outputs, seqlen, params, weights, biases)

    ################################################################################################################
    # 3) Implementation of the loss function, L2 regularization, performance metrics and the optimizer
    ################################################################################################################
    with tf.name_scope("l2_regularization"):
        # L2 loss prevents this overkill neural network to overfit the data
        l2 = params["delta"] * sum(tf.nn.l2_loss(tf_var) for
                                  tf_var in tf.trainable_variables())

    with tf.name_scope("loss"):
       # softmax_cros_entropy loss function
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_output, logits=predictions)) + l2
        tf.summary.scalar("loss", loss)

    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_output, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    with tf.name_scope("optimizer/train"):
        if params["optimizer"] == "RMSProp":
            optimizer = tf.train.RMSPropOptimizer(params["learning_rate"], decay=params["decay_rate"],
                                                  momentum=params["momentum"])
        else:
            optimizer = tf.train.AdamOptimizer(params["learning_rate"])

        # take gradients of cosst w.r.t. all trainable variables
        gradients = optimizer.compute_gradients(loss)
        # clip the gradients by a pre-defined min and max norm
        clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
        # add the clipped gradients to the optimizer
        train_step = optimizer.apply_gradients(clipped_gradients)

    ################################################################################################################
    # 4) Initialization of the calculation graph and start of training and performance evaluation
    ################################################################################################################
    # summ all tf.summary to one target (all summaries in one graph).
    merged_summary = tf.summary.merge_all()

    # initialize the model variables
    sess.run(tf.global_variables_initializer())

    # Add ops to save and restore all the variables.
    # saver = tf.train.Saver()

    # call the Python class that writes data for TensorBoard
    # writer = tf.summary.FileWriter(TB_LOGDIR + '\hparam')
    writer_train = tf.summary.FileWriter('/output/' + hparam + 'train')
    writer_test = tf.summary.FileWriter('/output/' + hparam + 'test')
    # add the sessions graph to the writer
    writer_train.add_graph(sess.graph)

    # determine the seq. lengths of the test_seq_data
    length_test = np.sum(np.max(np.sign(test_temp), 2), 1)
    # define the evaluation dict
    test_dict = {x_input: test_temp, y_output: test_res, seqlen: length_test}

    train_accuracy_list = []
    test_accuracy_list = []

    for epoch in range(params["num_epochs"]):
        print('acutal_epoch=', epoch)
        for x_train_batch, y_train_batch in SFN.iterator(train_temp, train_res, params["batch_size"]):

            lenght = np.sum(np.max(np.sign(x_train_batch), 2), 1)

            # define the training dict
            train_dict = {x_input: x_train_batch, y_output: y_train_batch, dropout: params["dropout_rate"],
                          seqlen: lenght}

            # Run train step
            sess.run(train_step, feed_dict=train_dict)

        if (epoch + 1) % 2 == 0:
            # define the evaluation dict
            test_dict = {x_input: test_temp, y_output: test_res, seqlen: length_test}
            # Record summaries for TensorBoard
            train_summ = sess.run(merged_summary, feed_dict=train_dict)
            writer_train.add_summary(train_summ, epoch)
            writer_train.flush()
            test_summ = sess.run(merged_summary, feed_dict=test_dict)
            writer_test.add_summary(test_summ, epoch)
            writer_test.flush()

            train_accuracy = sess.run(accuracy, feed_dict=train_dict)
            train_accuracy_list.append(train_accuracy)

            test_accuracy = sess.run(accuracy, feed_dict=test_dict)
            test_accuracy_list.append(test_accuracy)
            print("Epoch%s: train acc = " % epoch, train_accuracy, ' test acc = ', test_accuracy)

    ################################################################################################################
    # 5) Visualization of the achieved performance
    ################################################################################################################
    # plot the accuarcy process over all epochs
    SFN.acc_process_plot(test_accuracy_list, test_accuracy_list)
    # verify the trainingprozess of the model
    pred_y, y = sess.run([predictions, y_output], feed_dict=test_dict)
    # plot a confusion matrix
    # convert the one-hot coding (00001) into a decimal figure (4)
    y_test = pred_y.argmax(1)
    y_true = y.argmax(1)

    # first generate a  confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true, y_test)
    print('Confusion Matrix: ', confusion_matrix)
    labels = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
    SFN.plot_confusion_matrix(confusion_matrix, labels=labels)

    return y, pred_y




def main():
    '''
    The main () function loads the training and test data, declares all hyperparameters, and calls the predictive model.
    In addition, it enables an automatic and iterative grid search for suitable hyperparameters.
    Consequently, this function can be divided into 5 areas
    1) Loading and the defined training and test data regarding selected data set and data shape
    2) Definition of the ANN architecture
    3) Defining the training hyperparameters
    4) Setting the regularization strength
    5) Call the predictive model and thus start the ANN training plus the evaluation of its performance
    :return:
    '''

    # list to save the results of each hyperparameter-combination
    results = []

    ####################################################################################################################
    # 1) load the data and determine their structure
    ####################################################################################################################
    # Those are separate normalised input features for the neural network
    INPUT_SIGNAL_TYPES = ["body_acc_x_", "body_acc_y_", "body_acc_z_", "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
                          "total_acc_x_", "total_acc_y_", "total_acc_z_"]

    TRAIN = "train/"
    TEST = "test/"
    DATASET_PATH = "UCI HAR Dataset/"

    X_train_signals_paths = [
        DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
    ]
    X_test_signals_paths = [
        DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
    ]

    X_train = SFN.load_X(X_train_signals_paths)
    X_test = SFN.load_X(X_test_signals_paths)

    y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
    y_test_path = DATASET_PATH + TEST + "y_test.txt"

    y_train = SFN.one_hot(SFN.load_y(y_train_path))
    y_test = SFN.one_hot(SFN.load_y(y_test_path))

    # -----------------------------------
    # 2) define parameters for model
    # -----------------------------------
    training_data_count = len(X_train)  # 7352 training series
    test_data_count = len(X_test)  # 2947 testing series

    # Input Data
    # the input Data determine the shape of the temp sequences tensor
    num_steps = 128  # equivalent to timesteps
    num_input = 9
    num_classes = 6

    ####################################################################################################################
    # 2) define hyperparameters for RNN-architecture
    ####################################################################################################################
    for num_layer in [2]:
        for num_hidden in [32]:
            for rnn_cell in ["LN_LSTM",]: #or "LSTM" or "GRU"
                ########################################################################################################
                # 3) define training hyperparameters
                ########################################################################################################
                batch_size = 1500
                num_epochs = 250
                for bias_stddev in [1.0]:
                    for weight_stddev in [1.0]:
                        for optimizer in ["ADAM"]:
                            if optimizer == "RMSProp":
                                momentum = 0.85
                                decay_rate = 0.9
                            else:
                                momentum = 0
                                decay_rate = 0
                            for learning_rate in [0.0025]:
                                if rnn_cell == "LN_LSTM":
                                    # tends need higher learning rates
                                    learning_rate = 0.01
                                for activation in [None]:
                                    ####################################################################################
                                    # 4) define regularization hyperparameters
                                    ####################################################################################
                                    for dropout_rate in [0.8]:
                                        for delta in [0.001]:
                                            if rnn_cell == "GRU":
                                                delta = 0.001


                                            # save all selected parameters into a dict
                                            model_params = {'num_input': num_input, 'num_steps': num_steps,
                                                            'rnn_cell': rnn_cell, 'num_layer': num_layer,
                                                            'num_hidden': num_hidden, 'batch_size': batch_size,
                                                            'num_epochs': num_epochs, 'weight_stddev': weight_stddev,
                                                            'bias_stddev': bias_stddev, 'delta': delta,
                                                            'optimizer': optimizer, 'activation': activation,
                                                            'learning_rate': learning_rate, 'dropout_rate': dropout_rate,
                                                            'momentum': momentum, 'decay_rate': decay_rate,
                                                            'num_classes': num_classes}

                                            # create a notation/name containing all the selected parameters.
                                            # This allows a clear identification of the different training runs in
                                            # TensorBoard or during a cloud training
                                            #hparam = SFN.make_hparam_string(model_params)
                                            hparam="test"


                                            ############################################################################
                                            # 5) train and evaluate the recurrent neural network
                                            ############################################################################

                                            # runs the model with the selected model_params
                                            train_model = model(model_params, X_train, y_train,
                                                                X_test, y_test, hparam)
                                            # save the results
                                            results.append(train_model)

    print('-----------------FINISH-----------------------')


if __name__ == '__main__':
    main()