########################################################################################################################
# structure of this pyhton file:
# it contains four functions. The main function declares all hyperparameters and starts the prediction model.
# The predictive model, in turn, calls two functions that define the ANN architecture.
# All support Functions like, data loading or results visualization are imported from the Support_FN.py
########################################################################################################################

import tensorflow as tf
import sys
from sklearn import metrics
import Support_Fn as SFN
import time

def fc_hidden_layer(x_input, params, weights, biases):
    '''
    creats a full-connected hidden layer with the target to transform (activate) the input data to non-linear
    :param x_input: temp. sequence tensor
    :param params: pytion dictionary with all hyperparameters
    :param weights: weights of the additional hidden layer
    :param biases: biases of the additional hidden layer
    :return: transformed and activated x-input
    '''
    with tf.name_scope("Fc_hidden_layer"):
        # (NOTE: This step could be greatly optimised by shaping the dataset once
        # input shape: (batch_size, n_steps, n_input)
        x_input = tf.transpose(x_input, [1, 0, 2])  # permute n_steps and batch_size
        # Reshape to prepare input to hidden activation
        x_input = tf.reshape(x_input, [-1, params["num_input"]])
        # new shape: (n_steps*batch_size, n_input)

        # Linear activation
        x_input = tf.nn.relu(tf.matmul(x_input, weights['hidden']) + biases['hidden'])
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        x_input = tf.split(x_input, params["num_steps"], 0)
        # new shape: n_steps * (batch_size, n_hidden)
    return x_input


def RNN_Layer(params, dropout, x_input):
    '''
    this function creats the RNN-Architecture and compute the predicted
    resistances
    :param params: python dict. which contains all hyperparameter, like num_hidden
    neurons of each RNN-layer all hyperparameters are defined in the model() function
    :param dropout: enables a dropout regularization during training
    :param x_input: correspond the input data (batches of temp. sequences )
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

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.contrib.rnn.MultiRNNCell(rnn_layers, state_is_tuple=True)
    # rnn_outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=x_input, dtype=tf.float32)
    rnn_outputs, state = tf.contrib.rnn.static_rnn(cell=multi_rnn_cell, inputs=x_input,dtype=tf.float32)
    return rnn_outputs

def fc_output_layer(rnn_outputs, weights, biases):
    '''
    this function creates the full connected outputlayer
    no further data transformations are necessary as the sensor data always have the same length and the rnn_static ()
    api was used in the RNN_layer () function (see function above)
    :param rnn_outputs: tensor contains the predicted resistances of the RNN
    :param weights: weights of the output layer
    :param biases: baises of the output layer
    :return: the final prediction of the resistances, depends on RNN output
    '''
    with tf.name_scope("Fc_output_layer"):
        # select last outputs
        last = rnn_outputs[-1]
        # predict the output of the RNN Network
        prediction = tf.matmul(last, weights["out"] + biases["out"])
        # weights["out] has shape (num_neuons, num_classes)
        # prediction shape = (batch_size, num_classes)

    return prediction


def model(params, x_train, y_train, x_test, y_test, hparam):
    '''
    The model () function consists of six areas:
    1) Definition of variables and placeholders
    2) Call the ANNs
    3) Implementation of the loss function, L2 regularization, performance metrics and the optimizer
    4) Initialization of the calculation graph and start of training and performance evaluation
    5) Visualization of the achieved performance
    6) Measuring the forecast times for all training data (individually)
    :param params: python dictionary with all hyperparameters
    :param x_train: ndarry of the trainings sensor signals
    :param train_res: ndarray of the trainings conditions
    :param test_temp: ndarray of the test sensor signals
    :param test_res: ndarray of the test conditions
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
    x_input = tf.placeholder(tf.float32, [None, params["num_steps"], params["num_input"]], name="energy_sequence")
    y_output = tf.placeholder(tf.float32, [None, params["num_classes"]], name="milling_operation")
    dropout = tf.placeholder_with_default(1.0, shape=(), name="dropout")

    # Graph weights
    weights = {'hidden': tf.Variable(tf.truncated_normal(shape=[params["num_input"], params["num_hidden"]],
                                                      stddev=params["weight_stddev"]), name="hidden_weights"),
               'out': tf.Variable(tf.truncated_normal(shape=[params["num_hidden"], params["num_classes"]],
                                                      stddev=params["weight_stddev"], mean=1.0), name="output_weights")}
    tf.summary.histogram("weights_out", weights["hidden"])
    tf.summary.histogram("weights_out", weights["out"])

    # bias initialization to zero  reduce the exploding and vanishing gradient problem
    biases = {'hidden': tf.Variable(tf.truncated_normal(shape=[params["num_hidden"]], stddev=params["bias_stddev"]),
                                 name="hidden_bias"),
              'out': tf.Variable(tf.truncated_normal(shape=[params["num_classes"]], stddev=params["bias_stddev"]),
                                 name="output_bias")}
    tf.summary.histogram("biases_out", biases["hidden"])
    tf.summary.histogram("biases_out", biases["out"])

    ################################################################################################################
    # 2) Call the ANNs
    ################################################################################################################
    # call hidden fc
    reshaped_x_input = fc_hidden_layer(x_input, params, weights, biases)
    # call the RNN_Layer
    rnn_outputs = RNN_Layer(params, dropout, reshaped_x_input)
    # call the fc_ouput_layer to convert the rnn_outputs to the predictions
    predictions = fc_output_layer(rnn_outputs, weights, biases)

    ################################################################################################################
    # 3) Implementation of the loss function, L2 regularization, performance metrics and the optimizer
    ################################################################################################################
    with tf.name_scope("l2_Regularization"):
        # L2 loss prevents this overkill neural network to overfit the data
        l2 = params["delta"] * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

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
    writer_train = tf.summary.FileWriter('/output/' + 'train')
    writer_test = tf.summary.FileWriter('/output/' + 'test')
    # add the sessions graph to the writer
    writer_train.add_graph(sess.graph)

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for epoch in range(params["num_epochs"]):

        print('acutal_epoch=', epoch)
        # define the evaluation dict
        test_dict = {x_input: x_test, y_output: y_test}
        for x_train_batch, y_train_batch in SFN.iterator(x_train, y_train, params["batch_size"]):
            # define the training dict
            train_dict = {x_input: x_train_batch, y_output: y_train_batch, dropout: params["dropout_rate"]}

            # Run train step
            sess.run(train_step, feed_dict=train_dict)

        if (epoch + 1) % 2 == 0:
            # measure the run time
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            # record train summaries and runtime statistics for TensorBoard
            train_summ = sess.run(merged_summary, feed_dict=train_dict, options=run_options, run_metadata=run_metadata)
            writer_train.add_run_metadata(run_metadata, 'step%d' % epoch)
            writer_train.add_summary(train_summ, epoch)
            writer_train.flush()
            # record test summaries --> actually, time measurements are not possible für testing data
            test_summ = sess.run(merged_summary, feed_dict=test_dict)
            writer_test.add_summary(test_summ, epoch)
            writer_test.flush()

            train_acc = sess.run(accuracy, feed_dict=train_dict)
            train_acc_list.append(train_acc)

            test_acc = sess.run(accuracy, feed_dict=test_dict)
            test_acc_list.append(test_acc)
            print("Epoch%s: train acc = " % epoch, train_acc, ' val acc = ', test_acc)

        if (epoch + 1) % params["num_epochs"] == 0:
            # finally, evaluate the classifier based on Performance Measures

            # test_dict = {x_input: x_test, y_output: y_test}
            # metrics
            start_time = time.time()
            y_pred, y_true = sess.run([predictions, y_output], feed_dict=test_dict)
            print("--- %s seconds ---" % (time.time() - start_time))

            # convert the one-hot coding (00001) into a decimal figure (4)
            y_pred = y_pred.argmax(1)
            y_true = y_true.argmax(1)

            # first generate a  confusion matrix
            confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
            print('Confusion Matrix: ', confusion_matrix)

            # second compute the Precision- and Recall- and F1-score
            # use the macro average because the individual classes occur with varying degrees of frequency
            precision = metrics.precision_score(y_true, y_pred, average="macro")
            recall = metrics.recall_score(y_true, y_pred, average="macro")
            f1 = metrics.f1_score(y_true, y_pred, average="macro")
            print("Precision: " + str(precision), " Recall: " + str(recall), " F1: " + str(f1))

    ################################################################################################################
    # 5) Visualization of the achieved performance
    ################################################################################################################
    # plot the accuracy process over all epochs
    SFN.acc_process_plot(test_acc_list, train_acc_list)
    # plot the confusion matix
    SFN.plot_confusion_matrix(confusion_matrix)
    # results = {'plott_pred': [plott_pred_list], 'plott_true': [plott_true_list]}
    # params.update(results)

    ################################################################################################################
    # 6) Measuring the forecast times for all training data (individually)
    ################################################################################################################
    sum_times = 0.0
    iterations = 0
    time_measurements_list = []
    # determine how many labels should be predict at the same time
    time_batch_size = 1
    for x_time_batch, y_time_batch in SFN.iterator(x_test, y_test, time_batch_size):
        # count the number of time measurements for the average-computational-time
        iterations += 1
        # define the new feed dict
        time_dict = {x_input: x_time_batch, y_output: y_time_batch}
        # determine a new start time
        start_time = time.clock()
        # predict labels
        sess.run(predictions, feed_dict=time_dict)
        # determine the end time
        end_time = time.clock()
        # calculate the difference between the prediction start and end
        time_dif = (end_time- start_time)
        # sum all time differences
        sum_times += time_dif
        # save the time difference into a python list
        time_measurements_list.append(time_dif)
    # finally, plot the time measurements and the average prediction time
    SFN.plot_time_measurements(time_measurements_list, (sum_times/iterations), time_batch_size)


    return



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

    # define a python list to save the results
    results = []

    ####################################################################################################################
    # 1) load the data and their structure
    ####################################################################################################################
    # select the required Dataset
    dataset = "2F48"
    #dataset = "2F5"
    #dataset = "2F22"

    x_train, y_train, x_test, y_test, num_steps, num_input = SFN.load_train_test_data_for_RNN(dataset)

    # determine the number of classes
    num_classes = len(y_train[1])

    ####################################################################################################################
    # 2) define hyperparameters for RNN-architecture
    ####################################################################################################################
    for num_layer in [3]:
        for num_hidden in [50]:
            for rnn_cell in ["LSTM"]:  # or "LSTM" or "GRU" or "LN_LSTM or "Vanilla"
                ########################################################################################################
                # 3) define training hyperparameters
                ########################################################################################################
                batch_size = 200
                num_epochs = 2#150
                for bias_stddev in [0.0]:
                    for weight_stddev in [0.1]:
                        for optimizer in ["ADAM"]:
                            if optimizer == "RMSProp":
                                momentum = 0.85
                                decay_rate = 0.9
                            else:
                                momentum = 0
                                decay_rate = 0
                            for learning_rate in [0.0001]:
                                if rnn_cell == "LN_LSTM":
                                    # tends need higher learning rates
                                    learning_rate = 0.001
                                for activation in [None]:
                                    ####################################################################################
                                    # 4) define regularization hyperparameters
                                    ####################################################################################
                                    for dropout_rate in [0.85]:
                                        for delta in [0.0001]:
                                            if rnn_cell == "GRU":
                                                delta = 0.000001

                                                # save all selected parameters into a dict
                                            model_params = {'num_input': num_input, 'num_steps': num_steps,
                                                            'rnn_cell': rnn_cell, 'num_layer': num_layer,
                                                            'num_hidden': num_hidden, 'batch_size': batch_size,
                                                            'num_epochs': num_epochs, 'num_classes': num_classes,
                                                            'weight_stddev': weight_stddev,
                                                            'bias_stddev': bias_stddev, 'delta': delta,
                                                            'optimizer': optimizer, 'activation': activation,
                                                            'learning_rate': learning_rate,
                                                            'dropout_rate': dropout_rate,
                                                            'momentum': momentum, 'decay_rate': decay_rate, }

                                            # create a notation/name containing all the selected parameters.
                                            # This allows a clear identification of the different training runs in
                                            # TensorBoard or during a cloud training
                                            hparam = SFN.make_hparam_string(model_params)

                                            ############################################################################
                                            # 5) train and evaluate the recurrent neural network
                                            ############################################################################

                                            # runs the model with the selected model_params
                                            train_model = model(model_params, x_train, y_train, x_test, y_test,
                                                                hparam)
                                            # save the results
                                            results.append(train_model)

    print('-----------------FINISH-----------------------')


if __name__ == '__main__':
    main()

