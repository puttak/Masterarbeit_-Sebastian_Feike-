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

    # stack all RNN layers together
    multi_rnn_cell = tf.contrib.rnn.MultiRNNCell(rnn_layers,
                                                 state_is_tuple=True)
    # compute the hidden- and the output- states
    rnn_outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                           inputs=x_input, dtype=tf.float32)
    return rnn_outputs

def fc_output_layer(rnn_outputs, params, weights, biases):
    '''
    this function creates the full connected outputlayer
    :param rnn_outputs: tensor contains the predicted resistances of the RNN
    :param weights: weights of the output layer
    :param biases: baises of the output layer
    :return: the final prediction of the resistances, depends on RNN output
    '''
    with tf.name_scope("full_connectes_output_layer"):
        rnn_outputs = tf.transpose(rnn_outputs, [1,0,2])
        rnn_outputs = rnn_outputs[-1]

        # predict the output of the RNN Network
        prediction = tf.matmul(rnn_outputs, weights["out"]) + biases["out"]

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
    x_input = tf.placeholder(tf.float32, [None, params["num_steps"], params["num_input"]], name="temp_peaks")
    y_output = tf.placeholder(tf.float32, [None, params["num_classes"]], name="resistance")
    # determine the neurons dropout ration
    dropout = tf.placeholder_with_default(1.0, shape=(), name="dropout")

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
    rnn_outputs = RNN_Layer(params, dropout, x_input)
    # call the fc_ouput_layer to convert the rnn_outputs to the predictions
    predictions = fc_output_layer(rnn_outputs, params, weights, biases)

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

    # define the evaluation dict
    test_dict = {x_input: test_temp, y_output: test_res}

    train_accuracy_list = []
    test_accuracy_list = []

    for epoch in range(params["num_epochs"]):
        print('acutal_epoch=', epoch)
        for x_train_batch, y_train_batch in SFN.iterator(train_temp, train_res, params["batch_size"]):

            # define the training dict
            train_dict = {x_input: x_train_batch, y_output: y_train_batch, dropout: params["dropout_rate"]}

            # Run train step
            sess.run(train_step, feed_dict=train_dict)

        if (epoch + 1) % 2 == 0:
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
    labels = ["> 0.01 Ohm", ">= 0.01 Ohm"]
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
    # select the required Dataset
    # all_data = False --> contains only measurement attempt 42 and 44
    train_temp, train_res, test_temp, test_res = SFN.load_peak_train_and_test_data(all_data=False)

    # convert the peaks into num_steps und num_input shape
    num_steps = 8
    num_input = 1
    train_temp = np.reshape(train_temp, (len(train_temp), num_steps, num_input))
    test_temp = np.reshape(test_temp, (len(test_temp), num_steps, num_input))

    # define the num_classes
    num_classes = len(train_res[1])

    ####################################################################################################################
    # 2) define hyperparameters for RNN-architecture
    ####################################################################################################################
    for num_layer in [3]:
        for num_hidden in [5]:
            for rnn_cell in ["LSTM", "GRU"]:  # or "LSTM" or "GRU"
                ########################################################################################################
                # 3) define training hyperparameters
                ########################################################################################################
                batch_size = 20
                num_epochs = 1000
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
                                    learning_rate = 0.012
                                for activation in [None]:
                                    ####################################################################################
                                    # 4) define regularization hyperparameters
                                    ####################################################################################
                                    for dropout_rate in [0.8]:
                                        for delta in [0.0001]:
                                            if rnn_cell == "GRU":
                                                delta = 0.00001

                                            # save all selected parameters into a dict
                                            model_params = {'num_input': num_input, 'num_steps': num_steps,
                                                            'rnn_cell': rnn_cell, 'num_layer': num_layer,
                                                            'num_hidden': num_hidden, 'batch_size': batch_size,
                                                            'num_epochs': num_epochs, 'weight_stddev': weight_stddev,
                                                            'bias_stddev': bias_stddev, 'delta': delta,
                                                            'optimizer': optimizer, 'activation': activation,
                                                            'learning_rate': learning_rate, 'num_classes': num_classes,
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
                                            train_model = model(model_params, train_temp, train_res,
                                                                test_temp, test_res, hparam)
                                            # save the results
                                            results.append(train_model)
    print('-----------------FINISH-----------------------')


if __name__ == '__main__':
    main()

