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
        prediction = tf.nn.sigmoid(tf.matmul(last, weights["out"]) + biases["out"])
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
    # input data (temp. sequenzes)
    x_input = tf.placeholder(tf.float32, [None, params["num_steps"], params["num_input"]], name="temperature")
    # output/ target values (resistances)
    y_output = tf.placeholder(tf.float32, [None, 1], name="resistance")
    # determine the neurons dropout ration
    dropout = tf.placeholder_with_default(1.0, shape=(), name="dropout")
    # a placeholder for indicating each sequence length
    seqlen = tf.placeholder(tf.int32, [None], name="seq_length")

    # declare weights and biases for the full connected output layer
    weights = {'out': tf.Variable(tf.truncated_normal(shape=[params["num_hidden"], 1],
                                                      stddev=params["weight_stddev"]), name="weights_out")}
    tf.summary.histogram("weights_out", weights["out"])
    # bias initialization to zero  reduce the exploding and vanishing gradient problem
    biases = {'out': tf.Variable(tf.truncated_normal(shape=[1], stddev=params["bias_stddev"]), name="bias_out")}
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
        # mean squared error loss function
        loss = tf.reduce_mean(tf.square(predictions - y_output)) + l2
        tf.summary.scalar("loss", loss)

    with tf.name_scope("rmse"):
        # compute the root mean squared error
        rmse = tf.reduce_mean(tf.sqrt(tf.square(predictions - y_output)), name="rmse")
        tf.summary.scalar("rmse", rmse)

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
    writer_train = tf.summary.FileWriter('/output/' + hparam + 'train')
    writer_test = tf.summary.FileWriter('/output/' + hparam + 'test')
    # add the sessions graph to the writer
    writer_train.add_graph(sess.graph)

    # determine the seq. lengths of the test_seq_data
    length_test = np.sum(np.max(np.sign(test_temp), 2), 1)
    # define the evaluation dict
    test_dict = {x_input: test_temp, y_output: test_res, seqlen: length_test}

    train_loss_list = []
    train_rmse_list = []
    test_loss_list = []
    test_rmse_list = []

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

            train_loss, train_rmse = sess.run([loss, rmse], feed_dict=train_dict)
            train_loss_list.append(train_loss)
            train_rmse_list.append(train_rmse)

            test_loss, test_rmse = sess.run([loss, rmse], feed_dict=test_dict)
            test_loss_list.append(test_loss)
            test_rmse_list.append(test_rmse)
            print("Epoch%s: train loss = " % epoch, train_loss, ' test loss = ', test_loss)
            print("Epoch%s: train rmse = " % epoch, train_rmse, ' test rmse = ', test_rmse)

    ################################################################################################################
    # 5) Visualization of the achieved performance
    ################################################################################################################
    # plot the loss process over all epochs
    SFN.loss_process_plot(test_loss_list, train_loss_list)
    #plot the rmse process over all epochs
    SFN.loss_process_plot(test_rmse_list, train_rmse_list)

    # verify the trainingprozess of the model
    pred_y, y = sess.run([predictions, y_output], feed_dict=test_dict)
    # plot a q_q_plot
    SFN.correlation_test(pred_y, y)

    # verify the trainingprozess of the model
    SFN.q_q_plot(pred_y)

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
    # Input Data
    # the input Data determine the shape of the temp sequences tensor
    num_steps = 80 # equivalent to timesteps
    num_input = 1  # input parameters per timestep
    # load the train and test data corresponding to the selected:
    # - num_steps and num_input size
    # - the third option occur a Trucated BPTT transformation with 50% overlapping
    train_temp, train_res, test_temp, test_res = SFN.load_train_and_test_data(num_steps, num_input, TBTT=False,
                                                                              all_data=False)

    ####################################################################################################################
    # 2) define hyperparameters for RNN-architecture
    ####################################################################################################################
    for num_layer in [3]:
        for num_hidden in [70]:
            for rnn_cell in ["LSTM"]: #or "LSTM" or "GRU" or "LN_LSTM or "Vanilla"
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
                            for learning_rate in [0.001]:
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
                                                            'learning_rate': learning_rate, 'dropout_rate': dropout_rate,
                                                            'momentum': momentum, 'decay_rate': decay_rate,}

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

    SFN.compare_Correlation(results)
    print('-----------------FINISH-----------------------')


if __name__ == '__main__':
    main()