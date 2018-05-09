########################################################################################################################
# structure of this pyhton file:
# it contains four functions. The main function declares all hyperparameters and starts the prediction model.
# The predictive model, in turn, calls two functions that define the ANN architecture.
# All support Functions like, data loading or results visualization are imported from the Support_FN.py
########################################################################################################################

import tensorflow as tf
from sklearn import metrics
import Support_Fn as SFN
import time


def generate_FCNN_architecture(params, x_input, dropout, batch_norm):
    """
    This feature creates all FC layers, reducing the implementation effort
    with higher APIs. As a result, no additional function will be needed
    which initializes the weights and biases When creating FC layer, the
    following should be noted:
    the last layer is always the output layer. This differs from the
    hidden layers because in the last layer no RelU activation function
    should be used. In the hidden layers, a ReLU is recommended in
    contrast. For this reason, a distinction must be made between hidden
    and output layers when creating the individual FC layers
    (depending on the number of num_layers).
    :param params: python library containing all hyperparameters
    (definition of the hyperparameters within the main () function)
    :param x_input: ndarray of input data (structure-borne sound measurements)
    :param dropout: dropout parameter which is less than 1 during training
    (regularization)
    :return: the predicted output of the output layer
    """

    hidden_neurons_list = params["hidden_neurons"]

    # initialize the input with x_input data
    input = x_input

    # create as many FC layers as defined in the hyperparameter num_layer
    for layer in range(params["FFNN_layer"]):
        # distinguishes between hidden and output layers in FC layer
        # creation
        if layer < (params["FFNN_layer"] - 1):
            # hidden layer
            with tf.name_scope("fc_hidden_layer_".format(layer)):
                # calculates only the input - weight multiplication
                # and bias additon
                pre_act_output = \
                    tf.contrib.layers.fully_connected(input, params["hidden_neurons"][layer],activation_fn=None)

                # performing layer normalization
                layer_norm = \
                    tf.contrib.layers.batch_norm(pre_act_output,
                                                 center=True,
                                                 scale=True,
                                                 is_training=batch_norm)
                # just now layer activation
                act_output = tf.nn.relu(layer_norm, name="act_output")

                # apply dropout
                dropout_output = tf.nn.dropout(act_output, dropout, name="dropout_".format(layer))
                input = dropout_output
        else:
            # output layer
            with tf.name_scope("output_layer"):
                layer_output = \
                    tf.contrib.layers.fully_connected(input, params["hidden_neurons"][layer],activation_fn=None)
            return layer_output


def model(params, x_train, y_train, x_test, y_test):
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
    :return: Depending on the application, for example, cloud training (normally nothing)
    '''
    # delete the old graph
    tf.reset_default_graph()
    # initialize the computational graph
    sess = tf.Session()

    ################################################################################################################
    # 1) Definition of placeholders (variables will be defined in the generate_FCNN_architecture()
    ################################################################################################################
    # declare the Placeholders
    x_input = tf.placeholder(tf.float32, [None, params["num_features"]], name="measured_amplitudes")
    y_output = tf.placeholder(tf.float32, [None, params["num_classes"]], name="milling_operation")
    dropout = tf.placeholder_with_default(1.0, shape=(), name="dropout")
    batch_norm = tf.placeholder_with_default(False, shape=(), name="batch_norm")

    ################################################################################################################
    # 2) Call the FCNN
    ################################################################################################################
    predictions = generate_FCNN_architecture(params, x_input, dropout, batch_norm)

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

    with tf.name_scope("optimizer"):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if params["optimizer"] == "RMSProp":
                optimizer = tf.train.RMSPropOptimizer(params["learning_rate"], decay=params["decay_rate"],
                                                      momentum=params["momentum"])

            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])

            train_step = optimizer.minimize(loss)

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

    train_acc_list = []
    test_acc_list = []

    for epoch in range(params["num_epochs"]):

        print('acutal_epoch=', epoch)
        # define the evaluation dict
        test_dict = {x_input: x_test, y_output: y_test}
        for x_train_batch, y_train_batch in SFN.iterator(x_train, y_train, params["batch_size"]):
            # define the training dict
            train_dict = {x_input: x_train_batch, y_output: y_train_batch, dropout: params["dropout_rate"],
                          batch_norm: True}

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
            y_test, y_true = sess.run([predictions, y_output], feed_dict=test_dict)
            print("--- %s seconds ---" % (time.time() - start_time))

            # convert the one-hot coding (00001) into a decimal figure (4)
            y_test = y_test.argmax(1)
            y_true = y_true.argmax(1)

            # first generate a  confusion matrix
            confusion_matrix = metrics.confusion_matrix(y_true, y_test)
            print('Confusion Matrix: ', confusion_matrix)

            # second compute the Precision- and Recall- and F1-score
            # use the macro average because the individual classes occur with varying degrees of frequency
            precision = metrics.precision_score(y_true, y_test, average="macro")
            recall = metrics.recall_score(y_true, y_test, average="macro")
            f1 = metrics.f1_score(y_true, y_test, average="macro")
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
    print("Please be patient, the prediction time will take longer for RNNs")
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

    return #time_measurements_list, (sum_times/iterations), time_batch_size

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
    results = []

    ####################################################################################################################
    # 1) load the data and their structure
    ####################################################################################################################
    # select the required Dataset
    dataset = "2F48"
    #dataset = "2F5"
    #dataset = "2F22"

    x_train, y_train, x_test, y_test = SFN.load_train_test_data_for_FFNN(dataset)

    # the input data determine the shape of the features and the amount of classes
    num_features = len(x_train[1])
    num_classes = len(y_train[1])
    ####################################################################################################################
    # 2) define hyperparameters for RNN-architecture
    ####################################################################################################################
    # the following hyperparameter "FFNN_layer" (determines the number ob layer)
    # let you choose which predefined FCNN structure to select
    # the best result has been educated with the 4-layer network so far
    for FFNN_layer in [4]:

        if FFNN_layer == 2: # possible accuracy up to 90%
            hidden_neurons_layer1 = 101
            hidden_neurons_layer2 = num_classes
            hidden_neurons_layer3 = 0
            hidden_neurons_layer4 = 0
            hidden_neurons_layer5 = 0

        elif FFNN_layer == 3: # possible accuracy up to 95,5%
            hidden_neurons_layer1 = 101
            hidden_neurons_layer2 = 50
            hidden_neurons_layer3 = num_classes
            hidden_neurons_layer4 = 0
            hidden_neurons_layer5 = 0

        elif FFNN_layer == 4: # possible accuracy up to 96,2%
            hidden_neurons_layer1 = 200
            hidden_neurons_layer2 = 101
            hidden_neurons_layer3 = 50
            hidden_neurons_layer4 = num_classes
            hidden_neurons_layer5 = 0

        elif FFNN_layer == 5:
            hidden_neurons_layer1 = 200
            hidden_neurons_layer2 = 100
            hidden_neurons_layer3 = 50
            hidden_neurons_layer4 = 25
            hidden_neurons_layer5 = num_classes
        # save the selected numbers of neurons per layer into a python list
        hidden_neurons = [hidden_neurons_layer1, hidden_neurons_layer2, hidden_neurons_layer3,
                                hidden_neurons_layer4, hidden_neurons_layer5]

        ################################################################################################################
        # 3) define training hyperparameters
        ################################################################################################################
        # training parameters
        batch_size = 200 #400
        num_epochs = 1000
        bias_stddev = 0.1  # or 0.01; 0.001  # bias initialization to zero reduce the exploding and vanishing gradient problem
        for weight_stddev in [0.5]:  # or Normal_Data = 0.01 or 0.001
            for optimizer in ["ADAM"]:  # RMSProp or ADAM
                # using Batch_nom requires a smaller learning rate
                for learning_rate in [0.00002]:  # Adam  0.0001 --> 0.0005
                    # decay_rate relates only to RMSProp Optimizer (standard is 0.9)
                    for decay_rate in [0.85]:  # 0.8 > Best < 09
                        # momentum relates only to RMSProp Optimizer (standard is 0.0)
                        for momentum in [0.8]:
                            ############################################################################################
                            # 4) define regularization hyperparameters
                            ############################################################################################
                            for dropout_rate in [0.85]:
                                for delta in [0.001]:  # 0.0001

                                    # save all selected parameters into a dict
                                    model_params = {'num_features': num_features, 'num_classes': num_classes,
                                                    'FFNN_layer': FFNN_layer, 'hidden_neurons':hidden_neurons,
                                                    'batch_size': batch_size, 'num_epochs': num_epochs,
                                                    'weight_stddev': weight_stddev, 'bias_stddev': bias_stddev,
                                                    'delta': delta, 'optimizer': optimizer,
                                                    'learning_rate': learning_rate, 'decay_rate': decay_rate,
                                                    'momentum': momentum, 'dropout_rate': dropout_rate}

                                    # create a notation/name containing all the selected parameters.
                                    # This allows a clear identification of the different training runs in
                                    # TensorBoard
                                    # hparam = SFN.make_hparam_string(model_params)

                                    ####################################################################################
                                    # 5) train and evaluate the recurrent neural network
                                    ####################################################################################

                                    # runs the model with the selected model_params
                                    train_model = model(model_params, x_train, y_train,
                                                        x_test, y_test)

                                    results.append(train_model)

    # this function compares the time measurements of two predictive models
    SFN.compare_computed_time(results)

    print('-----------------FINISH-----------------------')


if __name__ == '__main__':
    main()

