import tensorflow as tf
import pandas as pd
from sklearn import metrics
import Support_Fn as SFN
import time


def generate_FFNN_architecture(params, x_input, dropout, batch_norm):
    """

    :param params:
    :param x_input:
    :param dropout:
    :return:
    """

    hidden_neurons_list = params["hidden_neurons"]

    # initialize the input with x_input data
    input = x_input

    for layer in range(params["FFNN_layer"]):

        # the output layer uses no activation function and no dropout
        if layer < (params["FFNN_layer"] - 1):
            # hidden layer
            with tf.name_scope("fc_hidden_layer_".format(layer)):

                pre_act_output = tf.contrib.layers.fully_connected(input, hidden_neurons_list[layer],
                                                                  activation_fn=None)#, scope='pre_act_output')
                layer_norm = tf.contrib.layers.batch_norm(pre_act_output, center=True, scale=True,
                                                          is_training=batch_norm)#, scope='Layer_norm')
                act_output = tf.nn.relu(layer_norm, name="act_output")

                dropout_output = tf.nn.dropout(act_output, dropout, name="dropout_".format(layer))
                input = dropout_output
        else:
            # output layer
            with tf.name_scope("output_layer"):
                layer_output = tf.contrib.layers.fully_connected(input, hidden_neurons_list[layer],
                                                                  activation_fn=None)# , scope='pre_act_output')
    return layer_output


def model(params, x_train, y_train, x_test, y_test):
        '''
        The model () function consists of five areas:
        1) Definition of placeholders (variables will be defined in the FCNN-Function)
        2) Call the ANNs
        3) Implementation of the loss function, L2 regularization, performance metrics and the optimizer
        4) Initialization of the calculation graph and start of training and performance evaluation
        5) Visualization of the achieved performance
        :param params: python dictionary with all hyperparameters
        :param train_temp: ndarry of the trainings temp. sequences
        :param train_res: ndarray of the trainings resistances
        :param test_temp: ndarray of the test temp. sequences
        :param test_res: ndarray of the test resistances
        :return: Depending on the application, for example, cloud training (normally nothing)
        '''
        # delete the old graph
        tf.reset_default_graph()
        # initialize the computational graph
        sess = tf.Session()

        ################################################################################################################
        # 1) Definition of placeholders
        ################################################################################################################
        # declare the Placeholders
        x_input = tf.placeholder(tf.float32, [None, params["num_features"]], name="measured_amplitudes")
        y_output = tf.placeholder(tf.float32, [None, params["num_classes"]], name="milling_operation")
        dropout = tf.placeholder_with_default(1.0, shape=(), name="dropout")
        batch_norm = tf.placeholder_with_default(False, shape=(), name="batch_norm")

        ################################################################################################################
        # 2) Call the ANNs
        ################################################################################################################
        predictions = generate_FFNN_architecture(params, x_input, dropout, batch_norm)

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

                train_loss, train_acc = sess.run([loss, accuracy], feed_dict=train_dict)
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)

                test_loss, test_acc = sess.run([loss, accuracy], feed_dict=test_dict)
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
                print("Epoch%s: train loss = " % epoch, train_loss, ' val loss = ', test_loss)
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
        # plot the loss process over all epochs
        SFN.loss_process_plot(test_loss_list, train_loss_list)
        # plot the accuracy process over all epochs
        SFN.acc_process_plot(test_acc_list, train_acc_list)
        # plot the confusion matix
        labels = ["> 0.01 Ohm", ">= 0.01 Ohm"]
        SFN.plot_confusion_matrix(confusion_matrix, labels=labels)
        # results = {'plott_pred': [plott_pred_list], 'plott_true': [plott_true_list]}
        # params.update(results)
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
    :return:
    '''

    ####################################################################################################################
    # 1) load the data and determine their structure
    ####################################################################################################################
    # select the required Dataset
    # all_data = False --> contains only measurement attempt 42 and 44
    x_train, y_train, x_test, y_test = SFN.load_peak_train_and_test_data(all_data=False)
    result_df = pd.DataFrame()
    step = 0

    ####################################################################################################################
    # 2) define hyperparameters for FCNN-architecture
    ####################################################################################################################
    # Input Data
    # the input Data determine the shape of the temp sequences tensor
    num_features = len(x_train[1])
    num_classes = len(y_train[1])
    # FFNN Neural Network's internal structure
    for FFNN_layer in [4]:

        if FFNN_layer == 2: # possible accuracy up to 90%
            hidden_neurons_layer1 = 100
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

        hidden_neurons = [hidden_neurons_layer1, hidden_neurons_layer2, hidden_neurons_layer3,
                                hidden_neurons_layer4, hidden_neurons_layer5]

        ########################################################################################################
        # 3) define training hyperparameters
        ########################################################################################################
        batch_size = 20
        num_epochs = 3
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
                            ####################################################################################
                            # 4) define regularization hyperparameters
                            ####################################################################################
                            for dropout_rate in [0.8]:
                                for delta in [0.0001]:


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


                                    # --------------------------------------------
                                    # 3) train and evaluate the FFNN
                                    # -------------------------------------------

                                    # runs the model with the selected model_params
                                    train_model = model(model_params, x_train, y_train,
                                                        x_test, y_test)

    print('-----------------FINISH-----------------------')


if __name__ == '__main__':
    main()

