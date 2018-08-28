import argparse
import numpy as np
import tensorflow as tf
import _pickle as pickle
import math
import time

class Dataset:

    def __init__(self, num_workers, myid, batch_size):
        ''' Load training, testing and validation dataset '''
    
        '''Training data'''
#        print('Loading Training dataset ...')
        inp = pickle.load(open("small_train.p", "rb"), encoding='latin1')
        targ = pickle.load(open("small_train_sen.p", "rb"), encoding='latin1')
#        print('Training dataset Loaded')
    
        '''Divide training data'''
        inpSize = len(inp) // num_workers
        inpOffset = inpSize * myid
        if (myid < (len(inp) % num_workers)):
            inpSize = inpSize + 1
            inpOffset = inpOffset + myid
        else:
            inpOffset = inpOffset + (len(inp) % num_workers)
#        print('!!!!!!', inpSize, '!!!!!!!')
#        print('!!!!!!', inpOffset, '!!!!!!!')
        inp = inp[inpOffset:(inpOffset+inpSize)]
    
        targSize = len(targ) // num_workers
        targOffset = targSize * myid
        if (myid < (len(targ) % num_workers)):
            targSize = targSize + 1
            targOffset = targOffset + myid
        else:
            targOffset = targOffset + (len(targ) % num_workers)
        targ = targ[targOffset:(targOffset+targSize)]
    
        '''Shuffle input'''
#        print('Shuffling input ...')
        x = np.c_[inp.reshape(len(inp), -1), targ.reshape(len(targ), -1)]
        shuffle_inp = x[:, :inp.size // len(inp)].reshape(inp.shape)
        shuffle_targ = x[:, inp.size // len(inp):].reshape(targ.shape)
        np.random.shuffle(x)
        np.random.shuffle(x)
        inp = shuffle_inp
        targ = shuffle_targ
#        print('Input shuffled')
    
        '''Testing data'''
#        print('Loading Testing dataset ....')
        test_inp = pickle.load(open("small_test.p", "rb"), encoding='latin1')
        test_targ = pickle.load(open("small_test_sen.p", "rb"), encoding='latin1')
#        print('Testing dataset loaded')
    
        '''Validation data'''
#        print('Loading Validation dataset ...')
        valid_inp = inp[200000:]
        valid_targ = targ[200000:]
        inp = inp[:200000]
        targ = targ[:200000]
#        print('Validation dataset loaded')
    
        '''Organize into tuples'''
        self.train = (inp, targ)
        self.test = (test_inp, test_targ)
        self.valid = (valid_inp, valid_targ)
        # print('Training set', train[0].shape, train[1].shape)
        # print('Validation set', valid[0].shape, valid[1].shape)
        # print('Test set', test[0].shape, test[1].shape)
        self.n_batches = self.train[0].shape[0] // batch_size
        self.mb_idx = 0
        self.batch_size = batch_size
    def get_data(self):
        return self.train, self.test, self.valid
    def get_next_batch(self):
        train_X = self.train[0]
        train_Y = self.train[1]
        batch_X = train_X[self.mb_idx * self.batch_size:(self.mb_idx + 1) * batch_size]
        batch_Y = train_Y[self.mb_idx * self.batch_size:(self.mb_idx + 1) * batch_size]
        self.mb_idx = (self.mb_idx + 1) % self.n_batches
        return batch_X, batch_Y

class DeepNeuralNetwork:

    def __init__(self, n_in, n_out, test, valid, hiddenLayers, activation=tf.nn.sigmoid, batch_size=128, learning_rate=0.01):
        ''' Deep Neural Network implementation '''
        self.graph = tf.Graph()
        with self.graph.as_default():

            '''Training dataset, given in mini-batches '''
            self.tf_train = (tf.placeholder(tf.float32, shape=(
                batch_size, n_in)), tf.placeholder(tf.float32, shape=(batch_size, n_out)))

            ''' Validation dataset '''
            tf_valid = (tf.cast(tf.constant(valid[0]), tf.float32), tf.cast(
                tf.constant(valid[1]), tf.float32))

            '''Testing dataset'''
            tf_test = (tf.cast(tf.constant(test[0]), tf.float32), tf.cast(
                tf.constant(test[1]), tf.float32))

            self.weights = []  # Weights list
            self.bias = []  # Bias list
            self.l2_reg = 0.  # L2 Regularization

            '''Inputs'''
            train_input = self.tf_train[0]
            valid_input = tf_valid[0]
            test_input = tf_test[0]

            layerIn = n_in  # input to layer
            ''' Add hidden layers '''
            for layerOut, hdf in hiddenLayers:
                train_input = self._addLayer(
                    train_input, layerIn, layerOut, activation=activation, dropout=hdf, l2_reg=True)
                valid_input = self._addLayer(
                    valid_input, layerIn, layerOut, activation=activation, weights=self.weights[-1], bias=self.bias[-1])
                test_input = self._addLayer(
                    test_input, layerIn, layerOut, activation=activation, weights=self.weights[-1], bias=self.bias[-1])
                ''' Input to next layer is output of current layer '''
                layerIn = layerOut

            '''Output layers '''
            train_logits = self._addLayer(train_input, layerIn, n_out)
            valid_logits = self._addLayer(
                valid_input, layerIn, n_out, weights=self.weights[-1], bias=self.bias[-1])
            test_logits = self._addLayer(
                test_input, layerIn, n_out, weights=self.weights[-1], bias=self.bias[-1])

            '''Cost function '''
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=train_logits, labels=self.tf_train[1])) + 0.0001 * self.l2_reg

            ''' Adagrad Optimizer '''
            self.global_step = tf.train.get_or_create_global_step()
            self.optimizer = tf.train.AdagradOptimizer(
                learning_rate).minimize(self.cost, global_step = self.global_step)

            ''' Prediction functions '''
            self.train_pred = tf.nn.softmax(train_logits)
            self.valid_pred = tf.nn.softmax(valid_logits)
            self.test_pred = tf.nn.softmax(test_logits)

    def _addLayer(self, input, n_in, n_out, activation=None, weights=None, bias=None, dropout=None, l2_reg=False):
        if(weights is None):
            ''' Xavier init '''
            init_range = math.sqrt(6.0 / (n_in + n_out))
            init_w = tf.random_uniform([n_in, n_out], -init_range, init_range)
            weights = tf.cast(tf.Variable(init_w), tf.float32)
            self.weights.append(weights)
        if(bias is None):
            bias = tf.cast(tf.Variable(tf.zeros([n_out])), tf.float32)
            self.bias.append(bias)
        if(l2_reg):
            ''' L2 regularization '''
            l2_reg = tf.nn.l2_loss(weights)
            self.l2_reg += l2_reg

        ''' Affine transformation '''
        layer = tf.matmul(input, weights) + bias

        if(activation is not None):
            layer = activation(layer)
        if(dropout is not None):
            ''' Dropout + scaling '''
            layer = tf.nn.dropout(layer, 1 - dropout) * 1 / (1 - dropout)
        return layer


def accuracy(pred, labels):
    return (100.0 * np.sum(np.argmax(pred, 1) == np.argmax(labels, 1)) / pred.shape[0])

if __name__ == '__main__':
    ''' Parse host lists '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="grpc",
        help="Communication protocol"
    )
    FLAGS, unparsed = parser.parse_known_args()
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.33)
    config = tf.ConfigProto(gpu_options = gpu_options)
    server = tf.train.Server(cluster,
                             job_name = FLAGS.job_name,
                             task_index = FLAGS.task_index,
                             config = config,
                             protocol = FLAGS.protocol)
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        ''' Params '''
        n_epochs = 600
        batch_size = 128
        learning_rate = 0.01
        num_layers = 3
        hiddenLayers = [(1024, 0.5)] * num_layers
        activation = tf.nn.tanh

        ''' Dataset '''
        dataset = Dataset(len(worker_hosts),FLAGS.task_index,batch_size)
        train, valid, test = dataset.get_data()
        train_X = train[0]
        train_Y = train[1]

        n_in = train_X.shape[1]
        n_out = train_Y.shape[1]

        ''' Model '''
        with tf.device(tf.train.replica_device_setter(
            worker_device = "/job:worker/task:%d" % FLAGS.task_index,
            cluster = cluster)):
                dnn = DeepNeuralNetwork(n_in, n_out, test, valid, hiddenLayers,
                                        activation=activation, batch_size=batch_size, learning_rate=learning_rate)

                hooks = [tf.train.StopAtStepHook(last_step = n_epochs * dataset.n_batches * len(worker_hosts))]
                with dnn.graph.as_default():
                    with tf.train.MonitoredTrainingSession(master = server.target,
                                                     is_chief = (FLAGS.task_index == 0),
                                                     hooks = hooks) as session:
#                        tf.initialize_all_variables().run()
                        loop = 0
                        start = time.time()
                        print("time, training accuracy, validation accuracy")
                        while not session.should_stop():
                            avg_cost = 0.
                            avg_acc = 0.
                            valid_acc = []

                            ''' Mini-batching '''
                            batch_X, batch_Y = dataset.get_next_batch()
                            ''' Input to placeholders '''
                            feed_dict = {dnn.tf_train[0]
                                : batch_X, dnn.tf_train[1]: batch_Y}

                            ''' Train step '''
                            _, cost, train_pred = session.run(
                                [dnn.optimizer, dnn.cost, dnn.train_pred], feed_dict=feed_dict)

                            acc = accuracy(train_pred, batch_Y)

                            if(loop % dataset.n_batches == 0):
                                ep = loop // dataset.n_batches
                                elasped = time.time() - start
#                                print("Cost at {} - {}".format(ep, cost))
#                                print("Training accuracy : {}".format(acc))
                                val_acc = accuracy(dnn.valid_pred.eval(session = session), valid[1])
#                                print("Validation accuracy : {}".format(val_acc))
                                print("%f %f %f" % (elasped, acc, val_acc))
                                valid_acc.append(val_acc)
                                # TODO add patience for early stopping
                                if(len(valid_acc) > 3 and int(valid_acc[-1] * 100) - int(valid_acc[-3] * 100) > -1):
                                    break
                            loop = loop + 1

#                        ''' Testing '''
#                        print("Test accuracy : {}".format(
#                            accuracy(dnn.test_pred.eval(), test[1])))
