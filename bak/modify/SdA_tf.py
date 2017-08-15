import numpy as np

import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import dA_tf

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)


class SdA(object):

    def __init__(self,numpy_rng,n_ins,hidden_layers = [500,500],n_outs=10,corruption_levels=[0.1,0.1],train_set_x, batch_size):

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0


        for i in xrange(self.n_layers):# have n_layers hidden layers

            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            # Construct a denoising autoencoder that shared weights with this
            # layer


            dA_layer = dA_tf.Autoencoder(n_input = layer_input, n_hidden = hidden_layers_sizes[i],
                             transfer_function=tf.nn.sigmoid(), optimizer = tf.train.AdamOptimizer(learning_rate = 0.001))
            dA_layer._initialize_weights()


            # trainning




            pretraining_functions(x,y)

            self.sigmoid_layers.append(sigmoid_layer)
 

    def pretraining_functions(self, train_set_x, batch_size):

        def standard_scale(X_train, X_test):
            preprocessor = prep.StandardScaler().fit(X_train)
            X_train = preprocessor.transform(X_train)
            X_test = preprocessor.transform(X_test)
            return X_train, X_test

        def get_random_block_from_data(data, batch_size):
            start_index = np.random.randint(0, len(data) - batch_size)
            return data[start_index:(start_index + batch_size)]

        X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

        n_samples = int(mnist.train.num_examples)
        training_epochs = 20
        batch_size = 128
        display_step = 1


        for dA in self.dA_layers:
            for epoch in range(training_epochs):
                    avg_cost = 0.
                total_batch = int(n_samples / batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_xs = get_random_block_from_data(X_train, batch_size)

                    # Fit training using batch data
                    cost = self.dA_layer.partial_fit(batch_xs)
                    # Compute average loss
                    avg_cost += cost / n_samples * batch_size

                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

            print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))







        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(inputs=[index,
                              theano.Param(corruption_level, default=0.2),
                              theano.Param(learning_rate, default=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x: train_set_x[batch_begin:
                                                             batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns



















