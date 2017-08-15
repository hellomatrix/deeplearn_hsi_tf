

# import numpy as np
# import tensorflow as tf



# class dA_tf(Object):

#     def __init__ (
#             self,
#             numpy_rng,
#             input=None,
#             n_visible = 784,
#             n_hidden = 500,
#             W = None,
#             bhind = None,
#             bvis = None
#     ):


#         self.n_visible=n_visable
#         self.n_hidden = n_hidden

#         if not W:

#             initial_W = numpy.asarray(
#                 numpy_rng.uniform(
#                     low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
#                     high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
#                     size=(n_visible, n_hidden)
#                 ),
#                 dtype = tf.float32
#             )



#             initial_W = numpy.asarray(
#                 numpy_rng.uniform(
#                     low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
#                     high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
#                     size=(n_visible, n_hidden)
#                 ),
#                 dtype=theano.config.floatX
#             )

#             self.W = W
#             self.b = bhind
#             self.b_prime = bvis
#             self.W_prime = self.W.T

#             self.params = [self.W,self.b,self.b_prime]


import tensorflow as tf

class Autoencoder(object):

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer = tf.train.AdamOptimizer()):

        self.n_input = n_input # number of input
        self.n_hidden = n_hidden # number of hidden
        self.transfer = transfer_function # active function

        network_weights = self._initialize_weights() # init a one layer AE and return weights
        self.weights = network_weights # weights

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input]) # init a input vec memory 
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1'])) # the output of hidden with nonlinear function
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2']) # the output of AE reconstruction

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0)) # the MSE
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer() # init the variables 
        self.sess = tf.Session()
        self.sess.run(init) # init the vars at very beginning 


        # init paras memory  
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.get_variable("w1", shape=[self.n_input, self.n_hidden],
            initializer=tf.contrib.layers.xavier_initializer()) # why put here
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32)) # output size equals to input size 
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

     # all the code will run after sess.run
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X}) # all the code will run after sess.run
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X})

    def transform(self, X): # cal hidden layer
        return self.sess.run(self.hidden, feed_dict={self.x: X})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X}) # return functions, which need feed_dict 

    def getWeights(self):
        return self.sess.run(self.weights['w1']) # return vars

    def getBiases(self):
        return self.sess.run(self.weights['b1'])



