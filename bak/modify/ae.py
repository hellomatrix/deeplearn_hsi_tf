import tensorflow as tf
from utilites import tf_var2dic, xavier_init

from modify.rbm import RBM


def layer_with_saver(input_,
                     output_size,
                     bias_start=0.0,
                     activation_fn=None,
                     initializer=None,
                     name='layer'):
    shape = input_.get_shape().as_list()
    if initializer is None:
        initializer = tf.random_normal_initializer(stddev=0.02)
    if initializer == 'xavier':
        initializer = xavier_init(shape[1], output_size, activation_fn)
    with tf.variable_scope(name):
        w = tf.get_variable('W', dtype=tf.float32, initializer=initializer)

        b = tf.get_variable(
            'bias', [output_size],
            initializer=tf.constant_initializer(bias_start))

        out = tf.nn.bias_add(tf.matmul(input_, w), b)

    saver = tf.train.Saver({'w': w, 'b': b})

    if activation_fn is not None:
        return activation_fn(out), w, b, saver
    else:
        return out, w, b, saver


class AutoEncoder(object):
    def __init__(self,
                 input_size,
                 layer_sizes,
                 variable_scope='au',
                 reuse=False,
                 tied_weights=False,
                 optimizer=tf.train.AdamOptimizer(),
                 transfer_function=tf.nn.sigmoid):

        self.variable_scope = variable_scope
        self.tied_weights = tied_weights

        # Build the encoding layers
        self.x = tf.placeholder(tf.float32, [None, input_size])
        next_layer_input = self.x

        self.encoding_matrices = []
        self.encoding_biases = []
        self.encoding_saver = []
        self.transform_layers = [
            next_layer_input,
        ]
        self.rbm_net = []
        rbm_input_size = input_size
        with tf.variable_scope(variable_scope, reuse=reuse):
            for i, layer_s in enumerate(layer_sizes):
                name_i = 'L%d' % i
                next_layer_input, w, b, saver = \
                    layer_with_saver(next_layer_input,
                                     layer_s,
                                     activation_fn=transfer_function,
                                     initializer='xavier',
                                     name=name_i)
                self.encoding_matrices.append(w)
                self.encoding_biases.append(b)
                self.encoding_saver.append(saver)
                self.transform_layers.append(next_layer_input)
                #
                rbm_var_scope = '%s/%s' % (variable_scope, name_i)
                self.rbm_net.append(
                    RBM(rbm_input_size, layer_s, rbm_var_scope, 0.3))
                rbm_input_size = layer_s

        # The fully encoded x value is now stored in the next_layer_input
        self.encoded_x = next_layer_input

        # build the reconstruction layers by reversing the reductions
        layer_sizes.reverse()
        self.encoding_matrices.reverse()

        self.decoding_matrices = []
        self.decoding_biases = []

        for i, dim in enumerate(layer_sizes[1:] + [int(self.x.get_shape()[1])
                                                   ]):
            W = None
            # if we are using tied weights, so just lookup the encoding matrix for this step and transpose it
            if tied_weights:
                W = tf.identity(tf.transpose(self.encoding_matrices[i]))
            else:
                W = tf.Variable(
                    xavier_init(self.encoding_matrices[i].get_shape()[
                        1].value, self.encoding_matrices[i].get_shape()[0]
                                .value, transfer_function))
            b = tf.Variable(tf.zeros([dim]))
            self.decoding_matrices.append(W)
            self.decoding_biases.append(b)

            output = transfer_function(tf.matmul(next_layer_input, W) + b)
            next_layer_input = output

        # i need to reverse the encoding matrices back for loading weights
        self.encoding_matrices.reverse()
        self.decoding_matrices.reverse()

        # the fully encoded and reconstructed value of x is here:
        self.reconstructed_x = next_layer_input

        # compute cost
        self.cost = tf.sqrt(
            tf.reduce_mean(tf.square(self.x - self.reconstructed_x)))
        self.optimizer = optimizer.minimize(self.cost)

        # initalize variables
        # init = tf.initialize_all_variables()

    def transform(self, X, sess):
        return sess.run(self.encoded_x, {self.x: X})

    def transform_l(self, X, l, sess):
        return sess.run(self.transform_layers[l], {self.x: X})

    def reconstruct(self, X, sess):
        return sess.run(self.reconstructed_x, feed_dict={self.x: X})

    def load_rbm_weights(self, path, layer, sess):
        saver = self.encoding_saver[layer]
        saver.restore(sess, path)

        if not self.tied_weights:
            sess.run(self.decoding_matrices[layer].assign(
                tf.transpose(self.encoding_matrices[layer])))

    def print_weights(self, sess):
        print('Matrices')
        for i in range(len(self.encoding_matrices)):
            print('Matrice', i)
            print(self.encoding_matrices[i].eval(sess).shape)
            print(self.encoding_matrices[i].eval(sess))
            if not self.tied_weights:
                print(self.decoding_matrices[i].eval(sess).shape)
                print(self.decoding_matrices[i].eval(sess))

    def load_weights(self, path, sess):
        dict_w = self.get_dict_layer_names()
        saver = tf.train.Saver(dict_w)
        saver.restore(sess, path)

    def save_weights(self, path, sess):
        dict_w = self.get_dict_layer_names()
        saver = tf.train.Saver(dict_w)
        save_path = saver.save(sess, path)

    def get_dict_layer_names(self):
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     self.variable_scope)
        var_dict = tf_var2dic(var_list)
        return var_dict

    def partial_fit(self, X, sess):
        cost, opt = sess.run(
            (self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def KL(self, rho1, rho2):
        rho1 = tf.clip_by_value(rho1, 1e-5, (1-1e-5))
        rho2 = tf.clip_by_value(rho2, 1e-5, (1-1e-5))
        return rho1 * tf.log(rho1 / rho2) + (1. - rho1) * tf.log(
            (1. - rho1) / (1. - rho2))
