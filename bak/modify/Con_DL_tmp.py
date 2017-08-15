import os

import numpy as np
import tensorflow as tf
from read_data import get_MLdata_batch

from modify.au_tmp import AutoEncoder
from modify.utilites import makeGaussian, conv2d

path_base = os.path.dirname(os.path.abspath(__file__))


class config(object):
    hidden_layers = [60, 60, 60, 60]
    epoch_size = 10000
    userbm = True
    rbm_epoch = 10
    au_epoch = 10
    tied_weights = True
    batch_size = 50
    window = 7
    learn_rate = 0.01
    ratio = [6, 2, 2]


class contextual_dl(object):
    def __init__(self, hsi_img, gnd_img, config):
        '''
        '''
        self.hsi_img = hsi_img
        self.gnd_img = gnd_img
        self.ratio = config.ratio
        self.window = config.window
        self.label_size = gnd_img.max()
        self.layer_sizes = config.hidden_layers
        self.data_path = path_base + '/data'
        self.rbm_epoch = config.rbm_epoch
        self.au_epoch = config.au_epoch
        self.learn_rate = config.learn_rate
        self.userbm = config.userbm
        self.batch_size = config.batch_size

        # build graph
        self.fine_tuning()

    def input_batch(self):
        batch_generate = get_MLdata_batch(self.hsi_img, self.gnd_img,
                                          self.batch_size, self.ratio,
                                          self.window)
        self.batch_generate = batch_generate
        train_data = batch_generate.train_data()
        self.batch_length = batch_generate.batch_length
        self.dim = dim = self.hsi_img.shape[2]
        double_threshold = self.window - 1
        input_window = 2 * double_threshold + 1
        self.input_window = input_window
        batch_shape = (None, input_window, input_window, dim)
        self.train_data = train_data
        return batch_shape

    def extract_spatial_information(self):
        batch_shape = self.input_batch()
        with tf.name_scope('placeholder'):
            self.label = tf.placeholder(tf.int32, shape=[None], name='label')
            self.input = tf.placeholder(
                tf.float32, shape=batch_shape, name='input_X')

        filter1_init = np.array(makeGaussian(self.window), dtype=np.float32)
        with tf.name_scope('extract_spatial_info'):
            filter1 = tf.get_variable(
                'filter1', dtype=tf.float32, initializer=filter1_init)
            self.filter1 = filter1

            # shape is [batch_size, dim, input_window, input_window]
            filter_input = tf.transpose(self.input, [0, 3, 1, 2], 'transpose')
            filter_input = tf.reshape(filter_input, (-1, self.input_window,
                                                     self.input_window, 1))

            spatial_merged = conv2d(filter_input, filter1)
            spatial_merged = tf.reshape(spatial_merged, (
                -1, self.dim, self.window, self.window))
            # [batch_size, window, window, dim]
            spatial_merged = tf.transpose(spatial_merged, [0, 2, 3, 1],
                                          'transBac')
        return spatial_merged

    def spectral_feature_mining(self):
        spatial_merged = self.extract_spatial_information()
        with tf.name_scope('spectral_feature_mining'):
            spatial_merged = tf.reshape(spatial_merged, (-1, self.dim))
            self.spatial_merged = spatial_merged
            input_layer = spatial_merged
            # Encoder layers
            layer_name = 'au_shared'
            l2_weights = []
            l2_regularizer = tf.contrib.layers.l2_regularizer(0.5, layer_name)
            with tf.name_scope('AU_layers'):
                with tf.variable_scope(layer_name):
                    for i, lay_size in enumerate(self.layer_sizes):
                        input_layer, w, _ = ord_layer(
                            input_layer,
                            lay_size,
                            activation_fn=tf.nn.sigmoid,
                            name='L%d' % i)
                        l2_weights.append(l2_regularizer(w))
                input_size = spatial_merged.get_shape().as_list()[-1]
                self.l2_weights = l2_weights
            with tf.name_scope('AE_pretrain'):
                self.au_net = AutoEncoder(
                    input_size,
                    self.layer_sizes,
                    variable_scope=layer_name,
                    reuse=True,
                    tied_weights=True)

            # shape is [batch_size, window, window, hidden_layer_size]
            output_layer = tf.reshape(input_layer, (
                -1, self.window, self.window, self.layer_sizes[-1]))

        return output_layer

    def feature_smooth(self):

        filter2_init = np.array(makeGaussian(self.window), dtype=np.float32)

        with tf.name_scope('feature_smooth'):
            filter2 = tf.get_variable(
                'filter2', dtype=tf.float32, initializer=filter2_init)
            self.filter2 = filter2
            # shape should be [batch_size, hidden_layer_size, window, window]
            spectraled_data = self.spectral_feature_mining()
            spectraled_data = tf.transpose(spectraled_data, [0, 3, 1, 2],
                                           'trans2')
            spectraled_data = tf.reshape(spectraled_data,
                                         [-1, self.window, self.window, 1])

            smoothed_data = conv2d(spectraled_data, filter2)
            smoothed_data = tf.reshape(smoothed_data,
                                       [-1, self.layer_sizes[-1]])
        return smoothed_data

    def fine_tuning(self):
        smoothed_data = self.feature_smooth()
        with tf.name_scope('fine_tuning'):
            with tf.name_scope('softmax'):
                logits, _, _ = ord_layer(
                    smoothed_data, self.label_size, name='logits')
            label_one_hot = tf.one_hot(self.label, self.label_size)
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=label_one_hot, logits=logits))
            loss = cross_entropy
            # L2 regularization
            for l2_loss in self.l2_weights:
                loss = loss + l2_loss
            self.loss = loss
            # train step
            self.global_step_tensor = tf.Variable(
                0, trainable=False, name='global_step')
            optimizer = tf.train.GradientDescentOptimizer(self.learn_rate)
            train_step = optimizer.minimize(
                loss, global_step=self.global_step_tensor)
            self.train_step = train_step
        #
        with tf.name_scope('accuracy'):
            predicted_label = \
                tf.argmax(tf.nn.softmax(logits, name='softmax'), 1)
            self.predicted_label = predicted_label = tf.cast(
                predicted_label, tf.int32)
            correct_prediction = tf.equal(predicted_label, self.label)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.accuracy = accuracy

        # save model
        with tf.name_scope('save_model'):
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            self.model_save = tf.train.Saver(var_list)

        # summary
        with tf.name_scope('summary_all'):
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', accuracy)
            filter1 = tf.reshape(self.filter1,
                                 [1, self.window, self.window, 1])
            filter2 = tf.reshape(self.filter2,
                                 [1, self.window, self.window, 1])
            tf.summary.image('filter1', filter1)
            tf.summary.image('filter2', filter2)
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(self.data_path +
                                                      '/train')
            self.test_writer = tf.summary.FileWriter(self.data_path + '/test')
            self.valid_writer = tf.summary.FileWriter(self.data_path +
                                                      '/valid')

    def save_model(self, sess):
        self.model_save.save(sess, self.data_path + '/model')

    def pre_train_rbm(self, sess):
        epoch = self.rbm_epoch
        for i, rbm in enumerate(self.au_net.rbm_net):
            print('pre_train rbm layer %d' % i)
            for _ in range(epoch):
                saved_batch = []
                for _ in range(self.batch_length):
                    batch_xs = sess.run(self.spatial_merged, {
                        self.input: self.train_data.next()[0]
                    })
                    batch_xs = self.au_net.transform_l(batch_xs, i, sess)
                    saved_batch.append(batch_xs)
                    rbm.partial_fit(batch_xs)
                print(rbm.compute_cost(np.vstack(saved_batch)))
            rbm.save_weights(self.data_path + '/rbmw%d.chp' % i)

    def pre_train_au(self, sess):
        if self.userbm:
            # check saved data
            rbm_saved = True
            layer_num = len(self.layer_sizes)
            for i in range(layer_num):
                filename = self.data_path + '/rbmw%d.chp.meta' % i
                if not os.path.isfile(filename):
                    rbm_saved = False
                    break
                else:
                    continue

            if not rbm_saved:
                print('training rbm ...')
                self.pre_train_rbm(sess)

            for i, _ in enumerate(self.au_net.rbm_net):
                self.au_net.load_rbm_weights(
                    self.data_path + '/rbmw%d.chp' % i, i, sess)
        epoch = self.au_epoch
        for i in range(epoch):
            cost = 0.0
            for j in range(self.batch_length):
                batch_xs = sess.run(self.spatial_merged,
                                    {self.input: self.train_data.next()[0]})
                cost += self.au_net.partial_fit(batch_xs, sess)
            print(cost)
        self.au_net.save_weights(self.data_path + '/au.chp', sess)

    def load_AE_weights(self, sess):
        filename = self.data_path + '/au.chp.meta'
        if not os.path.isfile(filename):
            self.pre_train_au(sess)
        self.au_net.load_weights(self.data_path + '/au.chp', sess)


def ord_layer(input_,
              output_size,
              stddev=0.02,
              bias_start=0.0,
              activation_fn=None,
              name='layer'):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        w = tf.get_variable(
            'W', [shape[1], output_size],
            tf.float32,
            tf.random_normal_initializer(stddev=stddev))
        # variable_summaries(w)
        b = tf.get_variable(
            'bias', [output_size],
            initializer=tf.constant_initializer(bias_start))
        # variable_summaries(b)

        out = tf.nn.bias_add(tf.matmul(input_, w), b)

        if activation_fn is not None:
            return activation_fn(out), w, b
        else:
            return out, w, b


def variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


if __name__ == '__main__':
    import scipy.io as sio
    hsi_file = './data/PaviaU.mat'
    gnd_file = './data/PaviaU_gt.mat'
    img = sio.loadmat(hsi_file)['paviaU']
    gnd_img = sio.loadmat(gnd_file)['paviaU_gt']
    img = img.astype(np.float32)
    gnd_img = gnd_img.astype(np.int32)

    config_ = config()
    cdl = contextual_dl(img, gnd_img, config_)
    train_data = cdl.train_data

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        cdl.load_AE_weights(sess)
        cdl.train_writer.add_graph(sess.graph)

        for i in range(10000):
            for j in range(cdl.batch_length):
                X, y = cdl.train_data.next()
                loss, summary, _ = sess.run(
                    [cdl.loss, cdl.merged,
                     cdl.train_step], {cdl.input: X,
                                       cdl.label: y})
                global_step = tf.train.global_step(sess,
                                                   cdl.global_step_tensor)
                cdl.train_writer.add_summary(summary, global_step)
            valid_X, valid_y = cdl.batch_generate.valid_data()
            summary = sess.run(cdl.merged,
                               {cdl.input: valid_X,
                                cdl.label: valid_y})
            cdl.valid_writer.add_summary(summary, global_step)
            cdl.save_model(sess)

            # print loss
            # print('global step: %s' % tf.train.global_step(
            #     sess, cdl.global_step_tensor))
