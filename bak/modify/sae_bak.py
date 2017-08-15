import os

import numpy as np
import tensorflow as tf

from data import PrepareData
from modify.au_tmp import AutoEncoder

path_base = '.'

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


class SAE(object):


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
        self.dim = hsi_img.shape[-1]

        #self.train_data=self.train_data_()

        self.input_ = tf.placeholder(tf.float32, shape=[None, self.dim],
                               name='input')
        self.label_one_hot = tf.placeholder(tf.float32, shape=[None, self.label_size],
                               name='label')
        # build graph
        self.fine_tuning()


    def encoder_constructing(self):

        with tf.name_scope('spectral_feature_mining'):
            input_ = self.input_
            # Encoder layers
            layer_name = 'AE_shared'

            with tf.name_scope('AU_layers'):
                with tf.variable_scope(layer_name):

                    for i, layer_size in enumerate(self.layer_sizes):
                        output_, w, _ = single_layer(
                            input_,
                            layer_size,
                            activation_fn=tf.nn.sigmoid,
                            name = 'L%d' % i) #from 0

                        input_ = output_

            output_layer = output_

            with tf.name_scope('AE_pretrain'):

                nets_input = self.dim
                self.au_net = AutoEncoder(
                    nets_input,
                    self.layer_sizes,
                    variable_scope=layer_name,
                    reuse=True,
                    tied_weights=True)

        return output_layer

    def fine_tuning(self):

        spectral_feature = self.encoder_constructing()

        with tf.name_scope('fine_tuning'):
            with tf.name_scope('softmax'):
                logits, _, _ = single_layer(
                    spectral_feature, self.label_one_hot.shape[1], name='logits')

                #label_one_hot = train_data[1] #???

                cross_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.label_one_hot, logits=logits))

            loss = cross_entropy

            self.loss = loss
            # train step
            self.global_step_tensor = tf.Variable(
                0, trainable=False, name='global_step')

            optimizer = tf.train.AdamOptimizer()

            train_step = optimizer.minimize(
                loss, global_step=self.global_step_tensor)

            self.train_step = train_step
        #
        with tf.name_scope('accuracy'):

            predicted_label = \
                tf.argmax(tf.nn.softmax(logits, name='softmax'), 1)

            gnd_label= tf.argmax(self.label_one_hot, 1)

   #          self.predicted_label = predicted_label = tf.cast(
    #            predicted_label, tf.float32)
            correct_prediction = tf.equal(predicted_label, gnd_label)
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
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(self.data_path +
                                                      '/train')
            self.test_writer = tf.summary.FileWriter(self.data_path + '/test')
            self.valid_writer = tf.summary.FileWriter(self.data_path +
                                                      '/valid')

    def save_model(self, sess):
        self.model_save.save(sess, self.data_path + '/model')

    def pre_train_rbm(self, sess, train_data):

        for i, rbm in enumerate(self.au_net.rbm_net):
            print('pre_train rbm layer %d' % i)
            for _ in range(20):

                saved_batch = []

                batch_xs = train_data[0]
                batch_xs = self.au_net.transform_l(batch_xs, i, sess)
                saved_batch.append(batch_xs)
                rbm.partial_fit(batch_xs)

                print('rmb cost %d' % rbm.compute_cost(np.vstack(saved_batch)))
            rbm.save_weights(self.data_path + '/rbmw%d.chp' % i)

    def pre_train_au(self, sess, train_data):

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
                self.pre_train_rbm(sess,train_data)

            for i, _ in enumerate(self.au_net.rbm_net):
                self.au_net.load_rbm_weights(
                    self.data_path + '/rbmw%d.chp' % i, i, sess)
        #epoch = self.au_epoch
        for i in range(10):
            cost = 0.0
            for j in range(10):
                batch_xs = train_data[0]
                cost += self.au_net.partial_fit(batch_xs, sess)
            print('au_net cost %d'%cost)
        self.au_net.save_weights(self.data_path + '/au.chp', sess)


    def load_AE_weights(self, sess, train_data):
        filename = self.data_path + '/au.chp.meta'
        if not os.path.isfile(filename):
            self.pre_train_au(sess, train_data)

            print(self.data_path + '/au.chp')

        self.au_net.load_weights(self.data_path + '/au.chp', sess)

def single_layer(input_,
              layer_size,
              stddev=0.02,
              bias_start=0.0,
              activation_fn=None,
              name='layer'):

    output_size = layer_size
    shape = input_.get_shape().as_list() # good for fetch data

    with tf.variable_scope(name):
        w = tf.get_variable(
            'W', [shape[1], output_size],
            tf.float32,
            tf.random_normal_initializer(stddev=stddev))

        # variable_summaries(w)
        b = tf.get_variable(
            'bias', [output_size],
            tf.float32,
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
    hsi_file = '．．/hsi_data/Pavia/PaviaU.mat'
    gnd_file = '../hsi_data/Pavia/PaviaU_gt.mat'

    img = sio.loadmat(hsi_file)['paviaU']
    gnd_img = sio.loadmat(gnd_file)['paviaU_gt']
    img = img.astype(np.float32)
    gnd_img = gnd_img.astype(np.int32)

    # prepare data
    pd = PrepareData(img, gnd_img)
    train_data = pd.train_data

    # prepare nets
    config_ = config()
    cdl = SAE(img, gnd_img, config_)

    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())
    # x, y = train_data[0],train_data[1]
    # sess.run(cdl.input_layer, {cdl.input_: x})

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        cdl.load_AE_weights(sess,train_data)

        x, y = train_data[0],train_data[1]
 
        for i in range(100):
            loss, _ = sess.run(
                [cdl.loss, cdl.train_step],
                {cdl.input_: x,
                 cdl.label: y})
            print(loss)

            # print loss
            # print('global step: %s' % tf.train.global_step(
            #     sess, cdl.global_step_tensor))