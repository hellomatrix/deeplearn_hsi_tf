

import tensorflow as tf
from Con_DL import contextual_dl
import scipy.io as sio
import numpy as np
import pdb

flags = tf.flags
flags.DEFINE_string('save_path', 'data', 'path for saved data')
FLAGS = flags.FLAGS


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


hsi_file = '/home/zgh/project/data/Pavia/PaviaU.mat'
gnd_file = '/home/zgh/project/data/Pavia/PaviaU_gt.mat'
img = sio.loadmat(hsi_file)['paviaU']
gnd_img = sio.loadmat(gnd_file)['paviaU_gt']

img = img.astype(np.float32)
gnd_img = gnd_img.astype(np.int32)
config_ = config()


def main(_):
    cdl = contextual_dl(img, gnd_img, config_)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        cdl.load_AE_weights(sess)
        cdl.train_writer.add_graph(sess.graph)

        for i in range(10):
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


if __name__ == '__main__':
    tf.app.run()
