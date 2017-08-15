import tensorflow as tf
from Con_DL import contextual_dl
import scipy.io as sio
from tensorflow.python import debug as tf_debug
import numpy as np

hsi_file = './data/PaviaU.mat'
gnd_file = './data/PaviaU_gt.mat'
img = sio.loadmat(hsi_file)['paviaU']
gnd_img = sio.loadmat(gnd_file)['paviaU_gt']
img = img.astype(np.float32)
gnd_img = gnd_img.astype(np.int32)

cdl = contextual_dl(img, gnd_img, [60, 60, 60, 60])
train_data = cdl.train_data

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type='curses')
sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
cdl.load_AE_weights(sess)
cdl.train_writer.add_graph(sess.graph)

for i in range(100):
    for j in range(cdl.batch_length):
        X, y = cdl.train_data.next()
        loss, summary, _ = sess.run([cdl.loss, cdl.merged, cdl.train_step],
                                    {cdl.input: X,
                                     cdl.label: y})
        global_step = tf.train.global_step(sess, cdl.global_step_tensor)
        cdl.train_writer.add_summary(summary, global_step)
    valid_X, valid_y = cdl.batch_generate.valid_data()
    summary = sess.run(cdl.merged, {cdl.input: valid_X, cdl.label: valid_y})
    cdl.valid_writer.add_summary(summary, global_step)
