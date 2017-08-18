
import Config
import os.path
from Data import Data
import new_fully_connected_hsi_classfier as fc
import tensorflow as tf
import svm_method

dataname = 'liuyang_data'

trans_function = tf.nn.sigmoid

log = '/' + dataname + '/' + 'spectral'

pre_need_train = True
final_need_train = True

final_model_dir = Config.final_model_dir + log
if not os.path.exists(final_model_dir):
    os.makedirs(final_model_dir)

pre_train_model_dir = Config.pretrain_model_dir + log
if not os.path.exists(pre_train_model_dir):
    os.makedirs(pre_train_model_dir)

data_sets = svm_method.data_sets

fc.run_trainning(data_sets, final_model_dir, pre_train_model_dir,
                 transfer_function=trans_function,pre_need_train=pre_need_train, final_need_train=final_need_train)