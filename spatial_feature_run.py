
import Config
import os.path
from Data import Data
import new_fully_connected_hsi_classfier as fc
import tensorflow as tf

dataname = Config.Salinas

trans_function = tf.nn.sigmoid

log = '/' + dataname + '/' + 'spatial'

pre_need_train = False
final_need_train = False

final_model_dir = Config.final_model_dir + log
if not os.path.exists(final_model_dir):
    os.makedirs(final_model_dir)

pre_train_model_dir = Config.pretrain_model_dir + log
if not os.path.exists(pre_train_model_dir):
    os.makedirs(pre_train_model_dir)

pd = Data(data_name=dataname)
data_sets = pd.get_train_valid_test_of_spatial_feature()

fc.run_trainning(data_sets, final_model_dir, pre_train_model_dir,
                 transfer_function=trans_function,pre_need_train=pre_need_train, final_need_train=final_need_train)


