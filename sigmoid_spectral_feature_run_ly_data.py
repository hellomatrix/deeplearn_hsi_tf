
import Config
import os.path
from Data import Data
import new_fully_connected_hsi_classfier as fc
import tensorflow as tf

# dataname = 'Salinas_origin'
# dataname = Config.Salinas
dataname = Config.Salinas17817

trans_function = tf.nn.sigmoid

log = '/sigmoid/block_sampling_ly/' + dataname + '/' + 'spectral'

pre_need_train = True
final_need_train = True

final_model_dir = Config.final_model_dir + log
if not os.path.exists(final_model_dir):
    os.makedirs(final_model_dir)

pre_train_model_dir = Config.pretrain_model_dir + log
if not os.path.exists(pre_train_model_dir):
    os.makedirs(pre_train_model_dir)

pd = Data(data_name=dataname)
data_all = pd.get_block_sampling_ly_all()


data_sets =  data_all[0]

fc.run_trainning(data_sets, final_model_dir, pre_train_model_dir,
                 transfer_function=trans_function,pre_need_train=pre_need_train, final_need_train=final_need_train)