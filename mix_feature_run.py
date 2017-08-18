
import Config
import os.path
from Data import Data
import new_fully_connected_hsi_classfier as fc
import tensorflow as tf

dataname = Config.Salinas17817

trans_function = tf.nn.sigmoid

log = '/' + dataname + '/' + 'mix'

pre_need_train = True
final_need_train = True

pre_train_exist = Config.pretrain_model_dir+'/' + Config.dic_pre_train[dataname] + '/' + 'mix'
pre_train_model_dir = Config.pretrain_model_dir + log

if pre_need_train == False and not os.path.exists(pre_train_exist):
    print('no exist pre_train model')
    os._exit()

if pre_need_train == False and os.path.exists(pre_train_exist):
    pre_train_model_dir = pre_train_exist

elif pre_need_train == True and not os.path.exists(pre_train_model_dir):
    os.makedirs(pre_train_model_dir)


final_model_dir = Config.final_model_dir + log
if final_need_train == True and not os.path.exists(final_model_dir):
    os.makedirs(final_model_dir)

if final_need_train == False and not os.path.exists(final_model_dir):
    print('no exist pre_train model')
    os._exit()


pd = Data(data_name=dataname)
data_sets = pd.get_train_valid_test_of_mix_feature()

fc.run_trainning(data_sets, final_model_dir, pre_train_model_dir,
                 transfer_function=trans_function,pre_need_train=pre_need_train, final_need_train=final_need_train)