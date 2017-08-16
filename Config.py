#class Config(object):

encoder_layers = [60, 60, 60, 60]

# epoch_pretrain_times = 60 # for others
epoch_ae_pretrain_times = 500 # for ksc only
# SAE_train_times =100 # for others
epoch_sae_pretrain_times =10000 # for ksc only
# epoch_train_times = 10000  # for others
epoch_final_train_times = 100000 # for ksc only

#au_epoch = 10
#tied_weights = True
batch_size = 100

learn_rate = 0.01
# final_learn_rate = 0.01
final_learn_rate = 0.0001
set_ratio = [6, 2, 2]

max_steps = 100000

data_path= '../hsi_data'

# hsi_file = '../hsi_data/Pavia/PaviaU.mat'
# gnd_file = '../hsi_data/Pavia/PaviaU_gt.mat'
# PaviaU = 'PaviaU'

# hsi_file1 = '../hsi_data/KSC/KSC.mat'
# gnd_file1 = '../hsi_data/KSC/KSC_gt.mat'

ksc = 'KSC'
#ksc_class_num = 16

paviaU = 'paviaU'
#paviaU_class_num = 10

Salinas = 'Salinas'

random_state = 25535

pretrain_ckpt = './pretrain_SAE_ckpt/'
final_ckpt_dir = './final_model_ckpt/'