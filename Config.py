#class Config(object):

encoder_layers = [60, 60, 60, 60]


epoch_ae_pretrain_times = 150 # for others
epoch_sae_pretrain_times =300# for others
epoch_final_train_times = 10000 # for others


# #for test
# test_ae_pretrain_times = 5 # for ksc only
# # SAE_train_times =100 # for others
# test_sae_pretrain_times =20# for ksc only
# # epoch_train_times = 10000  # for others
# test_final_train_times = 50 # for ksc only


#au_epoch = 10
#tied_weights = True
batch_size = 200
learn_rate = 0.01
# final_learn_rate = 0.01
final_learn_rate = 0.001
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

pretrain_model_dir = './pretrain_model/'
final_model_dir = './final_model/'