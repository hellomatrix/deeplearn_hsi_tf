#class Config(object):

encoder_layers = [60, 60, 60, 60]


# #-----------------for others
epoch_ae_pretrain_times = 150 # for
epoch_sae_pretrain_times =300# for others
epoch_final_train_times = 100000 # for others
batch_size = 200

# #-----------------for ksc
# epoch_ae_pretrain_times = 1500
# epoch_sae_pretrain_times =3000
# epoch_final_train_times = 100000
# batch_size = 100


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
Salinas17817 = 'Salinas17817'
Salinas_origin = 'Salinas_origin'

dic_pre_train={Salinas:Salinas,Salinas17817:Salinas,paviaU:paviaU,ksc:ksc}

random_state = 25535

pretrain_model_dir = './pretrain_model'
final_model_dir = './final_model'