


#class Config(object):

encoder_layers = [60, 60, 60, 60]


# epoch_pretrain_times = 60
epoch_ae_pretrain_times = 500
# SAE_train_times =100
epoch_sae_pretrain_times =1000
# epoch_train_times = 10000  # fro salinas
epoch_final_train_times = 10000
#au_epoch = 10
#tied_weights = True
batch_size = 100

learn_rate = 0.01
# final_learn_rate = 0.01
final_learn_rate = 0.0001
set_ratio = [6, 2, 2]


#NUM_CLASSES = 10
#IMAGE_PIXELS = 103 # a variable later
#spectral_log = 'spectral_log'
#spatial_log='spatial_log'
#mix_feature_log = 'mix_log'

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
