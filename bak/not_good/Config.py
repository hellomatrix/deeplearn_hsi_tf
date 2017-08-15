


#class Config(object):

encoder_layers = [60, 60, 60, 60]

epoch_size = 10000
userbm = True
rbm_epoch = 10
au_epoch = 10
tied_weights = True
batch_size = 100

learn_rate = 0.01
set_ratio = [6, 2, 2]

#NUM_CLASSES = 10
#IMAGE_PIXELS = 103 # a variable later

spectral_log = 'spectral_log'
spatial_log='spatial_log'
mix_feature_log = 'mix_log'

max_steps = 100000

data_path= '../hsi_data'

# hsi_file = '../hsi_data/Pavia/PaviaU.mat'
# gnd_file = '../hsi_data/Pavia/PaviaU_gt.mat'
# PaviaU = 'PaviaU'

# hsi_file1 = '../hsi_data/KSC/KSC.mat'
# gnd_file1 = '../hsi_data/KSC/KSC_gt.mat'

ksc = 'KSC'
ksc_class_num = 16

paviaU = 'PaviaU'
paviaU_class_num = 10

random_state = 25535