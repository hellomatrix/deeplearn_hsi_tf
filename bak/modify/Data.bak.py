
import numpy as np
import tensorflow as tf
import Config



class Data():

    input_size = None
    output_size= None
    train_data = None

    def __init__(self, hsi_file = Config.hsi_file,gnd_file = Config.gnd_file):

        hsi_img = sio.loadmat(hsi_file)['paviaU'].astype(np.float32)
        gnd_img = sio.loadmat(gnd_file)['paviaU_gt'].astype(np.int32)

        self.s_dim = hsi_img.shape[2]

        img_2d = gnd_img[gnd_img>0]
        img_3d = hsi_img[gnd_img>0]

        labels = img_2d
        samples = img_3d

        # all data

        self.all_data = [img_3d, img_2d]



        random_state = Config.random_state
        # split mask
        #random_mask = rand_num_generator.randint(1, sum(ratio), data[1].shape)


        self.train_data,self.valid_data,\
        self.test_data = self.train_valid_test(self.all_data)


    def get_split_mask(self,ratio = Config.set_ratio):



    # devide dataset into 3 set:train valid test
    def train_valid_test(self,data):

        ratio = Config.set_ratio # train valid test ratio
        batch_size = Config.batch_size
        random_state = Config.random_state

        rand_num_generator = np.random.RandomState(random_state)

        random_mask = rand_num_generator.randint(1, sum(ratio), data[1].shape)
        split_mask = np.ndarray(data[1].shape[0],data[1].shape[1])

        split_mask[random_mask <= ratio[0]] = 'train'
        split_mask[(random_mask <= ratio[1] + ratio[0]) *
                   (random_mask > ratio[0])] = 'valid'

        train_data_x = data[0][split_mask == 'train']
        train_data_y = data[1][split_mask == 'train']
        valid_data_x = data[0][split_mask == 'valid']
        valid_data_y = data[1][split_mask == 'valid']
        test_data_x = data[0][split_mask == 'tests', :]
        test_data_y = data[1][split_mask == 'tests']

        # tackle the batch size mismatch problem
        mis_match = train_data_x.shape[0] % batch_size
        if mis_match != 0:
            mis_match = batch_size - mis_match
            train_data_x = np.vstack((train_data_x, train_data_x[0:mis_match, :]))
            train_data_y = np.hstack((train_data_y, train_data_y[0:mis_match]))

        mis_match = valid_data_x.shape[0] % batch_size
        if mis_match != 0:
            mis_match = batch_size - mis_match
            valid_data_x = np.vstack((valid_data_x, valid_data_x[0:mis_match, :]))
            valid_data_y = np.hstack((valid_data_y, valid_data_y[0:mis_match]))

        mis_match = test_data_x.shape[0] % batch_size
        if mis_match != 0:
            mis_match = batch_size - mis_match
            test_data_x = np.vstack((test_data_x, test_data_x[0:mis_match, :]))
            test_data_y = np.hstack((test_data_y, test_data_y[0:mis_match]))

        return [train_data_x, train_data_y], \
               [valid_data_x, valid_data_y], \
               [test_data_x, test_data_y]


    # spectral_feature_input_fn
    def spectral_feature_input_fn(self,batch_size):
        return 0

    # spacial_featue_input_fn

    # mixed features




if __name__ == '__main__':

    import scipy.io as sio
    #config = Config()

    img = sio.loadmat(Config.hsi_file)['paviaU']
    gnd_img = sio.loadmat(Config.gnd_file)['paviaU_gt']
    img = img.astype(np.float32)
    gnd_img = gnd_img.astype(np.int32)

    # prepare data
    #pd = Data()
    #train_data = pd.all_data

    #print(train_data[0].shape,train_data[1].shape)
