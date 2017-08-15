
import numpy as np
import Config
import scipy.io as sio
from sklearn.decomposition import PCA


class Data():


    def __init__(self, data_name = ''):

        hsi_file = Config.data_path+'/'+data_name+'/'+data_name+'.mat'
        gnd_file = Config.data_path+'/'+data_name+'/'+data_name+'_gt.mat'

        hsi_img = sio.loadmat(hsi_file)[data_name].astype(np.float32)
        gnd_img = sio.loadmat(gnd_file)[data_name+'_gt'].astype(np.int32)

        #self.s_dim = hsi_img.shape[2]

        # img_2d = gnd_img[gnd_img>0]

        # img_3d = hsi_img[gnd_img>0]

        img_2d = np.reshape(gnd_img,(1,-1))
        img_3d = np.reshape(hsi_img,(-1,hsi_img.shape[2]))

        """ Scales all values in the ndarray ndar to be between 0 and 1 """
        # img_2d = self.scale_to_unit_interval(img_2d)
        # img_3d = self.scale_to_unit_interval(img_3d)
        #
        # # all data
        # self.all_data = [img_3d, img_2d]
        self.num_examples=img_3d.shape[0]
        self.img_2d_shape = gnd_img.shape

        self.hsi_img = self.scale_to_unit_interval(hsi_img)
        self.gnd_img = gnd_img

        self.split_mask = self.get_split_mask()


        ##-----------------------
        temp = self.split_mask.tolist()
        a=0
        for i in range(len(temp)):
            a+=temp[i].count('0')
        print('0 pixels in split_mask: %d'%a)
        ##------------------------


        # remove unlabel pixels
        self.split_mask[gnd_img==0]= '0'


        ##------------------------
        temp = self.split_mask.tolist()
        a=0
        for i in range(len(temp)):
            a+=temp[i].count('0')
        print('0 pixels after remove unlabel pixels : %d'%a)
        ##------------------------


    def scale_to_unit_interval(self,ndar, eps=1e-8):
        """ Scales all values in the ndarray ndar to be between 0 and 1 """
        ndar = np.float64(ndar.copy())
        ndar -= ndar.min()
        ndar *= 1.0 / (ndar.max() + eps)
        return ndar

    def get_split_mask(self,ratio=[6, 2, 2],random_state=Config.random_state):

        rand_num_generator = np.random.RandomState(random_state)

        random_mask = rand_num_generator.random_integers(1, sum(ratio), self.img_2d_shape)

        split_mask = np.array([['tests']*self.img_2d_shape[1],] * self.img_2d_shape[0])

        split_mask[random_mask <= ratio[0]] = 'train'

        split_mask[(random_mask <= ratio[1] + ratio[0]) * (random_mask > ratio[0])] = 'valid'


        print(split_mask.shape)

        return split_mask


    def get_train_valid_test_of_spectral_feature(self,batch_size=Config.batch_size):

        split_mask = self.split_mask

        train_data_x = self.hsi_img[split_mask=='train']
        train_data_y = self.gnd_img[split_mask=='train']

        ##-------------
        print('train:' ,train_data_x.shape)
        ##-------------

        valid_data_x = self.hsi_img[split_mask=='valid']
        valid_data_y = self.gnd_img[split_mask=='valid']


        test_data_x = self.hsi_img[split_mask=='tests']
        test_data_y = self.gnd_img[split_mask=='tests']

        print('train pixels:%d, valid pixels:%d, tests pixels:%d'\
              % (train_data_x.shape[0], valid_data_x.shape[0], test_data_x.shape[0]))

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

        print('train pixels:%d, valid pixels:%d, tests pixels:%d'\
              % (train_data_x.shape[0], valid_data_x.shape[0], test_data_x.shape[0]))

        return [train_data_x,train_data_y,valid_data_x,valid_data_y,test_data_x,test_data_y]


    def get_train_valid_test_of_spatial_feature(self,batch_size=Config.batch_size):


        # set other pixels to 0 except train valid test
        hsi_vec = np.reshape(self.hsi_img,[-1,self.hsi_img.shape[2]])
        zzeros= np.zeros(self.hsi_img.shape[2])

        print('hsi_vec_origin', hsi_vec[1,:])

        split_mask = np.reshape(self.split_mask,-1,) #set to vector

        print(split_mask.tolist().count('0'))
        hsi_vec[split_mask=='0'] = zzeros

        print('hsi_vec_setzero', len(hsi_vec.tolist()[1]),np.transpose(hsi_vec).tolist()[50].count(0))###

        # get pac values
        pca = PCA(n_components=3)
        newData = pca.fit_transform(hsi_vec)###

        print('newData', np.transpose(newData).tolist()[1].count(0))

        new_hsi_img = np.reshape(newData,[self.gnd_img.shape[0],self.gnd_img.shape[1],3])
        print(new_hsi_img.shape,new_hsi_img[1,1,:])

        # set neighbor pca as value of each pixels

        hsi = np.zeros([self.hsi_img.shape[0],self.hsi_img.shape[1],27])

        for i in range(1,self.hsi_img.shape[0]-1):
            for j in range(1,self.hsi_img.shape[1]-1):
                hsi[i, j, :] = np.reshape(new_hsi_img[i-1:i+2,j-1:j+2,:],[-1,])

        print(hsi.shape, hsi[:,1,1])
        # set 3 different datasets
        split_mask = self.split_mask

        train_data_x = hsi[split_mask=='train']
        train_data_y = self.gnd_img[split_mask=='train']

        ##-------------
        print('train_set shape:' ,train_data_x.shape, np.transpose(train_data_x).tolist()[14].count(0))
        ##-------------

        valid_data_x = hsi[split_mask=='valid']
        valid_data_y = self.gnd_img[split_mask=='valid']

        ##-------------
        print('valid_set shape:' ,valid_data_x.shape, np.transpose(valid_data_x).tolist()[14].count(0))
        ##-------------

        test_data_x = hsi[split_mask=='tests']
        test_data_y = self.gnd_img[split_mask=='tests']

        ##-------------
        print('test_set shape:' ,test_data_x.shape,np.transpose(test_data_x).tolist()[14].count(0))
        ##-------------

        print('train pixels:%d, valid pixels:%d, tests pixels:%d'\
              % (train_data_x.shape[0], valid_data_x.shape[0], test_data_x.shape[0]))

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

        print('train pixels:%d, valid pixels:%d, tests pixels:%d'\
              % (train_data_x.shape[0], valid_data_x.shape[0], test_data_x.shape[0]))

        return [train_data_x,train_data_y,valid_data_x,valid_data_y,test_data_x,test_data_y]

    def get_train_valid_test_of_mix_feature(self):

        spatial_set = self.get_train_valid_test_of_spatial_feature()
        spectral_set = self.get_train_valid_test_of_spectral_feature()

        print(spatial_set[0].shape,spatial_set[1].shape,spatial_set[2].shape,\
              spatial_set[3].shape,spatial_set[4].shape,spatial_set[5].shape)
        print(spectral_set[0].shape,spectral_set[1].shape,spectral_set[2].shape,\
              spectral_set[3].shape,spectral_set[4].shape,spectral_set[5].shape)

        train_data_x = np.hstack((spatial_set[0],spectral_set[0]))
        train_data_y = spatial_set[1]

        valid_data_x = np.hstack((spatial_set[2],spectral_set[2]))
        valid_data_y = spatial_set[3]

        test_data_x = np.hstack((spatial_set[4],spectral_set[4]))
        test_data_y = spatial_set[5]

        print(train_data_x.shape,train_data_y.shape,valid_data_x.shape, \
              valid_data_y.shape,test_data_x.shape,test_data_y.shape)

        return [train_data_x,train_data_y,valid_data_x,valid_data_y,test_data_x,test_data_y]


class Data_Set():

    def __init__(self,data):
        self.data = data
        self._index_in_epoch = 0
        self._num_examples = data[0].shape[0]
        self._images = data[0]
        self._labels = data[1]

    def next_batch(self,batch_size=Config.batch_size):

        start = self._index_in_epoch
        self._index_in_epoch +=batch_size

        if self._index_in_epoch>self._num_examples:

            self._index_in_epoch+=1

            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)

            self._images=self._images[perm]
            self._labels=self._labels[perm]

            start = 0

            self._index_in_epoch = batch_size
            assert batch_size<=self._num_examples

        end=self._index_in_epoch

        #print('start:%d,end:%d'%(start,end))  #shuoming zai bingxing

        return self._images[start:end],self._labels[start:end]







if __name__ == '__main__':

    import scipy.io as sio

    # img = sio.loadmat(Config.hsi_file)['paviaU']
    # gnd_img = sio.loadmat(Config.gnd_file)['paviaU_gt']
    # img = img.astype(np.float32)
    # gnd_img = gnd_img.astype(np.int32)

    # prepare data
    pd = Data(Config.ksc)
    #train_data = pd.all_data

    print(pd.split_mask.shape)

    # ap = pd.gnd_img.shape[0]*pd.gnd_img.shape[1]
    #
    # TVT = pd.get_train_valid_test_of_spectral_feature()
    # print(TVT[0].shape, TVT[1].shape, TVT[2].shape,TVT[3].shape, TVT[4].shape, TVT[5].shape)
    #
    # print('all pixels:%d, train pixels:%d, valid pixels:%d, tests pixels:%d'%(ap,TVT[0].shape[0],TVT[1].shape[0],TVT[2].shape[0]))
    #
    # ds = Data_Set([TVT[0],TVT[1]])
    # for i in range(5):
    #     ds.next_batch((50))
    #
    # print (TVT[0][1:2],TVT[1])


    TVT_pac = pd.get_train_valid_test_of_spatial_feature()
    print(TVT_pac[0].shape, TVT_pac[1].shape, TVT_pac[2].shape,TVT_pac[3].shape, TVT_pac[4].shape, TVT_pac[5].shape)
    print (TVT_pac[0][1:2],TVT_pac[1])


    TVT_mix = pd.get_train_valid_test_of_mix_feature()
    print (TVT_mix[0][1:2],TVT_mix[1])