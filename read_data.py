import numpy as np
import pdb
import sys
import pandas as pd


class config(object):
    floatX = np.float32


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def cutoff_margine(length, width, window, win_level=1):
    '''
    eliminate the marginal pixeles
    return mask
    '''
    assert window % 2 == 1
    threshold = (window - 1) // 2
    thres_l = threshold * win_level
    assert 2 * thres_l + 1 < width
    assert 2 * thres_l + 1 < length
    mask_false = np.array([False] * width)
    mask_true = np.hstack((np.array([False] * thres_l), np.array(
        [True] * (width - 2 * thres_l)), np.array([False] * thres_l)))
    mask = np.vstack((np.tile(mask_false, [thres_l, 1]), np.tile(
        mask_true, [length - 2 * thres_l, 1]), np.tile(mask_false,
                                                       [thres_l, 1])))
    return mask, threshold


def get_extracted_index(gnd_img, window=7, flag='supervised', win_level=1):
    '''
    according 'flag' get the pixel index (not coordinate)
    '''
    length, width = gnd_img.shape
    # eliminate the marginal pixels, according to window_size
    mask, threshold = cutoff_margine(length, width, window, win_level)
    reshaped_mask = mask.reshape(mask.size)

    extracted_pixel_ind = None

    reshaped_gnd = gnd_img.reshape(gnd_img.size)
    if flag == 'supervised':
        extracted_pixel_ind = (reshaped_gnd > 0) * reshaped_mask
        extracted_pixel_ind = np.arange(reshaped_gnd.size)[extracted_pixel_ind]
    elif flag == 'unsupervised':
        extracted_pixel_ind = np.arange(reshaped_gnd.size)[reshaped_mask]
    else:
        print(sys.stderr)
        print("'flag' parameter error")
    return extracted_pixel_ind, threshold


def extract_data(hsi_img=None, gnd_img=None, window_size=1, flag='supervised'):
    '''
    extract the train data from image according to the gnd_img and "flag" argument
    Parameters:
    hsi_img: 3-D numpy.ndarray, initial HSI data
    gnd_img: 2-D numpy.ndarray, ground truth data
    window_size: Determins the scale of spatial information incorporated.
                 Note: must be odd
    flag: "supervised" or "unsupervised"

    '''
    # regularization
    hsi_img = scale_to_unit_interval(hsi_img)
    length, width, dim = hsi_img.shape

    extracted_pixel_ind, threshold = get_extracted_index(
        gnd_img, window_size, flag)

    reshaped_gnd = gnd_img.reshape(gnd_img.size)
    gndtruth = reshaped_gnd[extracted_pixel_ind]

    if window_size == 1:
        data_spatial = np.array([])
    else:
        data_spatial = np.zeros(
            [extracted_pixel_ind.size, window_size, window_size, dim],
            dtype=config.floatX)
        i = 0
        for ipixel in extracted_pixel_ind:
            ipixel_h = ipixel % width
            ipixel_v = ipixel / width
            data_spatial[i, :] = hsi_img[
                ipixel_v - threshold:ipixel_v + threshold + 1,
                ipixel_h - threshold:ipixel_h + threshold + 1, :]
            i += 1

    return data_spatial, gndtruth, extracted_pixel_ind


def train_valid_test(data, ratio=[6, 2, 2], batch_size=50, random_state=None):
    """
    This function splits data into three parts, according to the "ratio" parameter
    given in the lists indicating training, validating, testing data ratios.
    data:             a list containing:
                      1. A 2-D np.array object, with each patterns listed in
                         ROWs. Input data dimension MUST be larger than 1.
                      2. A 1-D np.array object, tags for each pattern.
                         '0' indicates that the tag for the corrresponding
                         pattern is unknown.
    ratio:            A list having 3 elements, indicating ratio of training,
                      validating and testing data ratios respectively.
    batch_size:       bathc_size helps to return an appropriate size of training
                      samples, which has divisibility over batch_size.
                      NOTE: batch_size cannot be larger than the minimal size of
                      all the trainin, validate and test dataset!
    random_state:     If we give the same random state and the same ratio on the
                      same data, the function will yield a same split for each
                      function call.
    return:
    [train_data_x, train_data_y]:
    [valid_data_x, valid_data_y]:
    [test_data_x , test_data_y ]:
                      Lists containing 2 np.array object, first for data and
                      second for truth. They are for training, validate and test
                      respectively. All the tags are integers in the range
                      [0, data[1].max()-1].
    split_mask
    """
    rand_num_generator = np.random.RandomState(random_state)

    #---------------------------split dataset-----------------------------------
    random_mask = rand_num_generator.randint(1, sum(ratio), data[0].shape[0])
    split_mask = np.array([
        'tests',
    ] * data[0].shape[0])
    split_mask[random_mask <= ratio[0]] = 'train'
    split_mask[(random_mask <= ratio[1] + ratio[0]) *
               (random_mask > ratio[0])] = 'valid'

    train_data_x = data[0][split_mask == 'train', :]
    train_data_y = data[1][split_mask == 'train'] - 1
    valid_data_x = data[0][split_mask == 'valid', :]
    valid_data_y = data[1][split_mask == 'valid'] - 1
    test_data_x = data[0][split_mask == 'tests', :]
    test_data_y = data[1][split_mask == 'tests'] - 1

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
           [test_data_x , test_data_y], split_mask


def train_valid_test_index(label_data, ratio, random=False):
    ser = pd.Series(range(label_data.size), index=label_data)
    grouped = ser.groupby(level=0)

    ratio = np.array(ratio)
    ratio_cum = ratio.cumsum()
    ratio_cum = ratio_cum * 1. / ratio_cum[-1]

    # [train_index, valid_index, test_index]
    indexes = [[], [], []]
    for key, g in grouped:
        inds = g.values
        if random:
            np.random.shuffle(inds)
        l = len(inds)
        ends = (ratio_cum * l).astype(np.int32)
        starts = np.hstack([[0], ends[:-1]])
        for i, (s, e) in enumerate(zip(starts, ends)):
            indexes[i].append(inds[s:e])
    indexes = [np.hstack(index) for index in indexes]
    return indexes


def train_valid_test_nonrand(data, ratio=[6, 2, 2], batch_size=50):
    """
    This function splits data into three parts, according to the "ratio" parameter
    given in the lists indicating training, validating, testing data ratios.
    data:             a list containing:
                      1. A 2-D np.array object, with each patterns listed in
                         ROWs. Input data dimension MUST be larger than 1.
                      2. A 1-D np.array object, tags for each pattern.
                         '0' indicates that the tag for the corrresponding
                         pattern is unknown.
    ratio:            A list having 3 elements, indicating ratio of training,
                      validating and testing data ratios respectively.
    batch_size:       bathc_size helps to return an appropriate size of training
                      samples, which has divisibility over batch_size.
                      NOTE: batch_size cannot be larger than the minimal size of
                      all the trainin, validate and test dataset!
    return:
    [train_data_x, train_data_y]:
    [valid_data_x, valid_data_y]:
    [test_data_x , test_data_y ]:
                      Lists containing 2 np.array object, first for data and
                      second for truth. They are for training, validate and test
                      respectively. All the tags are integers in the range
                      [0, data[1].max()-1].
    split_mask
    """

    #---------------------------split dataset-----------------------------------
    mask_inds = train_valid_test_index(data[1], ratio)
    split_mask = np.array([
        'train',
    ] * data[0].shape[0])
    split_mask[mask_inds[1]] = 'valid'
    split_mask[mask_inds[2]] = 'tests'

    train_data_x = data[0][split_mask == 'train', :]
    train_data_y = data[1][split_mask == 'train'] - 1
    valid_data_x = data[0][split_mask == 'valid', :]
    valid_data_y = data[1][split_mask == 'valid'] - 1
    test_data_x = data[0][split_mask == 'tests', :]
    test_data_y = data[1][split_mask == 'tests'] - 1

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
           [test_data_x , test_data_y], split_mask


def prepare_data(hsi_img=None,
                 gnd_img=None,
                 window_size=7,
                 batch_size=50,
                 ratio=[6, 2, 2],
                 random=True):
    """
    Process the data from file path to splited train-valid-test sets; Binded in
    dataset_spectral and dataset_spatial respectively.

    Parameters
    ----------

    hsi_img=None:       3-D numpy.ndarray, dtype=float, storing initial
                        hyperspectral image data.
    gnd_img=None:       2-D numpy.ndarray, dtype=int, containing tags for pixeles.
                        The size is the same to the hsi_img size, but with only
                        1 band.
    window_size:        Size of spatial window. Pass an integer 1 if no spatial
                        infomation needed.

    Return
    ------

    dataset_spatial:
    extracted_pixel_ind:
    split_mask:
    """
    data_spatial, gndtruth, extracted_pixed_ind = extract_data(
        hsi_img=hsi_img,
        gnd_img=gnd_img,
        window_size=window_size,
        flag='supervised')

    if random:
        split_data = train_valid_test
    else:
        split_data = train_valid_test_nonrand

    [train_spatial_x, train_y], [valid_spatial_x, valid_y], [test_spatial_x, test_y], split_mask = \
        split_data(data=[data_spatial, gndtruth], ratio=ratio,
                   batch_size=batch_size, random_state=123)

    dataset_spatial = [train_spatial_x,
                       train_y], [valid_spatial_x,
                                  valid_y], [test_spatial_x, test_y]

    return dataset_spatial, extracted_pixed_ind, split_mask


def generate_superPix_img(hsi_img=None,
                          gnd_img=None,
                          window_size=7,
                          flag='supervised'):
    '''
    generate super-pixel image
    every super-pixel is a [window_size, window_size, dim] matrix

    parameters:
    hsi_img: 3-D numpy.ndarray, initial HSI data
    gnd_img: 2-D numpy.ndarray, ground truth data
    window_size: Determins the scale of spatial information incorporated.
                 Note: must be odd
    flag: "supervised" or "unsupervised"
    '''
    hsi_img = scale_to_unit_interval(hsi_img)
    length, width, dim = hsi_img.shape

    cutoff_mask, threshold = cutoff_margine(length, width, window_size)
    extracted_gnd_img = gnd_img[cutoff_mask]
    extracted_l = length - 2 * threshold,
    extracted_w = width - 2 * threshold
    reshaped_cutoff_mask = cutoff_mask.reshape(cutoff_mask.size)
    extracted_ind = np.arange(cutoff_mask.size)[reshaped_cutoff_mask]

    superPix_img = np.zeros(
        [extracted_ind.size, window_size, window_size, dim])

    for i, ipixel in enumerate(extracted_ind):
        ipixel_h = ipixel % width
        ipixel_v = ipixel / width
        superPix_img[
            i, :] = hsi_img[ipixel_v - threshold:ipixel_v + threshold + 1,
                            ipixel_h - threshold:ipixel_h + threshold + 1, :]
    superPix_img = superPix_img.reshape(
        [extracted_l, extracted_w, window_size, window_size, dim])
    return superPix_img, extracted_gnd_img, extracted_ind


def get_superPix_split_index(extracted_gnd_img, ratio, window_size,
                             extracted_ind):
    '''
    gnd_truth_f: ground truth final version
    '''
    length, width, dim = extracted_gnd_img.shape
    cutoff_mask2 = cutoff_margine(length, width, window_size)
    reshaped_cutoff_mask2 = cutoff_mask2.reshape(cutoff_mask2.size)
    extracted_ind = extracted_ind[reshaped_cutoff_mask2]

    # ground truth level 2
    gnd_img_f = extracted_gnd_img[cutoff_mask2]
    relative_indexes = train_valid_test_index(gnd_img_f, ratio, random=True)
    abs_indexes = [extracted_ind[ind] for ind in relative_indexes]
    return relative_indexes, abs_indexes


def index2coordinate(index, height, width, window, win_level):
    '''
    from index to coodinate
    return mask for raw image
    height: raw image height
    width: raw image width
    '''
    assert window % 2 == 1
    threshold = (window - 1) // 2
    # double threshold (or level times threshold)
    d_th = win_level * threshold
    row = index // width
    col = index % width
    assert row >= d_th and row + d_th + 1 <= height
    assert col >= d_th and col + d_th + 1 <= width
    mask = np.zeros((height, width), dtype=bool)
    mask[row - d_th:row + d_th + 1, col - d_th:col + d_th + 1] = True
    return mask, d_th


def get_MLdata_by_index(index, hsi_img, gnd_img, window=7, win_level=1):
    '''
    Parameters
    ----------

    hsi_img=None:       3-D numpy.ndarray, dtype=float, storing initial
                        hyperspectral image data.
    gnd_img=None:       2-D numpy.ndarray, dtype=int, containing tags for pixeles.
                        The size is the same to the hsi_img size, but with only
                        1 band.
    window_size:        Size of spatial window. Pass an integer 1 if no spatial
                        infomation needed.
    win_level: merge times

    Return
    ------
    super-pixel is a [window_size, window_size, dim] matrix
    '''
    length, width = gnd_img.shape
    # d_th: double threshold (decided by win_level)
    mask, d_th = index2coordinate(index, length, width, window, win_level)
    new_width = 2 * d_th + 1
    superPix = hsi_img[mask].reshape((new_width, new_width, -1))
    label = gnd_img.reshape(gnd_img.size)[index]
    return superPix, label


class get_MLdata_batch(object):
    '''
    super-pixel is a [window_size, window_size, dim] matrix
    '''

    def __init__(self,
                 hsi_img,
                 gnd_img,
                 batch_size=50,
                 ratio=[6, 2, 2],
                 window=7,
                 random=True,
                 win_level=1,
                 flag='supervised'):
        extracted_ind, _ = get_extracted_index(
            gnd_img, window=7, win_level=win_level,
            flag=flag)
        extracted_gnd = gnd_img.reshape(gnd_img.size)[extracted_ind]
        t_v_t_index = train_valid_test_index(
            extracted_gnd, ratio, random=random)
        self.train_ind = train_ind = extracted_ind[t_v_t_index[0]]
        self.valid_ind = extracted_ind[t_v_t_index[1]]
        self.test_ind = extracted_ind[t_v_t_index[2]]
        self.batch_length = len(train_ind) // batch_size
        self.batch_size = batch_size
        self.hsi_img = scale_to_unit_interval(hsi_img)
        self.gnd_img = gnd_img
        self.window = window
        self.win_level = win_level
        self.extracted_ind = extracted_ind

    def train_data(self):
        batch_size = self.batch_size
        batch_ind = np.arange(self.batch_length)
        train_ind = self.train_ind
        hsi_img = self.hsi_img
        gnd_img = self.gnd_img
        while True:
            np.random.shuffle(train_ind)
            for i in batch_ind:
                batch_X = []
                batch_y = []
                train_ind_i = train_ind[i * batch_size:(i + 1) * batch_size]
                for ind in train_ind_i:
                    superPix, label = get_MLdata_by_index(
                        ind,
                        hsi_img,
                        gnd_img,
                        window=self.window,
                        win_level=self.win_level)
                    batch_X.append(superPix)
                    batch_y.append(label)
                batch_X = np.stack(batch_X)
                batch_y = np.hstack(batch_y)
                # NOTE: label start from 1
                batch_y = batch_y - 1
                yield batch_X, batch_y

    def train_data_whole(self):
        train_ind = self.train_ind
        hsi_img = self.hsi_img
        gnd_img = self.gnd_img
        superPixes = []
        labels = []
        for ind in train_ind:
            superPix, label = get_MLdata_by_index(
                ind,
                hsi_img,
                gnd_img,
                window=self.window,
                win_level=self.win_level)
            superPixes.append(superPix)
            labels.append(label)
        superPixes = np.stack(superPixes)
        labels = np.hstack(labels)
        return superPixes, labels - 1

    def all_data(self):
        all_ind = self.extracted_ind
        hsi_img = self.hsi_img
        gnd_img = self.gnd_img
        superPixes = []
        labels = []
        for ind in all_ind:
            superPix, label = get_MLdata_by_index(
                ind,
                hsi_img,
                gnd_img,
                window=self.window,
                win_level=self.win_level)
            superPixes.append(superPix)
            labels.append(label)
        superPixes = np.stack(superPixes)
        labels = np.hstack(labels)
        return superPixes, labels - 1

    def test_data(self):
        test_ind = self.test_ind
        hsi_img = self.hsi_img
        gnd_img = self.gnd_img
        superPixes = []
        labels = []
        for ind in test_ind:
            superPix, label = get_MLdata_by_index(
                ind,
                hsi_img,
                gnd_img,
                window=self.window,
                win_level=self.win_level)
            superPixes.append(superPix)
            labels.append(label)
        superPixes = np.stack(superPixes)
        labels = np.hstack(labels)
        return superPixes, labels - 1

    def valid_data(self):
        valid_ind = self.valid_ind
        hsi_img = self.hsi_img
        gnd_img = self.gnd_img
        superPixes = []
        labels = []
        for ind in valid_ind:
            superPix, label = get_MLdata_by_index(
                ind,
                hsi_img,
                gnd_img,
                window=self.window,
                win_level=self.win_level)
            superPixes.append(superPix)
            labels.append(label)
        superPixes = np.stack(superPixes)
        labels = np.hstack(labels)
        return superPixes, labels - 1


if __name__ == '__main__':
    import scipy.io as sio
    hsi_file = './data/PaviaU.mat'
    gnd_file = './data/PaviaU_gt.mat'
    img = sio.loadmat(hsi_file)['paviaU']
    gnd_img = sio.loadmat(gnd_file)['paviaU_gt']
    img = img.astype(np.float32)
    gnd_img = gnd_img.astype(np.int32)
    # -------------------- test prepare_data ---------------------
    # datasets, extracted_pixed_ind, split_mask = prepare_data(
    #     hsi_img=img, gnd_img=gnd_img, window_size=7, random=True)
    # ------------------------------------------------------------

    # # -------------------- test superPix_img ---------------------
    # superPix_img, extracted_gnd_img, extracted_ind = generate_superPix_img(
    #     hsi_img=img, gnd_img=gnd_img, window_size=7)
    # relative_indexes, abs_indexes = get_superPix_split_index(
    #     extracted_gnd_img, [6, 2, 2], 7, extracted_ind)
    # # -------------------------------------------------------------

    # # -------------------- test get data from index ---------------
    # extracted_ind, _ = get_extracted_index(gnd_img, window=7, win_level=2)
    # superPixel, label = get_MLdata_by_index(
    #     extracted_ind[0], img, gnd_img, window=7)
    # # -------------------------------------------------------------

    # -------------------- test batch -------------------------------
    batch = get_MLdata_batch(img, gnd_img, batch_size=3)
    # ---------------------------------------------------------------
