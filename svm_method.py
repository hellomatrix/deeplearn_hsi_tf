import scipy.io as sio
import numpy as np
from read_data import get_MLdata_batch, scale_to_unit_interval


window = 1

# hsi_file = '../hsi_data/Salinas/Salinas.mat'
# gnd_file = '../hsi_data/Salinas/Salinas_gt.mat'

# hsi_file = '../hsi_data/Salinas_origin/Salinas_origin.mat'
# gnd_file = '../hsi_data/Salinas_origin/Salinas_origin_gt.mat'
#
# hsi_file = '../hsi_data/'+ dataname+/Salinas_origin.mat'
# gnd_file = '../hsi_data/Salinas_origin/Salinas_origin_gt.mat'


h = sio.loadmat(hsi_file)
g = sio.loadmat(gnd_file)

img = h[list(h.keys())[-1]]
gnd_img = g[list(g.keys())[-1]]

# pdb.set_trace()
img = img.astype(np.float32)
gnd_img = gnd_img.astype(np.int32)
ml_data_generate = get_MLdata_batch(img, gnd_img, window=1, random=False)

train_X_w, train_y = ml_data_generate.train_data_whole()
valid_X_w, valid_y  = ml_data_generate.valid_data()
test_X_w, test_y = ml_data_generate.test_data()

print('here1')

# construct data
data_sets = [np.reshape(train_X_w,[train_X_w.shape[0],-1]), train_y,
             np.reshape(valid_X_w,[valid_X_w.shape[0],-1]), valid_y,
             np.reshape(test_X_w,[test_X_w.shape[0],-1]), test_y]


# # add spatial information
# def merge_accord_distance(superPix):
#     '''
#     merge the first 1/4 nearest pixels
#     window is odd
#     '''
#     window, _, dim = superPix.shape
#     ind0 = window // 2
#     vec0 = superPix[ind0, ind0]
#     superPix = superPix.reshape([-1, dim])
#     distances = []
#     for pix in superPix:
#         distances.append(euclidean(vec0, pix))
#     distances = np.array(distances)
#     dist_sorted = np.sort(distances)
#     pix_filter = distances <= dist_sorted[window**2 // 4]
#     pix_filted = superPix[pix_filter]
#     if pix_filted.shape[0] < 1:
#         pdb.set_trace()
#     merged_pix = pix_filted.mean(axis=0)
#     # return scale_to_unit_interval(vec0), scale_to_unit_interval(merged_pix)
#     return vec0, merged_pix
#
#
# def makeGaussian(size, fwhm=3, center=None):
#     """ Make a square gaussian kernel.
#     size is the length of a side of the square
#     fwhm is full-width-half-maximum, which
#     can be thought of as an effective radius.
#     """
#
#     x = np.arange(0, size, 1, float)
#     y = x[:, np.newaxis]
#
#     if center is None:
#         x0 = y0 = size // 2
#     else:
#         x0 = center[0]
#         y0 = center[1]
#     filter = np.exp(-4 * np.log(2) * ((x - x0)**2 + (y - y0)**2) / fwhm**2)
#
#     return filter / filter.sum()
#
#
# def merge_by_gaussian(superPix):
#     window, _, dim = superPix.shape
#     filter_ = makeGaussian(window, fwhm=window // 2)
#     superPix = superPix.reshape([-1, dim])
#     filter_ = filter_.reshape([-1, 1])
#     return (superPix * filter_).mean(axis=0)
#
#
# train_data = [merge_accord_distance(superPix) for superPix in train_data_w]
# valid_X = [merge_accord_distance(superPix) for superPix in valid_X_w]
# test_X = [merge_accord_distance(superPix) for superPix in test_X_w]
# train_data = np.array(train_data)
# valid_X = np.array(valid_X)
# test_X = np.array(test_X)
#
# train_data_G = [merge_by_gaussian(superPix) for superPix in train_data_w]
# valid_X_G = [merge_by_gaussian(superPix) for superPix in valid_X_w]
# test_X_G = [merge_by_gaussian(superPix) for superPix in test_X_w]
# train_data_G = np.array(train_data_G)
# valid_X_G = np.array(valid_X_G)
# test_X_G = np.array(test_X_G)
#
# best_c = 10000.
# best_g = 10.
# svm_classifier0 = svm.SVC(C=best_c, gamma=best_g, kernel='rbf')
# svm_classifier0.fit(train_data[:, 0, :], labels)
# valid_pred = svm_classifier0.predict(valid_X[:, 0, :])
# test_pred = svm_classifier0.predict(test_X[:, 0, :])
#
# accuracy_valid = accuracy_score(valid_y, valid_pred)
# accuracy_test = accuracy_score(test_y, test_pred)
# print('valid accuracy %f' % accuracy_valid)
# print('test accuracy %f' % accuracy_test)
#
# svm_classifier1 = svm.SVC(C=best_c, gamma=best_g, kernel='rbf')
# svm_classifier1.fit(train_data[:, 1, :], labels)
# valid_pred = svm_classifier1.predict(valid_X[:, 1, :])
# test_pred = svm_classifier1.predict(test_X[:, 1, :])
#
# accuracy_valid = accuracy_score(valid_y, valid_pred)
# accuracy_test = accuracy_score(test_y, test_pred)
# print('spatial valid accuracy %f' % accuracy_valid)
# print('spatial test accuracy %f' % accuracy_test)
#
# svm_classifier2 = svm.SVC(C=best_c, gamma=best_g, kernel='rbf')
# svm_classifier2.fit(train_data_G, labels)
# valid_pred = svm_classifier2.predict(valid_X_G)
# test_pred = svm_classifier2.predict(test_X_G)
#
# accuracy_valid = accuracy_score(valid_y, valid_pred)
# accuracy_test = accuracy_score(test_y, test_pred)
# print('Gausian valid accuracy %f' % accuracy_valid)
# print('Gausian test accuracy %f' % accuracy_test)
#
# # ----------------------- predict whole data ------------------------
# ml_data_generate_all = get_MLdata_batch(
#     img, gnd_img, window=window, random=False, flag='unsupervised')
# all_data_w, labels = ml_data_generate_all.all_data()
# all_data = [merge_accord_distance(superPix) for superPix in all_data_w]
# all_data = np.array(all_data)
#
# all_data_G = [merge_by_gaussian(superPix) for superPix in all_data_w]
# all_pred0 = svm_classifier0.predict(all_data[:, 0, :])
# all_pred1 = svm_classifier1.predict(all_data[:, 1, :])
# all_pred2 = svm_classifier2.predict(all_data_G)
#
# pickle.dump((all_pred0, all_pred1, all_pred2, labels),
#             open('indianPines_svm_results.pkl', 'wb'))
