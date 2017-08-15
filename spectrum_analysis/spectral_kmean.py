
#################################################
# kmeans: k-means cluster
# Author :
# Date   :
# HomePage :
# Email  :
#################################################

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np


import draw_heatmap as dh

# calculate Euclidean distance
def euclDistance(vector1, vector2):
	return np.sqrt(np.sum(np.square(vector2 - vector1)))

# init centroids with random samples
def initCentroids(dataSet, k):
	numSamples, dim = dataSet.shape
	centroids = np.zeros((k, dim))
	for i in range(k):
		index = int(np.random.uniform(0, numSamples))
		centroids[i, :] = dataSet[index, :]
	return centroids

# k-means cluster
def kmeans(dataSet, k):
	numSamples = dataSet.shape[0]
	# first column stores which cluster this sample belongs to,
	# second column stores the error between this sample and its centroid
	clusterAssment = np.mat(np.zeros((numSamples, 2)))
	clusterChanged = True

	## step 1: init centroids
	centroids = initCentroids(dataSet, k)

	while clusterChanged:
		clusterChanged = False
		## for each sample
		for i in range(numSamples):
			minDist  = 100000.0
			minIndex = 0
			## for each centroid
			## step 2: find the centroid who is closest
			for j in range(k):
				distance = euclDistance(centroids[j, :], dataSet[i, :])
				if distance < minDist:
					minDist  = distance
					minIndex = j
			
			## step 3: update its cluster
			if clusterAssment[i, 0] != minIndex:
				clusterChanged = True
				clusterAssment[i, :] = minIndex, minDist**2

		## step 4: update centroids
		for j in range(k):
			pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]
			centroids[j, :] = np.mean(pointsInCluster, axis = 0)

	print('Congratulations, cluster complete!')
	return centroids, clusterAssment

# show your cluster only available with 2-D data
def showCluster(dataSet, k, centroids, clusterAssment):
	numSamples, dim = dataSet.shape
	if dim != 2:
		print("Sorry! I can not draw because the dimension of your data is not 2!")
		return 1

	mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
	if k > len(mark):
		print("Sorry! Your k is too large! please contact Zouxy")
		return 1

	# draw all samples
	for i in range(numSamples):
		markIndex = int(clusterAssment[i, 0])
		plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

	mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
	# draw the centroids
	for i in range(k):
		plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)

	plt.show()

#if __name__ == "__main__":

hsi_file = '../hsi_data/Pavia/PaviaU.mat'
gnd_file = '../hsi_data/Pavia/PaviaU_gt.mat'

# load data
data = sio.loadmat(hsi_file)
img_3d = data['paviaU']

data = sio.loadmat(gnd_file)
img_2d = data['paviaU_gt']

# class 3
flag = img_2d == 3
new_c = img_2d[flag]
print(new_c.size) # pixels of class 3

temp=[]
ind =[]
for i in range(img_2d.shape[0]):
	for j in range(img_2d.shape[1]):
		if flag[i,j]==True:
			temp.append(img_3d[i,j,:])
			ind.append([i,j]) # index of i,j

pixel_vec = np.array(temp)
dataSet = np.mat(pixel_vec)
k = 5

centroids,clusterAssment = kmeans(dataSet,k)


# calculate spectral means of diff spectral shape

plt.figure(1)
mark = ['r', 'b', 'g', 'k', '^b', '+b', 'sb', 'db', '<b', 'pb']
spectra_class=[]

for i in range(k):
    temp = np.where(clusterAssment[:,0]==i) #
    spectra_mean=np.array(np.mean(dataSet[temp[0],:],0))
    spectra_class.append(spectra_mean)
    x = img_3d.shape[2]
    y = spectra_mean[0]

	# tt = np.array(ind)
	# for _ in range(len(ind)):
    #
    #
	# ind

# x = 136
# y = 76
#
# c = Image.new("RGB", (x, y))
#


# for i in range(0, x):
# 	for j in range(0, y):
# 		c.putpixel([i, j], (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
#
# c.show()


	#plt.plot(range(x),y,label='class' + str(i))
    #plt.plot(range(x),y,mark[i])

    #print(y)
    plt.plot(range(x),(y/np.mean(y)),label='class' + str(i))

plt.title('diff spectrum of inner class')
plt.legend()
#plt.show()

temp = spectra_class
dist_E = np.zeros([k,k])
for i_v in range(len(temp)):
    for j_v in range(len(temp)):
        dist_E[i_v,j_v] = euclDistance(temp[i_v][0],temp[j_v][0])


dh.draw_heatmap(dist_E,range(k),range(k))

print(dist_E)

# print "step 3: show the result..."
# showCluster(dataSet, k, centroids, clusterAssment)

