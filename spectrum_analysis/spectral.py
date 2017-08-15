
#################################################
# draw the spectrum of different neighborhood in one class

# Author : ghzhang
# Date   :
# HomePage :
# Email  :
##################################################
#

from PIL import Image
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random


from matplotlib.mlab import PCA
import os

import draw_heatmap as dh

# pwd = os.path.dirname(os.path.abspath("__file__"))
# pardir = os.path.join(os.path.dirname("__file__"), os.path.pardir)
# pardir_abs = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
#
#
# hsi_file = os.path.join(pardir_abs,'hsi_data','Pavia','PaviaU.mat')
# gnd_file = os.path.join(pardir_abs,'hsi_data','Pavia','PaviaU_gt.mat')

hsi_file = '../hsi_data/Pavia/PaviaU.mat'
gnd_file = '../hsi_data/Pavia/PaviaU_gt.mat'


# load data
data = sio.loadmat(hsi_file)
img_3d = data['paviaU']

data = sio.loadmat(gnd_file)
img_2d = data['paviaU_gt']

print(img_2d.shape)
print(img_3d.shape)

plt.figure(1)

ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)

plt.sca(ax1)
plt.title('HSI')
plt.imshow(np.array(img_2d))

# class 3
flag = img_2d == 3
new_c = img_2d[flag]
#print new_c.size # pixels of class 3

plt.sca(ax2)
plt.title('class 3')
plt.imshow(flag)

# modify other class pixels to zero
new_img_3d = np.zeros((610,340,103))
for k in range(img_3d.shape[2]):
    new_img_3d[:,:,k] = np.multiply(img_3d[:,:,k],flag)
    #print k

# show
plt.sca(ax3)
#plt.title('HSI class 3 area')
#pt.imshow(new_img_3d[:,:,3])

vec=[]
num=0#
co=0
ma=0
w = 5# window size

c = Image.new("RGB",(img_2d.shape[1],img_2d.shape[0]))

dict_s={'1':'1','2':'s','3':'p','4':'3','5':'*','6':'+','7':'o'}
dict_c={'1':'b','2':'r','3':'g','4':'k','5':'m','6':'r','7':'y'}

spec_vecs = [] # record dif area spectrum
spectrum_all = []
plt.figure(3)

for i in range(int(img_2d.shape[0]/w)):
    for j in range(int(img_2d.shape[1]/w)):

        block = new_img_3d[i*w:i*w+w,j*w:j*w+w,:]

        if 0 not in block[:,:,0]:
            temp = np.mean(block,axis=0)
            vec = np.mean(temp,axis=0)

            print(np.mean(vec), vec.shape)
            #normalization
            vec = vec/np.mean(vec)
            print(np.mean(vec),vec.shape)

            spec_vecs.append(vec)
            # color the block
            col = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            i_w = i*w
            j_w = j*w
            for ii in range(w):
                for jj in range(w):
                    c.putpixel([j_w+jj,i_w+ii],col)

            num+=1 # block number

            co=int(num/7)+1
            ma=num%7+1

            print(co,ma)
            plt.figure(3)
            la = str(dict_c[str(co)]+'_'+str(num)+'_'+dict_s[str(ma)])
            plt.plot(range(new_img_3d.shape[2]),vec[:],label=la,color=dict_c[str(co)],marker=dict_s[str(ma)])


# PCA

#spectrum_all = np.vstack([spec_vecs[:]])
#results = PCA(spectrum_all)

print(num)

plt.ylabel("Intensity")
plt.xlabel("lambda")
plt.title('curve of dif class')
plt.legend()


#ax1=plt.subplot(121)
#ax2=plt.subplot(122)


plt.figure(1)
#cc = np.array(c)
plt.sca(ax3)
plt.title('dif class')
plt.imshow(c)

# list 2 array
temp = np.array(spec_vecs)
dist_E = np.zeros([13,13])
for i_v in range(temp.shape[0]):
    for j_v in range(temp.shape[0]):
        dist_E[i_v,j_v] = round(np.linalg.norm(temp[i_v,:]-temp[j_v,:]),2)

dist_nE = (dist_E-np.min(dist_E))/(np.max(dist_E)-np.min(dist_E)) # normalization

#dist_E = round(dist_E.reshape((1,-1)),2)
print(dist_nE, dist_E)

#hm = HeatMap(dist_E)

#plt.sca(ax2)
#plt.text(hm)

dh.draw_heatmap(dist_nE,range(13),range(13))
#fig_2 = dh.draw_heatmap(dist_E,range(13),range(13))


#plt.show()









