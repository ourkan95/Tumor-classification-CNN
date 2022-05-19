import warnings
import os
from os.path import join
from os import walk
import cv2 as cv
warnings.filterwarnings("ignore")
import numpy as np
import mat73
import itertools
import Preprocessing as pp
import matplotlib.pyplot as plt



data_path = 'C:\\Users\osman\Desktop\YSA_Proje\data\\'
dim = (256,256)

# im = mat73.loadmat(data_path + '1.mat')['cjdata']['image']
# plt.imshow(pp.preprocesss(im), cmap = 'gray')

# im = pp.preprocesss(im)
# np.amax(im)
# im = im / 255
# np.amax(im)

df = np.empty((3064,dim[0],dim[1]))
# tumor_mask = np.empty((3064,dim[0],dim[1]))
labels = np.empty((3064,1)).astype('int')

n = 1
for i in range(0,3064):
    data = mat73.loadmat(data_path + '{}.mat'.format(n))
    img = data['cjdata']['image']
    # mask = data['cjdata']['tumorMask'].astype('uint8')
    label = data['cjdata']['label'].astype('int')
    

    img = pp.preprocesss(img)
    # img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    n +=1
    print(n)
    labels[i] = label
    for j in range(0,dim[0]):
        for k in range(0,dim[0]):
            df[i][j][k] = img[j][k]          
df2 = df.reshape(df.shape[0], -1)
# tumor_mask2 = tumor_mask.reshape(tumor_mask.shape[0],-1)

plt.imshow(df[58], cmap = 'gray')

np.savetxt("data256.csv", df2)
# np.savetxt("tumor_mask.csv", tumor_mask2)
for i in range(3064):
    if labels[i] == 1:
            labels[i] = 0
    elif labels[i] == 2:
            labels[i] = 1    
    else:
            labels[i] = 2
np.savetxt('labels.csv', labels)

# plot1 = plt.imshow(df[0], cmap = 'gray')
# plt.figure()
# plot2 = plt.imshow(tumor_mask[0], cmap = 'gray')



# img = mat73.loadmat(data_path + '159.mat')['cjdata']['image']


# mask = mat73.loadmat(data_path + '159.mat')['cjdata']['tumorMask'].astype('uint8')
# img2, mask2 = pp.preprocesss(img,mask)
# plt.imshow(img2, cmap='gray')
# plt.imshow(mask2, cmap='gray')






