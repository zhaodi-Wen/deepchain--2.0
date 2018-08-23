


from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
'''
def reformat(samples,labels):
    samples = np.transpose(samples,(3,0,1,2))
    labels = np.array(list(x[0] for x in labels))
    one_hot_labels = []
    for num in labels:
        one_hot = [0.0]*10
        if num==10:
            one_hot[0] = 1.0
        else:
            one_hot[num] = 1.0
        one_hot_labels.append(one_hot)
        labels = np.array(one_hot_labels).astype(np.float32)
        return samples,labels

import scipy.io
train_data = scipy.io.loadmat('./SVHN/train_32x32.mat', variable_names='X').get('X')
train_labels = scipy.io.loadmat('./SVHN/train_32x32.mat', variable_names='y').get('y')
test_data = scipy.io.loadmat('./SVHN/test_32x32.mat', variable_names='X').get('X')
test_labels = scipy.io.loadmat('./SVHN/test_32x32.mat', variable_names='y').get('y')
extra_data = scipy.io.loadmat('./SVHN/extra_32x32.mat', variable_names='X').get('X')
extra_labels = scipy.io.loadmat('./SVHN/extra_32x32.mat', variable_names='y').get('y')


print(train_data.shape, train_labels.shape)
print(test_data.shape, test_labels.shape)
print(extra_data.shape, extra_labels.shape)
print(extra_labels[69])



train_labels[train_labels == 10] = 0
test_labels[test_labels == 10] = 0
extra_labels[extra_labels == 10] = 0



import random


random.seed()


n_labels = 10
valid_index = []
valid_index2 = []
train_index = []
train_index2 = []
for i in np.arange(n_labels):
    valid_index.extend(np.where(train_labels[:,0] == (i))[0][:400].tolist())
    train_index.extend(np.where(train_labels[:,0] == (i))[0][400:].tolist())
    valid_index2.extend(np.where(extra_labels[:,0] == (i))[0][:200].tolist())
    train_index2.extend(np.where(extra_labels[:,0] == (i))[0][200:].tolist())


random.shuffle(valid_index)
random.shuffle(train_index)
random.shuffle(valid_index2)
random.shuffle(train_index2)


#valid_data = np.concatenate((extra_data[:,:,:,valid_index2], train_data[:,:,:,valid_index]), axis=3).transpose((3,0,1,2))
valid_data = np.concatenate((extra_data[:,:,:,valid_index2], train_data[:,:,:,valid_index]), axis=3)
#valid_labels = np.concatenate((extra_labels[valid_index2,:], train_labels[valid_index,:]), axis=0)[:,0]
valid_labels = np.concatenate((extra_labels[valid_index2,:], train_labels[valid_index,:]), axis=0)
#train_data_t = np.concatenate((extra_data[:,:,:,train_index2], train_data[:,:,:,train_index]), axis=3).transpose((3,0,1,2))
train_data_t = np.concatenate((extra_data[:,:,:,train_index2], train_data[:,:,:,train_index]), axis=3)
#train_labels_t = np.concatenate((extra_labels[train_index2,:], train_labels[train_index,:]), axis=0)[:,0]
train_labels_t = np.concatenate((extra_labels[train_index2,:], train_labels[train_index,:]), axis=0)
#test_data = test_data.transpose((3,0,1,2))
test_data = test_data
test_labels = test_labels


print(train_data_t.shape, train_labels_t.shape)
print(test_data.shape, test_labels.shape)
print(valid_data.shape, valid_labels.shape)
print(type(train_labels_t))
train_data,train_labels = reformat(train_data_t,train_labels_t)
test_data,test_labels = reformat(test_data,test_labels)


print(train_labels)
'''
'''
scipy.io.savemat('./SVHN/train_32x32_1.mat',{'X': train_data_t,'y': train_labels_t})

scipy.io.savemat('./SVHN/valid_32x32.mat',{'X': valid_data,'y': valid_labels})

scipy.io.savemat('./SVHN/test_32x32_1.mat',{'X': test_data,'y': test_labels})
'''

# coding: UTF-8
from scipy.io import loadmat as load
import matplotlib.pyplot as plt
import numpy as np
from  PIL import  Image

def reformat(samples, labels):
    # 改变原始数据的形状
    # (图片高，图片宽，通道数，图片数)->(图片数,图片高，图片宽，通道数)
    # labels 变成one-hot encoding
    samples = np.transpose(samples, (3, 0, 1, 2)).astype(np.float32)
    labels = np.array([x[0] for x in labels])
    one_hot_labels = []
    for num in labels:
        one_hot = [0.0] * 10
        if num == 10:
            one_hot[0] = 1.0
        else:
            one_hot[num] = 1.0
        one_hot_labels.append(one_hot)
    labels = np.array(one_hot_labels).astype(np.float32)
    return samples, labels


def normalize(samples):
    # 将图片从0～255 线性映射到 -1.0～+1.0
    # 并且灰度化
    a = np.add.reduce(samples, keepdims=True , axis=3)
    a = a/3.0
    return a/128.0-1.0


def distribution(labels, name):
    # 查看一下每个label的分布。就是比例
    pass


def inspect(dataset, labels,n):
    # 显示图片看看
    for i in range(n):
        plt.subplot(2,5,i+1)
        print(labels[i])
        index = np.where(labels[i]==1.0)
        plt.title(index[0][0])
        #normalize(dataset[i])
        #if dataset[i].shape[3] ==1:
        #    shape = dataset[i].shape
        #    dataset[i] = dataset[i].reshape(shape[0],shape[1],shape[2])
        plt.imshow(dataset[i])
        plt.axis('off')
    plt.show()


train = load('./SVHN/train_32x32.mat')
test = load('./SVHN/test_32x32.mat')
valid = load('./SVHN/extra_32x32.mat')

print("Train Data Shape:", train['X'].shape)
print("Train Label Shape:", train['y'].shape)

train_samples = train['X']
train_labels = train['y']
test_samples = test['X']
test_labels = test['y']
valid_samples = valid['X']
valid_labels = valid['y']


_train_sample, _train_labels = reformat(train_samples, train_labels)
_test_sample, _test_labels = reformat(test_samples, test_labels)
_valid_sample ,_valid_labels = reformat(valid_samples,valid_labels)

print(_test_sample.shape,_test_labels.shape)
print(_train_sample.shape,_train_labels.shape)
print(_valid_sample.shape,valid_labels.shape)
num_labels = 10
image_size = 32

if __name__ == '__main__':
    # 探索数据
    #inspect(_valid_sample, _valid_labels, 10)
    normalize(_train_sample[1])
    print(_train_sample[1].shape)

