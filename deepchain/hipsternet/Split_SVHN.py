#coding: UTF-8
from __future__ import print_function
from sklearn.model_selection import train_test_split
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
#from hipsternet.constant import worker_num
from  hipsternet.normSVHN import norm_SVHN
#from normSVHN import norm_SVHN
from scipy.io import loadmat as load
import matplotlib.pyplot as plt
import os
from  PIL import  Image
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
'''
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


train = load('../SVHN/train_32x32.mat')
test = load('../SVHN/test_32x32.mat')
valid = load('../SVHN/extra_32x32.mat')

print("Train Data Shape:", train['X'].shape)
print("Train Label Shape:", train['y'].shape)

train_samples = train['X']
train_labels = train['y']
test_samples = test['X']
test_labels = test['y']
valid_samples = valid['X']
valid_labels = valid['y']


train_x, train_y = reformat(train_samples, train_labels)
test_x, test_y = reformat(test_samples, test_labels)
valid_x ,valid_y = reformat(valid_samples,valid_labels)

#print(_test_sample.shape,_test_labels.shape)
#print(_train_sample.shape,_train_labels.shape)
#print(_valid_sample.shape,valid_labels.shape)
num_labels = 10
image_size = 32

data = (train_x,train_y,test_x,test_y,valid_x,valid_y)
'''
# train_x,train_y,test_x,test_y = mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels

'''
print('before')
data = norm_SVHN.norm_SVHN()
print('after')
'''
class Split_data(object):

    def __init__(self,data, worker_num):
        self.data = data
        self.worker_num = worker_num

    def split_data(self):
        # data = input_data.read_data_sets('MNIST_Data/',one_hot=True )
        train_x, train_y, test_x, test_y, vali_x, vali_y = self.data

        split_slice_train = int(len(train_x) / (self.worker_num))
        split_slice_test = int(len(test_x) / (self.worker_num))
        split_slice_vali = int(len(vali_x) / (self.worker_num))
        split_data_train_x = []
        split_data_train_y = []
        split_data_test_x = []
        split_data_test_y = []
        split_data_vali_x = []
        split_data_vali_y = []
        start_train = 0
        start_test = 0
        start_vali = 0
        for i in range(self.worker_num):
            split_data_train_x.append(train_x[start_train:start_train + split_slice_train])
            split_data_train_y.append(train_y[start_train:start_train + split_slice_train])
            split_data_test_x.append(test_x[start_test:start_test + split_slice_test])
            split_data_test_y.append(test_y[start_test:start_test + split_slice_test])
            split_data_vali_x.append(vali_x[start_vali:start_vali + split_slice_vali])
            split_data_vali_y.append(vali_y[start_vali:start_vali + split_slice_vali])
            start_train += split_slice_train
            start_test += split_slice_test
            start_vali += split_slice_vali

        return split_data_train_x, split_data_train_y, split_data_test_x, split_data_test_y, \
            split_data_vali_x,split_data_vali_y
'''
data = Split_data(data, 4)
split_data_train_x, split_data_test_x, split_data_train_y, split_data_test_y,split_data_vali_x,split_data_vali_y = data.split_data()
print('split_data_train_x:', split_data_train_x[1].shape)
print('split_data_test_x:', split_data_test_x[0].shape)
print('split_data_train_y:', split_data_train_y[1].shape)
print('split_data_test_y:', split_data_test_y[0].shape)
'''

