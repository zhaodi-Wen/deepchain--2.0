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
from scipy.io import loadmat as load
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle as skshuffle
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
file = './1.txt'
f = open(file,'w')
def reformat(samples, labels):
    # 改变原始数据的形状
    # (图片高，图片宽，通道数，图片数)->(图片数,图片高，图片宽，通道数)
    # labels 变成one-hot encoding
    #samples = np.transpose(samples, (3, 0, 1, 2)).astype(np.float32)
    samples = np.transpose(samples, (3, 2, 0, 1)).astype(np.float32)
    #labels = np.array([x[0] for x in labels])
    '''
    print('labels',labels)
    print('labels shape',labels.shape)
    one_hot_labels = []
    
    for num in labels:
        one_hot = [0.0] * 10
        if num == 10:
            one_hot[0] = 1.0
        else:
            one_hot[num] = 1.0
        one_hot_labels.append(one_hot)
    labels = np.array(one_hot_labels).astype(np.float32)
    '''
    print('sample shape',samples.shape)
    index =[]
    print(labels.shape[0])
    #labels = labels.squeeze()
    #np.place(labels, labels == 10, 0)
    for i in range(labels.shape[0]):
        if labels[i][0]==10:
            #print(i)
            index.append(i)
            labels[i][0]=0
            #samples = np.delete(samples,i,0)
    #for i in range(labels.shape[0]):
    #    if labels[i][0]==10:
    #        print(i)
    labels = np.array([x[0] for x in labels])
    #labels = np.delete(labels,index)
    #samples = np.delete(samples[0],index)
    print('samples shape',samples.shape)
    #print('label max',labels.max())
    #print('samples shape',samples.shape)
    print('labels shape',labels.shape)

    return samples, labels


def normalize(samples):
    # 将图片从0～255 线性映射到 -1.0～+1.0
    # 并且灰度化
    #samples = np.add.reduce(samples, keepdims=True , axis=1)
    #f.write('train_samples middle {} {}' .format('\n' ,a[0]))
    #samples = samples/3.0
    #f.write('train_samples middle/3 {} {}'.format('\n', a[0]))
    return samples/256.0-1.0
    #return a


def distribution(labels, name):
    # 查看一下每个label的分布。就是比例
    pass


def inspect(dataset, labels,m,n):
    # 显示图片看看
    for i in range(m,n):
        plt.subplot(2,5,i+1-m)
        print(labels[i])
        #index = np.where(labels[i]==1.0)
        plt.title(labels[i])
        #normalize(dataset[i])
        #if dataset[i].shape[3] ==1:
        #    shape = dataset[i].shape
        #    dataset[i] = dataset[i].reshape(shape[0],shape[1],shape[2])
        plt.imshow(dataset[i,:,0])
        plt.axis('off')
    plt.show()

class norm_SVHN():
    def __init__(self):
        pass

    def get_minibatch(self,X, y, minibatch_size, shuffle=True):
        minibatches = []
        if shuffle:
            X, y = skshuffle(X,y)

        return X[:minibatch_size],y[:minibatch_size]

    def norm_SVHN(self):
        #np.set_printoptions(threshold=np.NAN,precision=8)
        train = load('./SVHN/train_32x32.mat')
        test = load('./SVHN/test_32x32.mat')
        valid = load('./SVHN/extra_32x32.mat')
        train_samples = train['X']
        train_labels = train['y']
        test_samples = test['X']
        test_labels = test['y']
        valid_samples = valid['X']
        valid_labels = valid['y']
        print('before train sample ',train_samples.shape)
        print('train_sample shape',train_samples.shape)
        train_x, train_y = reformat(train_samples, train_labels)
        test_x, test_y = reformat(test_samples, test_labels)
        valid_x ,valid_y = reformat(valid_samples,valid_labels)
        train_x = normalize(train_x)
        test_x = normalize(test_x)
        valid_x = normalize(valid_x)

        print(test_x.shape ,test_y.shape)
        print(train_x.shape, train_y.shape)
        print(valid_x.shape,valid_y.shape)

        num_labels = 10
        image_size = 32
        print('in normSVHN')
        train_x, train_y = self.get_minibatch(train_x, train_y, 55000)
        test_x, test_y = self.get_minibatch(test_x, test_y, 5000)
        valid_x, valid_y = self.get_minibatch(valid_x, valid_y, 5000)
        data = (train_x, train_y, test_x, test_y, valid_x, valid_y)
        return data
