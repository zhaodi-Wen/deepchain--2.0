import numpy as np
import hipsternet.input_data as input_data
import hipsternet.neuralnet as nn
from hipsternet.solver import *
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.examples.tutorials.mnist import input_data
#from hipsternet.split_mnist import Split_data
from hipsternet.constant import worker_num
from hipsternet.Split_SVHN import Split_data
from hipsternet.normSVHN import norm_SVHN

n_iter = 1500
# alpha = 1e-2 adam
alpha = 5e-1
mb_size = 64
n_experiment = 1
reg = 1e-5
print_after = 1
p_dropout = .8
loss = 'cross_ent'
nonlin = 'relu'
solver = 'sgd'
solver3 = 'sgd3'
# worker_num =10

filename1 = './1.txt'
filename2 = './2.txt'
f1 = open(filename1, 'w')
f2 = open(filename2, 'w')
print('before')
SVHN = norm_SVHN.norm_SVHN()
print('after')
#mnist = input_data.read_data_sets('MNIST_Data/', one_hot=True)
print('before in run_SVHN')
data = Split_data(SVHN,worker_num)
X_train,y_train,X_test,y_test,X_val,y_val= data.split_data()
print('X_train shape',X_train[1].shape)
print('y_train shape',y_train[1].shape)
print('X_test shape',X_test[1].shape)
print(type(X_train[1]))
a = []
for i in range(worker_num):
    a.extend( X_val[i])
a = np.array(a)
print(a.shape)
print('after in run SVHN')
