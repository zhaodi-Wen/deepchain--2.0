from typing import Tuple

import numpy as np
import hipsternet.loss as loss_fun
import hipsternet.layer as l
import hipsternet.regularization as reg
import hipsternet.utils as util
from numpy.core.multiarray import ndarray


class NeuralNet(object):

    loss_funs = dict(
        cross_ent=loss_fun.cross_entropy,
        hinge=loss_fun.hinge_loss,
        squared=loss_fun.squared_loss,
        l2_regression=loss_fun.l2_regression,
        l1_regression=loss_fun.l1_regression
    )

    dloss_funs = dict(
        cross_ent=loss_fun.dcross_entropy,
        hinge=loss_fun.dhinge_loss,
        squared=loss_fun.dsquared_loss,
        l2_regression=loss_fun.dl2_regression,
        l1_regression=loss_fun.dl1_regression
    )

    forward_nonlins = dict(
        relu=l.relu_forward,
        lrelu=l.lrelu_forward,
        sigmoid=l.sigmoid_forward,
        tanh=l.tanh_forward
    )

    backward_nonlins = dict(
        relu=l.relu_backward,
        lrelu=l.lrelu_backward,
        sigmoid=l.sigmoid_backward,
        tanh=l.tanh_backward
    )

    def __init__(self, D, C, H, lam=1e-3, p_dropout=.8, loss='cross_ent', nonlin='relu'):
        if loss not in NeuralNet.loss_funs.keys():
            raise Exception('Loss function must be in {}!'.format(NeuralNet.loss_funs.keys()))

        if nonlin not in NeuralNet.forward_nonlins.keys():
            raise Exception('Nonlinearity must be in {}!'.format(NeuralNet.forward_nonlins.keys()))

        self._init_model(D, C, H)

        self.lam = lam
        self.p_dropout = p_dropout
        self.loss = loss
        self.forward_nonlin = NeuralNet.forward_nonlins[nonlin]
        self.backward_nonlin = NeuralNet.backward_nonlins[nonlin]
        self.mode = 'classification'

        if 'regression' in loss:
            self.mode = 'regression'

    def train_step(self, X_train, y_train):
        """
        Single training step over minibatch: forward, loss, backprop
        """
        y_pred, cache = self.forward(X_train, train=True)

        loss = self.loss_funs[self.loss](self.model, y_pred, y_train, self.lam)
        grad = self.backward(y_pred, y_train, cache)

        return grad, loss

    def predict_proba(self, X):
        score, _ = self.forward(X, False)
        return util.softmax(score)

    def predict(self, X):
        if self.mode == 'classification':
            return np.argmax(self.predict_proba(X), axis=1)
        else:
            score, _ = self.forward(X, False)
            y_pred = np.round(score)
            return y_pred

    def forward(self, X, train=False):
        print('in forward')
        raise NotImplementedError()

    def backward(self, y_pred, y_train, cache):
        raise NotImplementedError()

    def _init_model(self, D, C, H):
        raise NotImplementedError()


class FeedForwardNet(NeuralNet):

    def __init__(self, D, C, H, lam=1e-3, p_dropout=.8, loss='cross_ent', nonlin='relu'):
        super().__init__(D, C, H, lam, p_dropout, loss, nonlin)

    def forward(self, X, train=True):

        gamma1, gamma2 = self.model['gamma1'], self.model['gamma2']
        beta1, beta2 = self.model['beta1'], self.model['beta2']

        u1, u2 = None, None
        bn1_cache, bn2_cache = None, None

        # First layer
        h1, h1_cache = l.fc_forward(X, self.model['W1'], self.model['b1'])
        bn1_cache = (self.bn_caches['bn1_mean'], self.bn_caches['bn1_var'])
        h1, bn1_cache, run_mean, run_var = l.bn_forward(h1, gamma1, beta1, bn1_cache, train=train)
        h1, nl_cache1 = self.forward_nonlin(h1)

        self.bn_caches['bn1_mean'], self.bn_caches['bn1_var'] = run_mean, run_var

        if train:
            h1, u1 = l.dropout_forward(h1, self.p_dropout)

        # Second layer
        h2, h2_cache = l.fc_forward(h1, self.model['W2'], self.model['b2'])
        bn2_cache = (self.bn_caches['bn2_mean'], self.bn_caches['bn2_var'])
        h2, bn2_cache, run_mean, run_var = l.bn_forward(h2, gamma2, beta2, bn2_cache, train=train)
        h2, nl_cache2 = self.forward_nonlin(h2)

        self.bn_caches['bn2_mean'], self.bn_caches['bn2_var'] = run_mean, run_var

        if train:
            h2, u2 = l.dropout_forward(h2, self.p_dropout)

        # Third layer
        score, score_cache = l.fc_forward(h2, self.model['W3'], self.model['b3'])

        cache = (X, h1_cache, h2_cache, score_cache, nl_cache1, nl_cache2, u1, u2, bn1_cache, bn2_cache)

        return score, cache

    def backward(self, y_pred, y_train, cache):
        X, h1_cache, h2_cache, score_cache, nl_cache1, nl_cache2, u1, u2, bn1_cache, bn2_cache = cache

        # Output layer
        grad_y = self.dloss_funs[self.loss](y_pred, y_train)

        # Third layer
        dh2, dW3, db3 = l.fc_backward(grad_y, score_cache)
        dW3 += reg.dl2_reg(self.model['W3'], self.lam)
        dh2 = self.backward_nonlin(dh2, nl_cache2)
        dh2 = l.dropout_backward(dh2, u2)
        dh2, dgamma2, dbeta2 = l.bn_backward(dh2, bn2_cache)

        # Second layer
        dh1, dW2, db2 = l.fc_backward(dh2, h2_cache)
        dW2 += reg.dl2_reg(self.model['W2'], self.lam)
        dh1 = self.backward_nonlin(dh1, nl_cache1)
        dh1 = l.dropout_backward(dh1, u1)
        dh1, dgamma1, dbeta1 = l.bn_backward(dh1, bn1_cache)

        # First layer
        _, dW1, db1 = l.fc_backward(dh1, h1_cache)
        dW1 += reg.dl2_reg(self.model['W1'], self.lam)

        grad = dict(
            W1=dW1, W2=dW2, W3=dW3, b1=db1, b2=db2, b3=db3, gamma1=dgamma1,
            gamma2=dgamma2, beta1=dbeta1, beta2=dbeta2
        )

        return grad

    def _init_model(self, D, C, H):
        self.model = dict(
            W1=np.random.randn(D, H) / np.sqrt(D / 2.),
            W2=np.random.randn(H, H) / np.sqrt(H / 2.),
            W3=np.random.randn(H, C) / np.sqrt(H / 2.),
            b1=np.zeros((1, H)),
            b2=np.zeros((1, H)),
            b3=np.zeros((1, C)),
            gamma1=np.ones((1, H)),
            gamma2=np.ones((1, H)),
            beta1=np.zeros((1, H)),
            beta2=np.zeros((1, H))
        )

        self.bn_caches = dict(
            bn1_mean=np.zeros((1, H)),
            bn2_mean=np.zeros((1, H)),
            bn1_var=np.zeros((1, H)),
            bn2_var=np.zeros((1, H))
        )


class ConvNet(NeuralNet):

    def __init__(self, D, C, H, lam=1e-3, p_dropout=.8, loss='cross_ent', nonlin='relu'):
        super().__init__(D, C, H, lam, p_dropout, loss, nonlin)

    def forward(self, X, train=True):
        gamma1 ,gamma2,gamma3 = self.model['gamma1'],self.model['gamma2'],self.model['gamma3']
        beta1, beta2,beta3 = self.model['beta1'],self.model['beta2'], self.model['beta3']

        u1,u2,u3 = None,None,None
        bn1_cache, pool_cache,bn3_cache= None, None,None
        
        # Conv-1
        #print(X.shape)
        #print(self.model['W1'].shape)
        #print(self.model['b1'].shape)
        h1, h1_cache = l.conv_forward(X, self.model['W1'], self.model['b1'])

        #bn1_cache = (self.bn_caches['bn1_mean'],self.bn_caches['bn1_var'])
        #h1,bn1_cache,run_mean,run_var = l.conv_bn_forward(h1,gamma1,beta1,bn1_cache,train=train)
        h1, nl_cache1 = l.relu_forward(h1)
        #print('h1 shape',h1.shape)
        
        #self.bn_caches['bn1_mean'],self.bn_caches['bn1_var']=run_mean,run_var
        
        #if train:
         #   h1 ,u1 = l.dropout_forward(h1,self.p_dropout)

        
        # Pool-1
        hpool, hpool_cache = l.maxpool_forward(h1)
        #print('hpool shape',hpool.shape)
        #bn2_cache = (self.bn_caches['bn2_mean'],self.bn_caches['bn2_var'])
        #hpool,pool_cache,run_mean,run_var = l.conv_bn_forward(hpool,gamma2,beta2,bn2_cache,train= train)
        h2 = hpool.ravel().reshape(X.shape[0], -1)
        #print('h2 shape',h2.shape)

        #self.bn_caches['bn2_mean'],self.bn_caches['bn2_var'] = run_mean,run_var
        
        #if train:
         #  h2,u2 = l.dropout_forward(h2,self.p_dropout)

        # FC-7
        h3, h3_cache = l.fc_forward(h2, self.model['W2'], self.model['b2'])
        #bn3_cache  = (self.bn_caches['bn3_mean'],self.bn_caches['bn3_var'])
        #h3, bn3_cache, run_mean, run_var = l.bn_forward(h3, gamma3, beta3, bn3_cache, train=train)
        h3, nl_cache3 = l.relu_forward(h3)
        #print('h3 shape',h3.shape)
    
        
        #self.bn_caches['bn3_mean'],self.bn_caches['bn3_var']= run_mean,run_var

        #if train:
        #    h3,u3 = l.dropout_forward(h3,self.p_dropout)

        # Softmax
        # forth layer
        score, score_cache = l.fc_forward(h3, self.model['W3'], self.model['b3'])
        #print('score shape',score.shape)
        #print('score',score)
        #return score, (X, h1_cache,h3_cache, score_cache, hpool_cache, hpool, nl_cache1, nl_cache3,u1,u2,u3,bn1_cache,pool_cache,bn3_cache)

        return score, (X, h1_cache, h3_cache, score_cache, hpool_cache, hpool, nl_cache1, nl_cache3)

    def backward(self, y_pred, y_train, cache):
        
        
        #X, h1_cache, h3_cache, score_cache, hpool_cache, hpool, nl_cache1,nl_cache3,u1,u2,u3,bn1_cache,pool_cache,bn3_cache = cache
        

        X, h1_cache, h3_cache, score_cache, hpool_cache, hpool, nl_cache1, nl_cache3= cache

        # Output layer
        grad_y = self.dloss_funs[self.loss](y_pred, y_train)

        # FC-7
        dh3, dW3, db3 = l.fc_backward(grad_y, score_cache)
        #dW3+=reg.dl2_reg(self.model['W3'],self.lam)
        dh3 = self.backward_nonlin(dh3, nl_cache3)
        #dh3 = l.dropout_backward(dh3,u3)
        #dh3,dgamma3,dbeta3= l.bn_backward(dh3,bn3_cache)
     

        dh2, dW2, db2 = l.fc_backward(dh3, h3_cache)
        #dh2 = l.dropout_backward(dh2,u2)
        dh2 = dh2.ravel().reshape(hpool.shape)
      

        #Pool-1
        #dpool,dgamma2,dbeta2 = l.conv_bn_backward(dh2,pool_cache)
        dpool = l.maxpool_backward(dh2, hpool_cache)
        


        # Conv-1
        dh1 = self.backward_nonlin(dpool, nl_cache1)
        #dX, dW1, db1 = l.conv_backward(dh1, h1_cache)
        #dW1+=reg.dl2_reg(self.model['W1'],self.lam)
        #dh1= l.dropout_backward(dh1,u1)
        #dh1,dgamma1,dbeta1 = l.conv_bn_backward(dh1,bn1_cache)

        dX, dW1, db1 = l.conv_backward(dh1, h1_cache)
        
        
        #grad = dict(W1=dW1, W2=dW2, W3=dW3,b1=db1, b2=db2, b3=db3,gamma1 = dgamma1,beta1 = dbeta1,gamma2 = dgamma2,beta2 = dbeta2,gamma3=dgamma3,beta3=dbeta3)
        
        
        grad = dict(
            W1=dW1, W2=dW2, W3=dW3, b1=db1, b2=db2, b3=db3

        )
        
        return grad

    def _init_model(self, D, C, H):
        self.model = dict(
            W1=np.random.randn(D, 1, 3, 3) / np.sqrt(D / 2.),
            W2=np.random.randn(D * 14 * 14, H) / np.sqrt(D * 14 * 14 / 2.),
            W3=np.random.randn(H, C) / np.sqrt(H / 2.),

            b1=np.zeros((D, 1)),
            b2=np.zeros((1, H)),
            b3=np.zeros((1, C)),
            gamma1 = np.ones((1,C)),
            gamma2 = np.ones((1,C)),
            gamma3 = np.ones((1,H)),
            beta1  = np.zeros((1,C)),
            beta2  = np.zeros((1,C)),
            beta3  = np.zeros((1,H))
        )
        
        self.bn_caches=dict(
            bn1_mean = np.zeros((1,C)),
            bn2_mean = np.zeros((1,C)),
            bn3_mean = np.zeros((1,H)),
            bn1_var  = np.zeros((1,C)),
            bn2_var  = np.zeros((1,C)),
            bn3_var  = np.zeros((1,H))
            )


class ConvNet_SVHN(NeuralNet):

    def __init__(self, D, C, H, lam=1e-3, p_dropout=.8, loss='cross_ent', nonlin='relu'):
        super().__init__(D, C, H, lam, p_dropout, loss, nonlin)

    def forward(self, X, train=False):
        gamma1, gamma3,gamma4 = self.model['gamma1'], self.model['gamma3'],self.model['gamma4']
        beta1, beta3, beta4 = self.model['beta1'], self.model['beta3'],self.model['beta4']

        u1, u2, u3 = None, None, None
        bn1_cache, bn2_cache, bn3_cache,bn4_cache = None, None, None,None

        # Conv-1
        #first layer
        h1, h1_cache = l.conv_forward(X, self.model['W1'], self.model['b1'])

        bn1_cache =(self.bn_caches['bn1_mean'],self.bn_caches['bn1_var'])
        h1, bn1_cache, run_mean, run_var = l.conv_bn_forward(h1, gamma1, beta1, bn1_cache, train=train)

        h1, nl_cache1 = l.relu_forward(h1)
        #print('h1 shape',h1.shape)

        self.bn_caches['bn1_mean'],self.bn_caches['bn1_var'] =run_mean,run_var
        # Pool-1
        hpool1, hpool1_cache = l.maxpool_forward(h1)


        #Conv -2
        # second layer
        h2, h2_cache = l.conv_forward(hpool1, self.model['W2'], self.model['b2'])
        bn2_cache = (self.bn_caches['bn2_mean'], self.bn_caches['bn2_var'])

        h2, bn2_cache, run_mean, run_var = l.conv_bn_forward(h2, gamma2, beta2, bn2_cache, train=train)
        h2, nl_cache2 = l.relu_forward(h2)


        #self.bn_caches['bn2_mean'],self.bn_caches['bn2_var'] =run_mean,run_var
        #Pool- 2
        hpool2, hpool2_cache = l.maxpool_forward(h2)


        #conv -3
        #third layer
        h3, h3_cache = l.conv_forward(hpool2, self.model['W3'], self.model['b3'])
        bn3_cache = (self.bn_caches['bn3_mean'], self.bn_caches['bn3_var'])

        h3, bn3_cache, run_mean, run_var = l.conv_bn_forward(h3, gamma3, beta3, bn3_cache, train=train)
        h3, nl_cache3 = l.relu_forward(h3)

        self.bn_caches['bn3_mean'],self.bn_caches['bn3_var'] =run_mean,run_var
        #pool -3
        hpool3, hpool3_cache = l.maxpool_forward(h3)

        hpool3_ = hpool3.ravel().reshape(X.shape[0],-1)

        # FC-7
        # forth layer
        h4, h4_cache = l.fc_forward(hpool3_, self.model['W4'], self.model['b4'])
        bn4_cache = (self.bn_caches['bn4_mean'], self.bn_caches['bn4_var'])

        h4, bn4_cache, run_mean, run_var = l.bn_forward(h4, gamma4, beta4, bn4_cache, train=train)
        h4, nl_cache4 = l.relu_forward(h4)
        #print('h4 shape',h4.shape)

        self.bn_caches['bn4_mean'],self.bn_caches['bn4_var'] = run_mean,run_var


        # Softmax
        # fifth layer
        score, score_cache = l.fc_forward(h4, self.model['W5'], self.model['b5'])
        #print('score shape',score.shape)
        return score, (X, h1_cache,h2_cache ,h3_cache, h4_cache, score_cache,
                       hpool1_cache, hpool1,hpool2_cache,hpool2, hpool3_cache, hpool3,
                       nl_cache1,nl_cache2 ,nl_cache3, nl_cache4,
                       bn1_cache,bn2_cache,bn3_cache,bn4_cache
                      )

    def backward(self, y_pred, y_train, cache):
        # X, h1_cache, h3_cache, score_cache, hpool_cache, hpool, nl_cache1,nl_cache3,u1,u2,u3,bn1_cache,pool_cache,bn3_cache = cache

        (X, h1_cache, h2_cache, h3_cache, h4_cache, score_cache,
         hpool1_cache, hpool1, hpool2_cache, hpool2,hpool3_cache, hpool3,
         nl_cache1, nl_cache2,nl_cache3, nl_cache4,
         bn1_cache,bn2_cache  ,bn3_cache, bn4_cache) = cache

        # Output layer
        grad_y = self.dloss_funs[self.loss](y_pred, y_train)

        # FC-7
        #print('grad_y shape',grad_y.shape)
        #print('score cache shape',score_cache.shape)
        dh5, dW5, db5 = l.fc_backward(grad_y, score_cache)
        # dW3+=reg.dl2_reg(self.model['W3'],self.lam)
        dh5 = self.backward_nonlin(dh5, nl_cache4)
        #print('dh5 shape{} dW5 shape {} db5 shape {}'.format(dh5.shape,dW5.shape,db5.shape))

        # dh3 = l.dropout_backward(dh3,u3).
        dh5,dgamma4,dbeta4= l.bn_backward(dh5,bn4_cache)
        #print('dh5 shape {} dgamma4 shape {} dbeta4 shape{}'.format(dh5.shape,dgamma4.shape,dbeta4.shape))

        dh4, dW4, db4 = l.fc_backward(dh5, h4_cache)
        # dh2 = l.dropout_backward(dh2,u2)
        dh4 = dh4.ravel().reshape(hpool3.shape)
        #print('hpool3 shape',hpool3.shape)
        #print('dh4 shape{} dW4 shape {} db4 shape {}'.format(dh4.shape, dW4.shape, db4.shape))

        # Pool-3
        # dpool,dgamma2,dbeta2 = l.conv_bn_backward(dh2,pool_cache)
        dpool3 = l.maxpool_backward(dh4, hpool3_cache)
        #print('dpool3 shape',dpool3.shape)

        # Conv-3
        dh3 = self.backward_nonlin(dpool3, nl_cache3)
        dh3,dgamma3,dbeta3 = l.conv_bn_backward(dh3,bn3_cache)
        #print('dh3 shape {} dgamma3 shape {} dbeta3 shape {}'.format(dh3.shape ,dgamma3.shape,dbeta3.shape))
        dh2, dW3, db3 = l.conv_backward(dh3, h3_cache)
        # dW1+=reg.dl2_reg(self.model['W1'],self.lam)
        # dh1= l.dropout_backward(dh1,u1)
        # dh1,dgamma1,dbeta1 = l.conv_bn_backward(dh1,bn1_cache)
        #print('dh2 shape{} dW3 shape {} db3 shape {}'.format(dh2.shape, dW3.shape, db3.shape))

        #Pool -2
        dpool2= l.maxpool_backward(dh2, hpool2_cache)
        #print('dpool2 shape',dpool2.shape)
        #conv -2
        dh2 = self.backward_nonlin(dpool2, nl_cache2)
        dh2,dgamma2,dbeta2 = l.conv_bn_backward(dh2,bn2_cache)
        #print('dh2 shape {} dgamma2 shape {} dbeta2 shape {}'.format(dh2.shape,dgamma3.shape,dbeta2.shape))
        dh1, dW2, db2 = l.conv_backward(dh2, h2_cache)
        #print('dh1 shape{} dW2 shape {} db2 shape {}'.format(dh1.shape, dW2.shape, db2.shape))

        #pool -1
        dpool1 = l.maxpool_backward(dh2, hpool1_cache)
        #print('dpool1 shape',dpool1.shape)
        #conv -1
        dh1 = self.backward_nonlin(dpool1, nl_cache1)
        dh1 ,dgamma1,dbeta1 = l.conv_bn_backward(dh1,bn1_cache)
        #print('dh1 shape {} dgamma1 shape {} dbeta1 shape {}'.format(dh1.shape,dgamma1.shape,dbeta1.shape))
        dX, dW1, db1 = l.conv_backward(dh1, h1_cache)
        # grad = dict(W1=dW1, W2=dW2, W3=dW3,b1=db1, b2=db2, b3=db3,gamma1 = dgamma1,beta1 = dbeta1,gamma2 = dgamma2,beta2 = dbeta2,gamma3=dgamma3,beta3=dbeta3)
        #print('dhX shape{} dW1 shape {} db1 shape {}'.format(dX.shape, dW1.shape, db1.shape))
        grad = dict(
            W1=dW1, W2 = dW2, W3=dW3, W4=dW4,W5=dW5,
            b1=db1, b2 = db2 ,b3=db3, b4=db4,b5=db5
        )

        return grad

    def _init_model(self, D, C, H):

        self.model = dict(
            W1=np.random.randn(D, 3, 3, 3) / np.sqrt(D / 2.),
            W2=np.random.randn(2 * D, D, 3, 3) / np.sqrt(2 * D / 2.),
            W3=np.random.randn(4 * D, 2 * D, 3, 3) / np.sqrt(4 * D / 2.),
            W4=np.random.randn(4 * D * 4 * 4, H) / np.sqrt(4 * D * 4 * 4 / 2.),
            W5=np.random.randn(H, C) / np.sqrt(H / 2.),
            b1=np.zeros((D, 1)),
            b2=np.zeros((2 * D, 1)),
            b3=np.zeros((4 * D, 1)),
            b4=np.zeros((1, H)),
            b5=np.zeros((1, C)),
            gamma1=np.ones((1, D)),
            gamma2=np.ones((1, 2 * D)),
            gamma3=np.ones((1, 4 * D)),
            gamma4=np.ones((1, H)),
            beta1=np.zeros((1, D)),
            beta2=np.zeros((1, 2 * D)),
            beta3=np.zeros((1, 4 * D)),
            beta4=np.zeros((1, H))
        )

        self.bn_caches = dict(
            bn1_mean=np.zeros((1, D)),
            bn2_mean=np.zeros((1, 2 * D)),
            bn3_mean=np.zeros((1, 4 * D)),
            bn4_mean=np.zeros((1, H)),
            bn1_var=np.zeros((1, D)),
            bn2_var=np.zeros((1, 2 * D)),
            bn3_var=np.zeros((1, 4 * D)),
            bn4_var=np.zeros((1, H))
        )


class ConvNet_new(NeuralNet):

    def __init__(self, D, C, H, lam=1e-3, p_dropout=.8, loss='cross_ent', nonlin='relu'):
        super().__init__(D, C, H, lam, p_dropout, loss, nonlin)

    def forward(self, X, train=True):
        gamma1, gamma2, gamma3, gamma4, gamma5 = \
                 self.model['gamma1'],self.model['gamma2'], \
                 self.model['gamma3'],self.model['gamma4'], \
                 self.model['gamma5']
        beta1, beta2, beta3, beta4, beta5 = \
            self.model['beta1'], self.model['beta2'],\
            self.model['beta3'], self.model['beta4'],\
            self.model['beta5']

        u1, u2, u3, u4, u5, u6 = None, None, None,None,None, None
        bn1_cache, bn2_cache, bn3_cache, bn4_cache, bn5_cache = None, None, None,None,None

        '''Convolutional layer - 1'''
        h1, h1_cache = l.conv_forward(X, self.model['W1'], self.model['b1'])
        h1, nl_cache1 = l.relu_forward(h1)

        '''Pool -1'''
        hpool1, hpool1_cache = l.maxpool_forward(h1)

        '''Conv -2'''
        h2, h2_cache = l.conv_forward(hpool1, self.model['W2'], self.model['b2'])
        h2, nl_cache2 = l.relu_forward(h2)

        '''Pool- 2'''
        hpool2, hpool2_cache = l.maxpool_forward(h2)



        '''reshape to Fully-connected layer'''
        hpool2_ = hpool2.ravel().reshape(X.shape[0],-1)

        '''FC -1'''
        h4, h4_cache = l.fc_forward(hpool2_, self.model['W4'], self.model['b4'])
        bn4_cache = (self.bn_caches['bn4_mean'], self.bn_caches['bn4_var'])
        h4, bn4_cache, run_mean, run_var = l.bn_forward(h4, gamma4, beta4, bn4_cache, train=train)
        h4, nl_cache4 = l.relu_forward(h4)
        self.bn_caches['bn4_mean'], self.bn_caches['bn4_var'] = run_mean,run_var


        '''FC -2'''
        h5, h5_cache = l.fc_forward(h4, self.model['W5'], self.model['b5'])
        bn5_cache = (self.bn_caches['bn5_mean'], self.bn_caches['bn5_var'])
        h5, bn5_cache, run_mean, run_var = l.bn_forward(h5, gamma5, beta5, bn5_cache,train=train)
        h5, nl_cache5 = l.relu_forward(h5)
        self.bn_caches['bn5_mean'], self.bn_caches['bn5_var'] = run_mean, run_var


        '''Output layer'''
        score, score_cache = l.fc_forward(h5, self.model['W6'], self.model['b6'])
        return score, (X, h1_cache, h2_cache,  h4_cache, h5_cache, score_cache,
                       hpool1_cache, hpool1, hpool2_cache, hpool2,
                       nl_cache1, nl_cache2,  nl_cache4, nl_cache5,
                        bn4_cache,bn5_cache
                       )

    def backward(self, y_pred, y_train, cache):

        (X, h1_cache, h2_cache,  h4_cache, h5_cache, score_cache,
         hpool1_cache, hpool1, hpool2_cache, hpool2,
         nl_cache1, nl_cache2,  nl_cache4, nl_cache5,
         bn4_cache,bn5_cache

        ) = cache

        '''Output layer'''
        grad_y = self.dloss_funs[self.loss](y_pred, y_train)
        dh5, dW6, db6 = l.fc_backward(grad_y, score_cache)

        '''FC-2'''
        dh5 = self.backward_nonlin(dh5, nl_cache5)
        dh5, dgamma5, dbeta5 = l.bn_backward(dh5, bn5_cache)
        dh4, dW5, db5 = l.fc_backward(dh5, h5_cache)


        '''FC -1'''
        dh4 = self.backward_nonlin(dh4, nl_cache4)
        dh4, dgamma4, dbeta4 = l.bn_backward(dh4,bn4_cache)
        dhpool3_, dW4, db4 = l.fc_backward(dh4, h4_cache)

        '''reshape'''
        dhpool3 = dhpool3_.ravel().reshape(hpool2.shape)


        '''Pool -2'''
        dpool2 = l.maxpool_backward(dhpool3, hpool2_cache)

        '''Conv -2'''
        dh2 = self.backward_nonlin(dpool2, nl_cache2)
        dh1, dW2, db2 = l.conv_backward(dh2, h2_cache)

        '''pool -1'''
        dpool1 = l.maxpool_backward(dh1, hpool1_cache)

        '''conv -1'''
        dh1 = self.backward_nonlin(dpool1, nl_cache1)
        dX, dW1, db1 = l.conv_backward(dh1, h1_cache)

        grad = dict(W1=dW1, W2=dW2,  W4=dW4, W5=dW5, W6=dW6,
                    b1=db1, b2=db2,  b4=db4, b5=db5, b6=db6,
                    gamma4=dgamma4,gamma5=dgamma5,
                    beta4=dbeta4,beta5=dbeta5
                )

        return grad

    def _init_model(self, D, C, H):

        self.model = dict(
            W1=np.random.randn(D, 1, 5, 5) / np.sqrt(D / 2.),
            W2=np.random.randn(2 * D, D, 5, 5) / np.sqrt(2 * D / 2.),
            W3=np.random.randn(4 * D, 2 * D, 5, 5) / np.sqrt(4 * D / 2.),
            W4=np.random.randn(2 * D * 7 * 7, H) / np.sqrt(2 * D * 7 * 7 / 2.),
            W5=np.random.randn(H, 2*H) / np.sqrt(H / 2.),
            W6=np.random.randn(2*H, C) / np.sqrt(H / 2.),
            b1=np.zeros((D, 1)),
            b2=np.zeros((2 * D, 1)),
            b3=np.zeros((4 * D, 1)),
            b4=np.zeros((1, H)),
            b5=np.zeros((1, 2*H)),
            b6=np.zeros((1, C)),
            gamma1=np.ones((1, D)),
            gamma2=np.ones((1, 2 * D)),
            gamma3=np.ones((1, 4 * D)),
            gamma4=np.ones((1, H)),
            gamma5=np.ones((1, 2*H)),
            beta1=np.zeros((1, D)),
            beta2=np.zeros((1, 2 * D)),
            beta3=np.zeros((1, 4 * D)),
            beta4=np.zeros((1, H)),
            beta5=np.zeros((1, 2*H)),
        )

        self.bn_caches = dict(
            bn1_mean=np.zeros((1, D)),
            bn2_mean=np.zeros((1, 2 * D)),
            bn3_mean=np.zeros((1, 4 * D)),
            bn4_mean=np.zeros((1, H)),
            bn5_mean=np.zeros((1, 2*H)),
            bn1_var=np.zeros((1, D)),
            bn2_var=np.zeros((1, 2 * D)),
            bn3_var=np.zeros((1, 4 * D)),
            bn4_var=np.zeros((1, H)),
            bn5_var=np.zeros((1, 2*H)),
        )

class RNN(NeuralNet):

    def __init__(self, D, H, char2idx, idx2char):
        self.D = D
        self.H = H
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.vocab_size = len(char2idx)
        super().__init__(D, D, H, None, None, loss='cross_ent', nonlin='relu')

    def initial_state(self):
        return np.zeros((1, self.H))

    def forward(self, X, h, train=True):
        Wxh, Whh, Why = self.model['Wxh'], self.model['Whh'], self.model['Why']
        bh, by = self.model['bh'], self.model['by']

        X_one_hot = np.zeros(self.D)
        X_one_hot[X] = 1.
        X_one_hot = X_one_hot.reshape(1, -1)

        hprev = h.copy()

        h, h_cache = l.tanh_forward(X_one_hot @ Wxh + hprev @ Whh + bh)
        y, y_cache = l.fc_forward(h, Why, by)

        cache = (X_one_hot, Whh, h, hprev, y, h_cache, y_cache)

        if not train:
            y = util.softmax(y)

        return y, h, cache

    def backward(self, y_pred, y_train, dh_next, cache):
        X, Whh, h, hprev, y, h_cache, y_cache = cache

        # Softmax gradient
        dy = loss_fun.dcross_entropy(y_pred, y_train)

        # Hidden to output gradient
        dh, dWhy, dby = l.fc_backward(dy, y_cache)
        dh += dh_next
        dby = dby.reshape((1, -1))

        # tanh
        dh = l.tanh_backward(dh, h_cache)

        # Hidden gradient
        dbh = dh
        dWhh = hprev.T @ dh
        dWxh = X.T @ dh
        dh_next = dh @ Whh.T

        grad = dict(Wxh=dWxh, Whh=dWhh, Why=dWhy, bh=dbh, by=dby)

        return grad, dh_next

    def train_step(self, X_train, y_train, h):
        ys = []
        caches = []
        loss = 0.

        # Forward
        for x, y in zip(X_train, y_train):
            y_pred, h, cache = self.forward(x, h, train=True)
            loss += loss_fun.cross_entropy(self.model, y_pred, y, lam=0)
            ys.append(y_pred)
            caches.append(cache)

        loss /= X_train.shape[0]

        # Backward
        dh_next = np.zeros((1, self.H))
        grads = {k: np.zeros_like(v) for k, v in self.model.items()}

        for t in reversed(range(len(X_train))):
            grad, dh_next = self.backward(ys[t], y_train[t], dh_next, caches[t])

            for k in grads.keys():
                grads[k] += grad[k]

        for k, v in grads.items():
            grads[k] = np.clip(v, -5., 5.)

        return grads, loss, h

    def sample(self, X_seed, h, size=100):
        chars = [self.idx2char[X_seed]]
        idx_list = list(range(self.vocab_size))
        X = X_seed

        for _ in range(size - 1):
            prob, h, _ = self.forward(X, h, train=False)
            idx = np.random.choice(idx_list, p=prob.ravel())
            chars.append(self.idx2char[idx])
            X = idx

        return ''.join(chars)

    def _init_model(self, D, C, H):
        self.model = dict(
            Wxh=np.random.randn(D, H) / np.sqrt(D / 2.),
            Whh=np.random.randn(H, H) / np.sqrt(H / 2.),
            Why=np.random.randn(H, D) / np.sqrt(C / 2.),
            bh=np.zeros((1, H)),
            by=np.zeros((1, D))
        )


class LSTM(RNN):

    def __init__(self, D, H, char2idx, idx2char):
        super().__init__(D, H, char2idx, idx2char)

    def initial_state(self):
        return (np.zeros((1, self.H)), np.zeros((1, self.H)))

    def forward(self, X, state, train=True):
        m = self.model
        Wf, Wi, Wc, Wo, Wy = m['Wf'], m['Wi'], m['Wc'], m['Wo'], m['Wy']
        bf, bi, bc, bo, by = m['bf'], m['bi'], m['bc'], m['bo'], m['by']

        h_old, c_old = state

        X_one_hot = np.zeros(self.D)
        X_one_hot[X] = 1.
        X_one_hot = X_one_hot.reshape(1, -1)

        X = np.column_stack((h_old, X_one_hot))

        hf, hf_cache = l.fc_forward(X, Wf, bf)
        hf, hf_sigm_cache = l.sigmoid_forward(hf)

        hi, hi_cache = l.fc_forward(X, Wi, bi)
        hi, hi_sigm_cache = l.sigmoid_forward(hi)

        ho, ho_cache = l.fc_forward(X, Wo, bo)
        ho, ho_sigm_cache = l.sigmoid_forward(ho)

        hc, hc_cache = l.fc_forward(X, Wc, bc)
        hc, hc_tanh_cache = l.tanh_forward(hc)

        c = hf * c_old + hi * hc
        c, c_tanh_cache = l.tanh_forward(c)

        h = ho * c

        y, y_cache = l.fc_forward(h, Wy, by)

        cache = (
            X, hf, hi, ho, hc, hf_cache, hf_sigm_cache, hi_cache, hi_sigm_cache, ho_cache,
            ho_sigm_cache, hc_cache, hc_tanh_cache, c_old, c, c_tanh_cache, y_cache
        )

        if not train:
            y = util.softmax(y)

        return y, (h, c), cache

    def backward(self, y_pred, y_train, d_next, cache):
        X, hf, hi, ho, hc, hf_cache, hf_sigm_cache, hi_cache, hi_sigm_cache, ho_cache, ho_sigm_cache, hc_cache, hc_tanh_cache, c_old, c, c_tanh_cache, y_cache = cache
        dh_next, dc_next = d_next

        dy = loss_fun.dcross_entropy(y_pred, y_train)

        dh, dWy, dby = l.fc_backward(dy, y_cache)
        dh += dh_next

        dho = c * dh
        dho = l.sigmoid_backward(dho, ho_sigm_cache)

        dc = ho * dh
        dc = l.tanh_backward(dc, c_tanh_cache)
        dc = dc + dc_next

        dhf = c_old * dc
        dhf = l.sigmoid_backward(dhf, hf_sigm_cache)

        dhi = hc * dc
        dhi = l.sigmoid_backward(dhi, hi_sigm_cache)

        dhc = hi * dc
        dhc = l.tanh_backward(dhc, hc_tanh_cache)

        dXo, dWo, dbo = l.fc_backward(dho, ho_cache)
        dXc, dWc, dbc = l.fc_backward(dhc, hc_cache)
        dXi, dWi, dbi = l.fc_backward(dhi, hi_cache)
        dXf, dWf, dbf = l.fc_backward(dhf, hf_cache)

        dX = dXo + dXc + dXi + dXf
        dh_next = dX[:, :self.H]
        dc_next = hf * dc

        grad = dict(Wf=dWf, Wi=dWi, Wc=dWc, Wo=dWo, Wy=dWy, bf=dbf, bi=dbi, bc=dbc, bo=dbo, by=dby)

        return grad, (dh_next, dc_next)

    def train_step(self, X_train, y_train, state):
        y_preds = []
        caches = []
        loss = 0.

        # Forward
        for x, y_true in zip(X_train, y_train):
            y, state, cache = self.forward(x, state, train=True)
            loss += loss_fun.cross_entropy(self.model, y, y_true, lam=0)

            y_preds.append(y)
            caches.append(cache)

        loss /= X_train.shape[0]

        # Backward
        dh_next = np.zeros((1, self.H))
        dc_next = np.zeros((1, self.H))
        d_next = (dh_next, dc_next)

        grads = {k: np.zeros_like(v) for k, v in self.model.items()}

        for y_pred, y_true, cache in reversed(list(zip(y_preds, y_train, caches))):
            grad, d_next = self.backward(y_pred, y_true, d_next, cache)

            for k in grads.keys():
                grads[k] += grad[k]

        for k, v in grads.items():
            grads[k] = np.clip(v, -5., 5.)

        return grads, loss, state

    def _init_model(self, D, C, H):
        Z = H + D

        self.model = dict(
            Wf=np.random.randn(Z, H) / np.sqrt(Z / 2.),
            Wi=np.random.randn(Z, H) / np.sqrt(Z / 2.),
            Wc=np.random.randn(Z, H) / np.sqrt(Z / 2.),
            Wo=np.random.randn(Z, H) / np.sqrt(Z / 2.),
            Wy=np.random.randn(H, D) / np.sqrt(D / 2.),
            bf=np.zeros((1, H)),
            bi=np.zeros((1, H)),
            bc=np.zeros((1, H)),
            bo=np.zeros((1, H)),
            by=np.zeros((1, D))
        )


class GRU(RNN):

    def __init__(self, D, H, char2idx, idx2char):
        super().__init__(D, H, char2idx, idx2char)

    def forward(self, X, h_old, train=True):
        m = self.model
        Wz, Wr, Wh, Wy = m['Wz'], m['Wr'], m['Wh'], m['Wy']
        bz, br, bh, by = m['bz'], m['br'], m['bh'], m['by']

        X_one_hot = np.zeros(self.D)
        X_one_hot[X] = 1.
        X_one_hot = X_one_hot.reshape(1, -1)

        X = np.column_stack((h_old, X_one_hot))

        hz, hz_cache = l.fc_forward(X, Wz, bz)
        hz, hz_sigm_cache = l.sigmoid_forward(hz)

        hr, hr_cache = l.fc_forward(X, Wr, br)
        hr, hr_sigm_cache = l.sigmoid_forward(hr)

        X_prime = np.column_stack((hr * h_old, X_one_hot))
        hh, hh_cache = l.fc_forward(X_prime, Wh, bh)
        hh, hh_tanh_cache = l.tanh_forward(hh)

        h = (1. - hz) * h_old + hz * hh

        y, y_cache = l.fc_forward(h, Wy, by)

        cache = (
            X, X_prime, h_old, hz, hz_cache, hz_sigm_cache, hr, hr_cache, hr_sigm_cache,
            hh, hh_cache, hh_tanh_cache, h, y_cache
        )

        if not train:
            y = util.softmax(y)

        return y, h, cache

    def backward(self, y_pred, y_train, dh_next, cache):
        X, X_prime, h_old, hz, hz_cache, hz_sigm_cache, hr, hr_cache, hr_sigm_cache, hh, hh_cache, hh_tanh_cache, h, y_cache = cache

        dy = loss_fun.dcross_entropy(y_pred, y_train)

        dh, dWy, dby = l.fc_backward(dy, y_cache)
        dh += dh_next

        dhh = hz * dh
        dh_old1 = (1. - hz) * dh
        dhz = hh * dh - h_old * dh

        dhh = l.tanh_backward(dhh, hh_tanh_cache)
        dX_prime, dWh, dbh = l.fc_backward(dhh, hh_cache)

        dh_prime = dX_prime[:, :self.H]
        dh_old2 = hr * dh_prime

        dhr = h_old * dh_prime
        dhr = l.sigmoid_backward(dhr, hr_sigm_cache)
        dXr, dWr, dbr = l.fc_backward(dhr, hr_cache)

        dhz = l.sigmoid_backward(dhz, hz_sigm_cache)
        dXz, dWz, dbz = l.fc_backward(dhz, hz_cache)

        dX = dXr + dXz
        dh_old3 = dX[:, :self.H]

        dh_next = dh_old1 + dh_old2 + dh_old3

        grad = dict(Wz=dWz, Wr=dWr, Wh=dWh, Wy=dWy, bz=dbz, br=dbr, bh=dbh, by=dby)

        return grad, dh_next

    def _init_model(self, D, C, H):
        Z = H + D

        self.model = dict(
            Wz=np.random.randn(Z, H) / np.sqrt(Z / 2.),
            Wr=np.random.randn(Z, H) / np.sqrt(Z / 2.),
            Wh=np.random.randn(Z, H) / np.sqrt(Z / 2.),
            Wy=np.random.randn(H, D) / np.sqrt(D / 2.),
            bz=np.zeros((1, H)),
            br=np.zeros((1, H)),
            bh=np.zeros((1, H)),
            by=np.zeros((1, D))
        )
