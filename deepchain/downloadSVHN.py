from __future__ import  absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import  tarfile
import  os
import numpy
from six.moves import urllib
import subprocess

def check_file(datadir):
    if os.path.exists(datadir):
        pass
    else:
        os.mkdir(datadir)
        return

def svhn_download(data_dir):
    import scipy.io as sio
    # svhn file loader
    def svhn_loader(url, path):
        cmd = ['curl', url, '-o', path]
        subprocess.call(cmd)
        m = sio.loadmat(path)
        return m['X'], m['y']

    if check_file(data_dir):
        print('SVHN was downloaded.')
    else:
        #data_url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
        #train_image, train_label = svhn_loader(data_url, os.path.join(data_dir, 'train_32x32.mat'))

        #data_url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
        #test_image, test_label = svhn_loader(data_url, os.path.join(data_dir, 'test_32x32.mat'))

        data_url = 'http://ufldl.stanford.edu/housenumbers/extra_32x32.mat'
        extra_image,extra_label = svhn_loader(data_url,os.path.join(data_dir,'extra_32x32.mat'))


if __name__ == '__main__':
    svhn_download('./SVHN')