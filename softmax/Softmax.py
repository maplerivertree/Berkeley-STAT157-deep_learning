# softmax regression

# import
from mxnet import nd, autograd
#%matplotlib inline




# use Fashion-MNIST
"""
Refer to 'Fashion_MNIST --> fashion_mnist_help.py'
https://github.com/zalandoresearch/fashion-mnist
http://colah.github.io/posts/2014-10-Visualizing-MNIST/
https://mxnet.incubator.apache.org/api/python/gluon/data.html#mxnet.gluon.data.vision.datasets.FashionMNIST
""" 
import numpy
import mnist_reader
"""local files located in path data/fashion"""
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
#print(X_train[100])
print("shpae of X_train= ", X_train.shape,"\nshape of y_train= ", y_train.shape)

batch_size = int(input("batch_size = "))
X_train = X_train[0:batch_size]
y_train = y_train[0:batch_size]
X_test = X_test[0:batch_size]
X_test = X_test[0:batch_size]
print("shpae of X_train= ", X_train.shape,"\nshape of y_train= ", y_train.shape)

