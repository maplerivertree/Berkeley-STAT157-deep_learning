# Fashion-MNIST

# METHOD 1: import from local data with mnist_reader

import numpy
import mnist_reader
"""local files located in path data/fashion"""
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
#print(X_train[100])
print(X_train.shape)
print(y_train.shape)



# METHOD 2: import via gluon API
from mxnet.gluon import data as gdata
mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)

X_train, y_train = mnist_train[17188] # extract X, y but indexing from mnist_train
print(X_train.shape)
print(y_train)

def get_fashion_mnist_labels(labels):
	text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
	'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
	return [text_labels[int(labels)]]

print(get_fashion_mnist_labels(y_train))



