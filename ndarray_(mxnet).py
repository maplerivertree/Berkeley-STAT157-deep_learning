#ndarray (mxnet) properties; matrix
""" nd.arrary(), nd.arange(), .shape, len(), .reshape(), .T, indexing """
from mxnet import nd
import numpy as np

"""https://mxnet.incubator.apache.org/versions
/master/tutorials/basic/ndarray.html
https://www.numpy.org/devdocs/reference/
arrays.ndarray.html"""

x = nd.array([[1,2,3], 
			[4,5,6], [3,6,19], [3,4,2]],
			dtype=np.int32)
print(x)
#INDEXing 2nd row, 3rd column
print('print(x[1,2]): ')
print(x[1,2])

#VECTOR-creation
#arange() NOT arrange()
v = nd.arange(9)
print(v)

#Length and Shape
print(len(x))
print(x.shape)

"""Clarity: for an NDarray, 2D, 3D are the 
number of axis (2nd number) from .shape, while
for a vector, n-dim refers to the length."""


#reshape(length, #axis)
X = nd.arange(12).reshape(4,3)
print(X)

#Transpose
print(X.T)

#Tensor; Matrix
X = nd.arange(60).reshape(4,5,3)
print(X)