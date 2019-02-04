#ndarray nerual network .dot logic
from mxnet import nd
a =(nd.arange(15).reshape(3,5)).T
print(a)
""" This is a 3-layer neural network, 0,1,2,3,4,5 are the params
for the 5 nods on layer 1.
a = 
[[ 0.  5. 10.]
 [ 1.  6. 11.]
 [ 2.  7. 12.]
 [ 3.  8. 13.]
 [ 4.  9. 14.]]
 """

x = (nd.arange(5)+1) 
""" This is the x input data set"""
print(x)

"""
x= [1. 2. 3. 4. 5.]

'"""


# To calculate y hat
y_hat = nd.dot(a.T, x)
print('y_hat' + str(y_hat))

a =(nd.arange(15).reshape(3,5))
print(a)


