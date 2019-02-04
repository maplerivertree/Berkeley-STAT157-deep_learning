#sum by dimension, .sum(x, *), .dot() vs. (a * b), mean
from mxnet import nd
# sum by dimention(row, colums, ..)

a = nd.arange(12).reshape(4,3)
print(a)
print(nd.sum(a, 0))
print(nd.sum(a, 1))
print(nd.mean(a))

""""""
a = nd.arange(24).reshape(4,3,2)
print(a)
print(nd.sum(a, 0))
print(nd.sum(a, 1))
print(nd.sum(a, 2))
print(nd.mean(a))

"""""" 

b = nd.ones(5)
c = nd.arange(5) + 1.0
print(b, c, nd.dot(b,c))

""""""
X = nd.arange(12).reshape(3,4)
v = nd.array([1,2,3,5])
#Difference nd.dot & multiplication
print(X, v)
print(nd.dot(X,v), X* v)

"""
[[ 0.  1.  2.  3.]
 [ 4.  5.  6.  7.]
 [ 8.  9. 10. 11.]]


[1. 2. 3. 5.]


[ 23.  67. 111.]


[[ 0.  2.  6. 15.]
 [ 4. 10. 18. 35.]
 [ 8. 18. 30. 55.]]
"""