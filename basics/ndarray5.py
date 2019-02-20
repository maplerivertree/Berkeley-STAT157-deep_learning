#ndarray5
from mxnet import nd
# broadcasting, indexing, slicing ★
"""Broadcasting is not easy to understand
It can be used to genrate large object without for loops"""


#broadcasting
a = nd.arange(3).reshape(3,1) # 3x1 
b = nd.arange(2).reshape(1,2) # 1x2
c = a + b  				  # 3x2
"""
a = [[0.]
 	[1.]
 	[2.]]

b = [[0. 1.]]

c = [[0. 1.]
 	[1. 2.]
 	[2. 3.]]
 """

#indexing
a = nd.arange(20).reshape(5,4)
b = a[1, 3]
print(a, b)
"""
a = [[ 0.  1.  2.  3.]
 	[ 4.  5.  6.  7.]
 	[ 8.  9. 10. 11.]
 	[12. 13. 14. 15.]
 	[16. 17. 18. 19.]]

b = [7.]
"""

#SLICING
#slicing columns
print(a[:, 1]) 
#slicing rows
print(a[2, :])

#with slicing, we can fill a matrix C
C = nd.zeros((5, 4)) # att: double brackets
num_col = a.shape[1] # MEASURE num of columns
for i in range(1, num_col+1):
	C[:, i-1] = a[:, i-1]**2
print(C)
