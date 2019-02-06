#ndarray5
# broadcasting, indexing, slicing
"""Broadcasting is not easy to understand
It can be used to genrate large object without for loops"""
from mxnet import nd

#broadcasting
a = nd.arange(3).reshape(3,1) # 3x1 
b = nd.arange(2).reshape(1,2) # 1x2
c = a + b  					  # 3x2
print(a, b, c)

#indexing and slicing
a = nd.arange(20).reshape(5,4)
b = a[1, 3]
c = a[1:2]    #contains the starting, doe not contain ending
print(a, b, c)
"""
[[ 0.  1.  2.  3.]
 [ 4.  5.  6.  7.]
 [ 8.  9. 10. 11.]
 [12. 13. 14. 15.]
 [16. 17. 18. 19.]]

[7.]

[[4. 5. 6. 7.]]
"""

