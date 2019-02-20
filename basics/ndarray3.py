#ndarray nerual network .dot logic
from mxnet import nd
A =(nd.arange(15).reshape(3,5)).T
print(A)
""" This is a 3-layer neural network, 0,1,2,3,4,5 are the params
for the 5 nods on layer 1.
A = 
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
y_hat = nd.dot(A.T, x)
print('y_hat' + str(y_hat))

a =(nd.arange(15).reshape(3,5))
print(a)

""" x as a matrix (multiple datasets)"""
X = nd.arange(50).reshape(5,10)
print(X)

""" 
A = 
[[ 0.  5. 10.]
 [ 1.  6. 11.]
 [ 2.  7. 12.]
 [ 3.  8. 13.]
 [ 4.  9. 14.]]

A.T=
[[ 0.  1.  2.  3.  4.]
 [ 5.  6.  7.  8.  9.]
 [10. 11. 12. 13. 14.]]

X =
[[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9.]
 [10. 11. 12. 13. 14. 15. 16. 17. 18. 19.]
 [20. 21. 22. 23. 24. 25. 26. 27. 28. 29.]
 [30. 31. 32. 33. 34. 35. 36. 37. 38. 39.]
 [40. 41. 42. 43. 44. 45. 46. 47. 48. 49.]]
 """

# To calculate y hat
y_hat = nd.dot(A.T, X)
print('y_hat' + str(y_hat))
print(A.T)

"""
y_hat=
y_hat
[[ 300.  310.  320.  330.  340.  350.  360.  370.  380.  390.]
 [ 800.  835.  870.  905.  940.  975. 1010. 1045. 1080. 1115.]
 [1300. 1360. 1420. 1480. 1540. 1600. 1660. 1720. 1780. 1840.]]
 """

