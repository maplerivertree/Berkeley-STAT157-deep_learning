# nd.random*, nd.ones_like(), nd.concat(a, b, sim = *)

from mxnet import nd
# Note the 2 brackets after zeros instead of 1
print(nd.zeros((3, 4)))
print(nd.zeros((3, 4, 5)))

print(nd.ones((2,3)))

# create tensors with random numberss
x = nd.random.normal(0, 1, shape = (3,4))
print(x)

a = nd.array([1, 3, 4, 5])
print('a= ' + str(a))
# nd.ones_like()

b = nd.ones_like(a)
print('b =' + str(b))
a_b = nd.concat(a, b, dim = 0)
print('nd.concat(a, b, dim = 0) = ' + str(a_b))

a = nd.arange(12).reshape(3, 4)
b = nd.arange(12).reshape(3, 4)
c = nd.concat(a, b, dim = 0)
d = nd.concat(a, b, dim = 1)
print(a , b, c, d)


# concatenation dim = *, follows the same dimension of the tensor

a = nd.arange(24).reshape(2, 3, 4)
b = nd.arange(24).reshape(2, 3, 4)
print('\nxxxx')
c = nd.concat(a, b, dim = 0)
d = nd.concat(a, b, dim = 1)
e = nd.concat(a, b, dim = 2)
print(a, b, c, d, e)

""" Results
a = 
[[[ 0.  1.  2.  3.]
  [ 4.  5.  6.  7.]
  [ 8.  9. 10. 11.]]

 [[12. 13. 14. 15.]
  [16. 17. 18. 19.]
  [20. 21. 22. 23.]]]
<NDArray 2x3x4 @cpu(0)>
b =
[[[ 0.  1.  2.  3.]
  [ 4.  5.  6.  7.]
  [ 8.  9. 10. 11.]]

 [[12. 13. 14. 15.]
  [16. 17. 18. 19.]
  [20. 21. 22. 23.]]]
<NDArray 2x3x4 @cpu(0)>
dim = 0 
[[[ 0.  1.  2.  3.]
  [ 4.  5.  6.  7.]
  [ 8.  9. 10. 11.]]

 [[12. 13. 14. 15.]
  [16. 17. 18. 19.]
  [20. 21. 22. 23.]]

 [[ 0.  1.  2.  3.]
  [ 4.  5.  6.  7.]
  [ 8.  9. 10. 11.]]

 [[12. 13. 14. 15.]
  [16. 17. 18. 19.]
  [20. 21. 22. 23.]]]
<NDArray 4x3x4 @cpu(0)>
dim = 1 
[[[ 0.  1.  2.  3.]
  [ 4.  5.  6.  7.]
  [ 8.  9. 10. 11.]
  [ 0.  1.  2.  3.]
  [ 4.  5.  6.  7.]
  [ 8.  9. 10. 11.]]

 [[12. 13. 14. 15.]
  [16. 17. 18. 19.]
  [20. 21. 22. 23.]
  [12. 13. 14. 15.]
  [16. 17. 18. 19.]
  [20. 21. 22. 23.]]]
<NDArray 2x6x4 @cpu(0)>
dim =2 
[[[ 0.  1.  2.  3.  0.  1.  2.  3.]
  [ 4.  5.  6.  7.  4.  5.  6.  7.]
  [ 8.  9. 10. 11.  8.  9. 10. 11.]]

 [[12. 13. 14. 15. 12. 13. 14. 15.]
  [16. 17. 18. 19. 16. 17. 18. 19.]
  [20. 21. 22. 23. 20. 21. 22. 23.]]] 

  """