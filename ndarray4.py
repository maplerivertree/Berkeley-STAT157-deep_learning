

from mxnet import nd
# Note the 2 brackets after zeros instead of 1
print(nd.zeros((3, 4)))
print(nd.zeros((3, 4, 5)))

print(nd.ones((2,3)))

# create tensors with random numberss
X = nd.random.normal(0, 1, shape = (3,4))
print(X)

