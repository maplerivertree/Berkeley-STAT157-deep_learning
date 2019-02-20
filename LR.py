#Linear Regression + Basic_Optimization

# import
"""
https://stackoverflow.com/questions/43027980/purpose-of-matplotlib-inline/43028034
https://matplotlib.org/tutorials/index.html
"""
from IPython import display
from matplotlib import pyplot as plt
from mxnet import nd, autograd
#%matplotlib inline 
import random



# data-set generation
"""
use ground truth w = [2, -3.4]^T and bias = 4.2
generate label by y = Xw +b +e with noise e; e obeys a Gaussian  
distribution with a mean of 0 and a standard deviation of 0.01;
https://en.wikipedia.org/wiki/Ground_truth
"""
num_feature, num_traningset = 2, 1000
w = nd.array([2, -3.4])
b = 4.2

X = nd.random.normal(scale = 1.0, shape = (num_traningset, num_feature))
"""scale: Standard deviation of the distribution."""
y = nd.dot(X, w) + b
""" adding e below """
y += nd.random.normal(scale = 0.01, shape=y.shape)
print("len(X) =" + str(len(X)))


# visualize;
"""
https://matplotlib.org/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py
https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot
"""
display.set_matplotlib_formats('svg')
plt.figure(figsize=(6, 3))
#visualize 2nd data-set, and y; markersize =3
plt.scatter(X[:, 1].asnumpy(), y.asnumpy(), 3);



