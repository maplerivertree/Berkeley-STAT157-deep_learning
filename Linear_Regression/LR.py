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
w_true = nd.array([2, -3.4])
b_true = 4.2

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



# iterate over training_set, and return batch_size
"""ref LR-help1.py for more info"""
def data_iter(batch_size, X, y):
	num_trainingset = len(X)
	indices = list(range(num_trainingset))
	random.shuffle(indices)
	for i in range(0, num_trainingset, batch_size):
		j = nd.array(indices[i: min(i + batch_size, num_trainingset)])
		yield X.take(j), y.take(j)

batch_size = int(input('\nType in batch size= '))
""" print out a batch """
for X_batch, y_batch in data_iter(batch_size, X, y):
	print("This is a sample batch")
	print("X_batch= " + str(X_batch), "y_batch= " + str(y_batch))
	break



# initialize hyperparameters, w, b, 
w = nd.random.normal(scale = 0.01, shape = (num_feature, 1))
b = nd.random.normal(scale = 0.01, shape = (1, ))
print("w= " + str(w), "b= " + str(b))



# attach gradients to hyperparameters, w, b, 
"""
https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/autograd.html
https://en.d2l.ai/chapter_crashcourse/autograd.html
"""
w.attach_grad()
b.attach_grad()



# define linear regression model
# define loss function
# define learning algo
"""
https://en.wikipedia.org/wiki/Stochastic_gradient_descent
if not familiar, refer to 
https://www.coursera.org/learn/machine-learning
https://www.deeplearning.ai/deep-learning-specialization

"""
def lin_reg(X, w, b):
	y_hat = nd.dot(X, w) + b
	return y_hat

"""use reshape to ensure y fits y_hat shape"""
def loss_f(y_hat, y):
	loss = (y_hat - y.reshape(y_hat.shape))**2/2 
	return loss


def sgd(hyperparams, learning_r, batch_size):
	for i in hyperparams:
		i[:] = i - learning_r * i.grad / batch_size



# training
num_inter = int(input("Type in number of interations for this training: ")) or 8
batch_size = int(input("Type in the batch size for this training: ")) or 20
learning_r = float(input("Type in the learning rate for this training: ")) or 0.05
for epoch in range(num_inter):
	for X_batch, y_batch in data_iter(batch_size, X, y):
		with autograd.record():
			loss = loss_f(lin_reg(X_batch, w, b), y_batch)
		loss.backward()
		sgd([w, b], learning_r, batch_size)
	new_loss = loss_f(lin_reg(X_batch, w, b), y_batch)
	print(epoch, new_loss.mean().asnumpy)

print(w, b)
print("discrepency in w", w_true - w.reshape(w_true.shape))
print("discrepency in b", b_true - b)
