from mxnet import nd
import random


features = nd.arange(150).reshape(30, 5)
print("features= " + str(features))
labels = nd.arange(30).reshape(30,1)
print("labels= " + str(labels))


"""STARTS here"""
def data_iter(batch_size, features, labels):
	num_exampels = len(features)
	indices = list(range(num_exampels))
	random.shuffle(indices)
	print("shuffled_indices= " + str(indices))

	for i in range(0, num_exampels, batch_size):
		j = nd.array(indices[i : min(i + batch_size, num_exampels)])
		yield features.take(j), labels.take(j) #ref line 31


#print this selected data batch
batch_size = int(input("what's batch_size= "))
for X, y in data_iter(batch_size, features, labels):
	print(X, y)
	break



"""How '.take()' works"""
test = nd.array([1,2,4])
print(features.take(test))
