#autograd
"""https://en.d2l.ai/chapter_crashcourse/autograd.html"""
"""https://en.d2l.ai/chapter_appendix/math.html"""

from mxnet import nd, autograd

x = nd.arange(4).reshape(4, 1)

x.attach_grad()

with autograd.record():
	y = 2 * nd.dot(x.T, x)
print(y)

y.backward()

print(x.grad)
""""""