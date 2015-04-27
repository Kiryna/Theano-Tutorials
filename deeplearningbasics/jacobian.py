import theano
import theano.tensor as T
import theano.gradient as grad
from theano import function

x = T.vector('x')
y = x ** 2
J, updates = theano.scan(lambda i, y, x: T.grad(y[i], x), sequences=T.arange(y.shape[0]), non_sequences=[y, x])
f = function([x], J, updates=updates)
print f([4, 4])

f_grad = function([x], grad.jacobian(y, x))
print f_grad([4, 4])
