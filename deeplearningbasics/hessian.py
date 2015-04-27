import theano
import theano.tensor as T
import theano.gradient as grad
from theano import  function

x = T.dvector('x')
y = x ** 2
cost = y.sum()
gy = T.grad(cost, x)
H, updates = theano.scan(lambda i, gy, x: T.grad(gy[i], x), sequences=T.arange(gy.shape[0]), non_sequences=[gy, x])
f = function([x], H, updates=updates)
print f([4, 4])

f_grad = function([x], grad.hessian(cost, x))
print f_grad([4, 4])
