import theano.tensor as T
from theano import function

x = T.vector('x')
v = T.vector('v')
y = T.sum(x ** 2)
gy = T.grad(y, x)
vH = T.grad(T.sum(gy * v), x)

f = function([x, v], vH)
print f([4, 4], [2, 2])
print

Hv = T.Rop(gy, x, v)
f_Rop = function([x, v], Hv)
print f([4, 4], [2, 2])
