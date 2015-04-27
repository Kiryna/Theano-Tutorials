import theano.tensor as T
from theano import pp
from theano import function

x = T.dscalar('x')
y = (x + 5) / (x ** 2 + 1)
gy = T.grad(y, x)
print pp(gy)

f = function([x], gy)
print f(4)
print f(94.2)

print pp(f.maker.fgraph.outputs[0])

a = T.dmatrix('a')
s = T.sum(1 / (1 + T.exp(-a)))
gs = T.grad(s, a)
dlogistic = function([a], gs)
print dlogistic([[0, 1], [-1, -2]])