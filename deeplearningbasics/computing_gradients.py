import theano.tensor as T
from theano import pp
from theano import function

x = T.dscalar('x')
y = T.dscalar('y')
z = T.dscalar('y')
mult_var_func = x ** 3 + y ** 2 + z
grad = T.grad(mult_var_func, [x, y, z])  # [3 * x ** 2, 2 * y, 1]

f = function([x, y, z], grad)
print f(1, 2, 3)
print f(1, 1, 1)

print '[', pp(f.maker.fgraph.outputs[0]), ', ', pp(f.maker.fgraph.outputs[1]), ', ', pp(f.maker.fgraph.outputs[2]), ']'

a = T.dmatrix('a')
s = T.sum(1 / (1 + T.exp(-a)))
gs = T.grad(s, a)
dlogistic = function([a], gs)
print dlogistic([[0, 1], [-1, -2]])