import theano.tensor as T
from theano import function
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)

print f(2, 3)
print f(16.3, 12.1)

print z.eval({x : 2, y : 3})