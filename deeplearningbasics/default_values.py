import theano.tensor as T
from theano import function
from theano import Param

x, y = T.scalars('x', 'y')
z = x + y
f = function([x, Param(y, default=1)], z)

print f(33)
print f(33, 2)

w = T.scalar('w')
z_two = (x + y) * w
f_two = function([x, Param(y, default=1), Param(w, default=2, name='w_by_name')], z_two)

print("Second function")
print f_two(33)
print f_two(33, 2)
print f_two(33, 0, 1)
print f_two(33, w_by_name=1)
print f_two(33, w_by_name=1, y=0)

