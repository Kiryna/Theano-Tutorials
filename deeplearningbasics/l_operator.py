import theano.tensor as T
from theano import function

W = T.dmatrix('W')
v = T.dvector('v')
x = T.dvector('x')
y = T.dot(x, W)

VJ = T.Lop(y, W, v)
f = function([v, x], VJ)
print f([2, 2], [0, 1])