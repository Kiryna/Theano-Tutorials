import theano.tensor as T
from theano import function

x = T.matrix('x')
s = 1 / (1 + T.exp(-x))
logistic = function([x], s)

s2 = (1 + T.tanh(x / 2)) / 2
logistic2 = function([x], s2)

print logistic([[0, 1], [-1, -2]])
print logistic2([[0, 1], [-1, -2]])