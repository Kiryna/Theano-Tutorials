from theano.tensor.shared_randomstreams import RandomStreams
from theano import function

srng = RandomStreams(seed=234)
rv_u = srng.uniform((2, 2))
rv_n = srng.normal((2, 2))

f = function([], rv_u)
g = function([], rv_n)
nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)

f_val0 = f()
f_val1 = f()

g_val0 = g()
g_val1 = g()

print f_val0
print f_val1
print g_val0
print g_val1
print nearly_zeros()

rng_val = rv_u.rng.get_value(borrow=True)
rng_val.seed(89234)
rv_u.rng.set_value(rng_val, borrow=True)
print f()
print

srng.seed(900890)
print f()
print g()
print

state_after_v0 = rv_u.rng.get_value().get_state()
print nearly_zeros()
print f()

rng = rv_u.rng.get_value(borrow=True)
rng.set_state(state_after_v0)
rv_u.rng.set_value(rng, borrow=True)
print f()
print f()
print f()
print

rng.set_state(state_after_v0)
rv_u.rng.set_value(rng, borrow=True)
print f()
print f()
print f()

