import theano.tensor as T
from theano import function
from theano import shared

state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state + inc)])

print state.get_value()
print accumulator(1)

print state.get_value()
print accumulator(300)

print state.get_value()

state.set_value(-1)
print state.get_value()

print accumulator(3)
print state.get_value()

print("\nDecrementor")
decrementor = function([inc], state, updates=[(state, state - inc)])
print decrementor(2)
print state.get_value()

fn_of_state = state * 2 + inc
foo = T.scalar(dtype=state.dtype)
skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)])

print("\nGivens")
print skip_shared(1, 3)
print state.get_value()