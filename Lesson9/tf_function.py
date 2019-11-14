import tensorflow as tf

v = tf.Variable(1.0)
@tf.function
def f():
  v.assign(2.0)
  return v.read_value()

print(f()) # Always prints 2.0.

def foo(x):
    if x > 0:
      y = x * x
    else:
      y = -x
    return y

converted_foo = tf.autograph.to_graph(foo)

x = tf.constant(1)
y = converted_foo(x)  # converted_foo is a TensorFlow Op-like.
assert tf.is_tensor(y)

@tf.function
def f(x):
  if x > 0:
    x = x + 1
  return x

tf.config.experimental_run_functions_eagerly(True)
print(f(tf.constant(1))) # tf.Tensor(2, shape=(), dtype=int32)