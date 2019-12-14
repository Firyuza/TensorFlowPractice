import tensorflow as tf

def loss_fn():
    return
def other_loss_fn():
    return

with tf.GradientTape() as t:
  loss = loss_fn()
with tf.GradientTape() as t:
  loss += other_loss_fn()
t.gradient(loss, ...)

