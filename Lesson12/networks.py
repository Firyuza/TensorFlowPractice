import tensorflow as tf

class Network(tf.keras.models.Model):
    def __init__(self, weight_decay, nrof_classes):
        super(Network, self).__init__()

        self.nrof_classes = nrof_classes

        self.seq_model = tf.keras.models.Sequential()


    def build(self, inputs_shape):
        self.seq_model.add(tf.keras.layers.Conv2D(32, [5, 5],
                                                  activation=tf.nn.relu, name='conv1'))
        self.seq_model.add(tf.keras.layers.MaxPool2D([2, 2], 2,
                                                     name='pool1'))
        self.seq_model.add(tf.keras.layers.Conv2D(64, [5, 5],
                                                  activation=tf.nn.relu, name='conv2'))
        self.seq_model.add(tf.keras.layers.MaxPool2D([2, 2], 2,
                                                     name='pool2'))
        self.seq_model.add(tf.keras.layers.Flatten())
        self.seq_model.add(tf.keras.layers.Dense(self.nrof_classes,
                                                 activation=None, name='fc1'))

        super(Network, self).build(inputs_shape)

        self.built = True

        return

    def call(self, inputs, training=True):
        output = self.seq_model(inputs)

        return output