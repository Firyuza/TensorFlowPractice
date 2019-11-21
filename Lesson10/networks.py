import tensorflow as tf

class Network(tf.keras.models.Model):
    def __init__(self, weight_decay, nrof_classes):
        super(Network, self).__init__()

        self.nrof_classes = nrof_classes

        self.seq_model = tf.keras.models.Sequential()
        # TODO

    def call(self, inputs, training=True):
        output = self.seq_model(inputs)

        return output