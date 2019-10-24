import tensorflow as tf

tf.enable_eager_execution()

class Network(tf.keras.models.Model):
    def __init__(self, nrof_classes):
        super(Network, self).__init__()
        self.base_model = tf.keras.applications.resnet50.ResNet50
        self.logits = tf.keras.layers.Dense(nrof_classes, name='dense_classification')

    def call(self, inputs):
        base_model_output = self.base_model(input_tensor=inputs,
                            weights='imagenet',
                            include_top=False)
        output = tf.keras.layers.GlobalAveragePooling2D()(base_model_output.output)

        return self.logits(output)