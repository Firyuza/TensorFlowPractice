import tensorflow as tf

class CustomSparseCategoricalAccuracy(tf.keras.metrics.Metric):
     def __init__(self, name, **kwargs):
         super(CustomSparseCategoricalAccuracy, self).__init__(name=name, **kwargs)

         self.categorical_accuracy = self.add_weight(name='positive', initializer='zeros', dtype=tf.int32)
         self.nrof_elements = self.add_weight(name='positive_nrof_elements', initializer='zeros', dtype=tf.int32)

     def update_state(self, true_labels, predicted_labels):
         # TODO
         value = None

         self.nrof_elements.assign_add(len(true_labels))

         return self.categorical_accuracy.assign_add(value)

     def result(self):
        return tf.identity(tf.divide(self.categorical_accuracy, self.nrof_elements))

     def reset_states(self):
         # The state of the metric will be reset at the start of each epoch.
         self.categorical_accuracy.assign(0)
         self.nrof_elements.assign(0)