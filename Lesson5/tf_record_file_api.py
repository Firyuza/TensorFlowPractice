import tensorflow as tf
import os
import numpy as np
import cv2

tf.compat.v1.enable_eager_execution()

class TFRecordFile:
    def __init__(self, filename):
        self.feature_description = {
            'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'feature1': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
            'feature2': tf.io.FixedLenFeature([], tf.string, default_value='')
        }

        self.filename = filename

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def serialize_example(self, feature0, feature1, feature2):
        """
        Creates a tf.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        feature = {
            'feature0': self._int64_feature(feature0),
            'feature1': self._float_feature(feature1),
            'feature2': self._bytes_feature(feature2)
        }

        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

        serialized_data = example_proto.SerializeToString()

        return serialized_data

    def _parse_function(self, example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, self.feature_description)

    def create_file(self, feature0, feature1, feature2):
        features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2))

        def generator():
            for features in features_dataset:
                yield self.serialize_example(*features)

        serialized_features_dataset = tf.data.Dataset.from_generator(
            generator, output_types=tf.string, output_shapes=())

        writer = tf.data.experimental.TFRecordWriter(self.filename)
        writer.write(serialized_features_dataset)

        filenames = [self.filename]
        raw_dataset = tf.data.TFRecordDataset(filenames)
        parsed_dataset = raw_dataset.map(self._parse_function)

        for raw_record in parsed_dataset.take(3):
            print(raw_record)

        return

data = {}
data['feature0'] = [1, 2, 3]
data['feature1'] = [1., 2., 3.]
data['feature2'] = ['name1', 'name2', 'name3']

tf_record_obj = TFRecordFile('./test.tfrecord')
tf_record_obj.create_file(data['feature0'],
                          data['feature1'],
                          data['feature2'])