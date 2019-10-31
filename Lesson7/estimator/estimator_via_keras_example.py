import tensorflow as tf
import numpy as np

from estimator.config import cfg

class EstimatorViaKerasExample:
    def __init__(self):
        self.__load_data()

        model = self.__create_model()
        # Create a estimator with model_fn
        self.image_classifier = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                                 model_dir=cfg.train.models_base_dir)

        return

    def __load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
        self.y_train = self.y_train.reshape([-1])
        self.y_test = self.y_test.reshape([-1])

        self.nrof_classes = cfg.dataset.nrof_classes

        return

    def __preperare_train_data(self, image, label):
        tf_mean = tf.constant(cfg.train.mu, dtype=tf.float32, shape=[1, 1, 3])
        tf_std = tf.constant(cfg.train.std, dtype=tf.float32, shape=[1, 1, 3])

        image = tf.cast(image, tf.float32)

        image -= tf_mean
        image /= tf_std

        return image, tf.one_hot(label, self.nrof_classes, dtype=tf.int32)

    def __preperare_valid_data(self, image, label):
        tf_mean = tf.constant(cfg.train.mu, dtype=tf.float32, shape=[1, 1, 3])
        tf_std = tf.constant(cfg.train.std, dtype=tf.float32, shape=[1, 1, 3])

        image = tf.cast(image, tf.float32)

        image -= tf_mean
        image /= tf_std

        return image, tf.one_hot(label, self.nrof_classes, dtype=tf.int32)

    def train_input_fn(self, features, labels):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.map(map_func=self.__preperare_train_data,
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size=cfg.train.batch_size)
        dataset = dataset.shuffle(buffer_size=len(features)).\
            prefetch(buffer_size=cfg.train.batch_size)

        return dataset

    def valid_input_fn(self, features, labels):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.map(map_func=self.__preperare_train_data,
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size=cfg.train.batch_size)
        dataset = dataset.shuffle(buffer_size=len(features)).\
            prefetch(buffer_size=cfg.train.batch_size)

        return dataset

    def __create_model(self):
        # create the base pre-trained model
        base_model = tf.keras.applications.resnet50.ResNet50(weights='imagenet',
                                                             include_top=False)

        model = tf.keras.models.Sequential()
        model.add(base_model)
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        softmax = tf.keras.layers.Dense(self.nrof_classes, activation='softmax')
        model.add(softmax)

        model.compile(optimizer=tf.keras.optimizers.SGD(lr=3e-4, momentum=0.9),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=[tf.keras.metrics.categorical_accuracy])

        return model

    def run_train(self):
        for epoch in range(cfg.train.nrof_epochs):
            train_spec = tf.estimator.TrainSpec(input_fn=lambda : self.train_input_fn(self.x_train, self.y_train))
            eval_spec = tf.estimator.EvalSpec(input_fn=lambda : self.valid_input_fn(self.x_test, self.y_test))
            eval_result = tf.estimator.train_and_evaluate(self.image_classifier, train_spec, eval_spec)

            print('Epoch %d' % epoch)
            print('Eval result: {}'.format(eval_result))

        return

estimator = EstimatorViaKerasExample()
estimator.run_train()