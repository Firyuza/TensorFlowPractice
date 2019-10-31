import tensorflow as tf
import numpy as np

from estimator.config import cfg

class Model(object):
    def __call__(self, inputs):
        net = tf.layers.conv2d(inputs, 32, [5, 5],
                               activation=tf.nn.relu, name='conv1')
        net = tf.layers.max_pooling2d(net, [2, 2], 2,
                                      name='pool1')
        net = tf.layers.conv2d(net, 64, [5, 5],
                               activation=tf.nn.relu, name='conv2')
        net = tf.layers.max_pooling2d(net, [2, 2], 2,
                                      name='pool2')
        net = tf.layers.flatten(net)

        logits = tf.layers.dense(net, 10,
                                 activation=None, name='fc1')
        return logits

class EstimatorExample:
    def __init__(self):
        self.__load_data()

        # Create a estimator with model_fn
        self.image_classifier = tf.estimator.Estimator(model_fn=self.model_fn,
                                                  model_dir=cfg.train.models_base_dir)

        return

    def __load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
        self.y_train = np.asarray(self.y_train.reshape([-1]), dtype=np.int32)
        self.y_test = np.asarray(self.y_test.reshape([-1]), dtype=np.int32)

        self.nrof_classes = 10
        return

    def __preperare_train_data(self, image, label):
        tf_mean = tf.constant(cfg.train.mu, dtype=tf.float32, shape=[1, 1, 3])
        tf_std = tf.constant(cfg.train.std, dtype=tf.float32, shape=[1, 1, 3])

        image = tf.cast(image, tf.float32)

        image -= tf_mean
        image /= tf_std

        return image, label

    def __preperare_valid_data(self, image, label):
        tf_mean = tf.constant(cfg.train.mu, dtype=tf.float32, shape=[1, 1, 3])
        tf_std = tf.constant(cfg.train.std, dtype=tf.float32, shape=[1, 1, 3])

        image = tf.cast(image, tf.float32)

        image -= tf_mean
        image /= tf_std

        return image, label

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
        dataset = dataset.batch(batch_size=cfg.train.batch_size)
        dataset = dataset.shuffle(buffer_size=len(features)).\
            prefetch(buffer_size=cfg.train.batch_size)

        return dataset

    def model_fn(self, features, labels, mode):
        phase_name = 'train' if mode == tf.estimator.ModeKeys.TRAIN else 'validation'

        model = Model()
        global_step = tf.train.get_global_step()

        logits = model(features)
        predicted_logit = tf.argmax(input=logits, axis=1,
                                    output_type=tf.int32)
        probabilities = tf.nn.softmax(logits)

        predictions = {
            "predicted_logit": predicted_logit,
            "probabilities": probabilities
        }
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions)

        with tf.name_scope('loss'):
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                labels=labels, logits=logits, scope='loss')
            tf.summary.scalar('%s/loss' % phase_name, cross_entropy)
        with tf.name_scope('accuracy'):
            accuracy = tf.metrics.accuracy(
                labels=labels, predictions=predicted_logit, name='acc')
            tf.summary.scalar('%s/accuracy' % phase_name, accuracy[1])

        valid_hook_list = []
        valid_tensors_log = {'%s/accuracy' % phase_name: accuracy[1],
                             '%s/loss' % phase_name: cross_entropy}
        valid_hook_list.append(tf.train.LoggingTensorHook(
            tensors=valid_tensors_log, every_n_iter=1))

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=cross_entropy,
                eval_metric_ops={'accuracy/accuracy': accuracy},
                evaluation_hooks=valid_hook_list)

        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=0.001)
        train_op = optimizer.minimize(
            cross_entropy, global_step=global_step)

        # Create a hook to print acc, loss & global step every 100 iter.
        train_hook_list = []
        train_tensors_log = {'%s/accuracy' % phase_name: accuracy[1],
                             '%s/loss' % phase_name: cross_entropy}
        train_hook_list.append(tf.train.LoggingTensorHook(
            tensors=train_tensors_log, every_n_iter=1))

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=cross_entropy,
                train_op=train_op,
                training_hooks=train_hook_list)

    def run_train(self):
        for epoch in range(cfg.train.nrof_epochs):
            self.image_classifier.train(input_fn=
                                   lambda : self.train_input_fn(self.x_train, self.y_train))

            metrics = self.image_classifier.evaluate(input_fn=
                                                lambda : self.valid_input_fn(self.x_test, self.y_test))

            print('Epoch %d' % epoch)
            print('Eval result: {}'.format(metrics))

        return

estimator = EstimatorExample()
estimator.run_train()