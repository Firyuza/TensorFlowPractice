import tensorflow as tf
import numpy as np
import time
import os

from Lesson6.static_graph_tf_data.config import cfg
from Lesson6.static_graph_tf_data.Net import Network

tf.enable_eager_execution()

class StaticGraphExample:
    def __init__(self):
        self.__load_data()

        self.__create_graph_and_session()

        return

    def __load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
        self.y_train = self.y_train.reshape([-1])
        self.y_test = self.y_test.reshape([-1])

        self.nrof_classes = cfg.dataset.nrof_classes

        self.dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.dataset = self.dataset.map(map_func=self.__process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.dataset = self.dataset.shuffle(buffer_size=len(self.x_train))
        self.dataset = self.dataset.batch(batch_size=cfg.train.batch_size)

        return

    def __process_image(self, image, label):
        tf_mean = tf.constant(cfg.train.mu, dtype=tf.float32, shape=[1, 1, 3])
        tf_std = tf.constant(cfg.train.std, dtype=tf.float32, shape=[1, 1, 3])

        image = tf.cast(image, tf.float32)

        image -= tf_mean
        image /= tf_std

        return image, label

    def __create_placeholders(self):
        self.image_input = tf.keras.layers.Input(shape=(cfg.train.image_size,
                                                              cfg.train.image_size,
                                                              cfg.train.image_channels), dtype=tf.float32)
        self.image_label_input = tf.keras.layers.Input(shape=(), dtype=tf.int64)
        self.phase_train_input = tf.keras.layers.Input(shape=(), dtype=tf.bool)

        self.global_step = tf.Variable(0, trainable=False)

        return

    def __create_graph_and_session(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.network = Network(cfg.dataset.nrof_classes)
            self.__create_placeholders()

            self.logits = self.network(self.image_input)

            self.tf_predictions = tf.nn.softmax(self.logits)
            self.tf_predicted_classes = tf.argmax(self.tf_predictions, axis=1)
            self.tf_correct_predictions = tf.cast(tf.equal(self.image_label_input, self.tf_predicted_classes),
                                               dtype=tf.float32)

            self.tf_accuracy = tf.reduce_mean(self.tf_correct_predictions)
            self.tf_top5_accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=self.tf_predictions,
                                                                          targets=self.image_label_input, k=5),
                                                           tf.float32))

            tf.summary.scalar('train_accuracy', self.tf_accuracy)
            tf.summary.scalar('train_top5_accuracy', self.tf_top5_accuracy)

            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.image_label_input,
                                                                                      logits=self.logits))
            self.learning_rate = tf.train.exponential_decay(cfg.train.learning_rate, self.global_step,
                                                            cfg.train.learning_rate_decay_steps,
                                                            cfg.train.learning_rate_decay_factor, staircase=True)

            self.trainable_variables = tf.trainable_variables()

            self.reg_loss = cfg.train.weight_decay * tf.add_n(
                [tf.nn.l2_loss(v) for v in self.network.trainable_variables if 'bn' not in v.name])

            self.total_loss = self.loss + self.reg_loss

            tf.summary.scalar('cross_entropy', self.loss)
            tf.summary.scalar('reg_loss', self.reg_loss)
            tf.summary.scalar('total_loss', self.total_loss)
            tf.summary.scalar('learning_rate', self.learning_rate)

            if cfg.train.optimizer == 'ADAGRAD':
                opt = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
            elif cfg.train.optimizer == 'ADADELTA':
                opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate, rho=cfg.train.rho, epsilon=cfg.train.epsilon)
            elif cfg.train.optimizer == 'ADAM':
                opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999)  # , epsilon=0.1)
            else:
                opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)

            self.train_op = opt.minimize(self.total_loss, self.global_step, var_list=self.trainable_variables)

            if cfg.train.use_gpu:
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.)
                self.session = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options))
            else:
                session_conf = tf.ConfigProto(
                    device_count={'CPU': 1, 'GPU': 0},
                    allow_soft_placement=True,
                    log_device_placement=False
                )
                self.session = tf.Session(graph=self.graph, config=session_conf)

            self.summary_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(cfg.train.logs_base_dir, self.session.graph)

            # Initialize variables
            self.session.run(tf.initialize_all_variables())

        return

    def save_model(self, model_dir, step):
        # Save the model checkpoint
        print('Saving variables')
        start_time = time.time()

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        checkpoint_path = os.path.join(model_dir, 'model.h5')
        self.network.save_weights(checkpoint_path)

        save_time_variables = time.time() - start_time

        print('Variables saved in %.2f seconds' % save_time_variables)

        return

    def run_train(self):
        with self.session.as_default():
            print('Start train')
            if cfg.train.restore_model_path != '':
                # self.saver.restore(cfg.train.restore_model_path)
                step = int(cfg.train.restore_model_path.split('/')[-1].split('-')[-1])

                print('Model restored')
            else:
                step = 0

            epoch = 0
            while epoch < cfg.train.epoch_size:
                step = self.train_epoch(epoch)
                self.save_model(cfg.train.models_base_dir + str(epoch) + '/', step)

                epoch += 1

        self.session.close()

        return

    def train_epoch(self, epoch):
        nrof_examples = len(self.x_train)
        nrof_batches = int(np.ceil(nrof_examples / cfg.train.batch_size))

        batch_number = 0
        nrof_used_samples = 0
        for images, labels in self.dataset:
            start_time = time.time()
            feed_dict = {self.image_input: images.numpy(),
                         self.image_label_input: labels.numpy(),
                         self.phase_train_input: [True]}

            loss, _, step, predicted_classes, summary_data,\
                accuracy_b, accuracy_top5_b = self.session.run(
                [self.total_loss, self.train_op, self.global_step,
                 self.tf_predicted_classes, self.summary_op,
                 self.tf_accuracy, self.tf_top5_accuracy],
                feed_dict=feed_dict)

            self.summary_writer.add_summary(summary_data, step)
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss: %2.6f\tAccuracy: %.3f' %
                  (epoch, batch_number + 1, nrof_batches, duration, loss, accuracy_b))

            batch_number += 1
            nrof_used_samples += len(images)

        assert nrof_used_samples == nrof_examples

        return step

example = StaticGraphExample()
example.run_train()