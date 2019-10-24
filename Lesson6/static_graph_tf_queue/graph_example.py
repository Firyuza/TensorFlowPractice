import tensorflow as tf
import numpy as np
import time
import os

from tqdm import tqdm

from Lesson6.static_graph_tf_queue.config import cfg

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

        return

    def __process_image(self, image):
        tf_mean = tf.constant(cfg.train.mu, dtype=tf.float32, shape=[1, 1, 3])
        tf_std = tf.constant(cfg.train.std, dtype=tf.float32, shape=[1, 1, 3])

        image -= tf_mean
        image /= tf_std

        return image

    def __create_placeholders(self):
        self.image_placeholder = tf.placeholder(dtype=tf.float32, shape=(None,
                                                                        cfg.train.image_size,
                                                                        cfg.train.image_size,
                                                                        cfg.train.image_channels), name='image')
        self.image_label_placeholder = tf.placeholder(dtype=tf.int64, shape=(None), name='image_label')
        self.batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        self.augmentation_probability = tf.placeholder(dtype=tf.float32)
        self.phase_train_placeholder = tf.placeholder(dtype=tf.bool)

        self.global_step = tf.Variable(0, trainable=False)

        return

    def __prepare_batch(self, images, labels, augmentation_probability):
        elems = tf.convert_to_tensor([False, True], dtype=tf.bool)
        probs = tf.convert_to_tensor([1.0 - augmentation_probability, augmentation_probability])
        rescaled_probs = tf.expand_dims(tf.log(probs), 0)
        indices = tf.multinomial(rescaled_probs, tf.shape(images)[0])
        augment_probs = tf.gather(elems, tf.squeeze(indices, [0]))

        self.queue = tf.FIFOQueue(capacity=100000,
                                   dtypes=[tf.float32, tf.int64, tf.bool],
                                   shapes=[(cfg.train.image_size,
                                            cfg.train.image_size,
                                            cfg.train.image_channels), (), ()])
        enqueue_op = self.queue.enqueue_many([images, labels, augment_probs])

        nrof_preprocess_threads = cfg.train.nrof_threads
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            image, label, augment_prob = self.queue.dequeue()
            image = self.__process_image(image)
            images_and_labels.append([image, label])

        image_batch, label_batch = tf.train.batch_join(images_and_labels,
                                                       batch_size=self.batch_size_placeholder,
                                                       enqueue_many=False)

        return enqueue_op, image_batch, label_batch

    def save_model(self, model_dir, step):
        # Save the model checkpoint
        print('Saving variables')
        start_time = time.time()

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        checkpoint_path = os.path.join(model_dir, 'model.ckpt')
        self.saver.save(self.session, checkpoint_path, global_step=step)

        save_time_variables = time.time() - start_time

        print('Variables saved in %.2f seconds' % save_time_variables)

        return

    def __create_graph_and_session(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.__create_placeholders()

            self.enqueue_op, self.images_batch, self.labels_batch = self.__prepare_batch(
                images=self.image_placeholder,
                labels=self.image_label_placeholder,
                augmentation_probability=self.augmentation_probability)

            self.backbone_model = tf.keras.applications.resnet50.ResNet50(input_tensor=self.images_batch,
                                                                      weights='imagenet',
                                                                      include_top=False)
            backbone_output = self.backbone_model.output
            backbone_output = tf.keras.layers.GlobalAveragePooling2D()(backbone_output)
            self.logits = tf.keras.layers.Dense(self.nrof_classes)(backbone_output)

            self.tf_predictions = tf.nn.softmax(self.logits)
            self.tf_predicted_classes = tf.argmax(self.tf_predictions, axis=1)
            self.tf_correct_predictions = tf.cast(tf.equal(self.labels_batch, self.tf_predicted_classes),
                                               dtype=tf.float32)

            self.tf_accuracy = tf.reduce_mean(self.tf_correct_predictions)
            self.tf_top5_accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=self.logits,
                                                                          targets=self.labels_batch, k=5),
                                                           tf.float32))

            tf.summary.scalar('train_accuracy', self.tf_accuracy)
            tf.summary.scalar('train_top5_accuracy', self.tf_top5_accuracy)

            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_batch,
                                                                                      logits=self.logits))
            self.learning_rate = tf.train.exponential_decay(cfg.train.learning_rate, self.global_step,
                                                            cfg.train.learning_rate_decay_steps,
                                                            cfg.train.learning_rate_decay_factor, staircase=True)

            self.trainable_variables = tf.trainable_variables()

            self.reg_loss = cfg.train.weight_decay * tf.add_n(
                [tf.nn.l2_loss(v) for v in self.trainable_variables if 'bn' not in v.name])

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
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.train.gpu_memory_fraction)
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

            # Create a saver
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

            # Initialize variables
            self.session.run(tf.initialize_all_variables())

            self.coordinator = tf.train.Coordinator()
            self.threads = tf.train.start_queue_runners(coord=self.coordinator, sess=self.session)

        return

    def __shuffle_data(self):
        indices = np.random.permutation(len(self.x_train))
        self.x_train = np.asarray(self.x_train)[indices]
        self.y_train = np.asarray(self.y_train)[indices]

        return

    def run_train(self):
        with self.session.as_default():
            print('Start train')
            if cfg.train.restore_model_path != '':
                self.saver.restore(self.session, cfg.train.restore_model_path)
                step = int(cfg.train.restore_model_path.split('/')[-1].split('-')[-1])

                print('Model restored')
            else:
                step = 0

            epoch = 0
            while epoch < cfg.train.epoch_size:
                step = self.train_epoch(epoch)
                self.save_model(cfg.train.models_base_dir + str(epoch) + '/', step)

                epoch += 1

        self.coordinator.request_stop()
        self.session.run(self.queue.close(cancel_pending_enqueues=True))
        self.coordinator.join(self.threads)
        self.session.close()

        return

    def train_epoch(self, epoch):
        self.__shuffle_data()
        nrof_examples = len(self.x_train)

        self.session.run(self.enqueue_op,
                      {self.image_placeholder: self.x_train,
                       self.image_label_placeholder: self.y_train,
                       self.augmentation_probability: cfg.train.augmentation_probability,
                       self.phase_train_placeholder: True,})

        nrof_batches = int(np.ceil(nrof_examples / cfg.train.batch_size))

        batch_number = 0
        nrof_used_samples = 0
        while batch_number < nrof_batches:
            start_time = time.time()
            batch_size = min(nrof_examples - batch_number * cfg.train.batch_size, cfg.train.batch_size)
            feed_dict = {self.phase_train_placeholder: True,
                         self.batch_size_placeholder: batch_size}

            loss, _, step, images_b, labels_b, \
            predicted_classes, accuracy_b, accuracy_top5_b, summary_data = self.session.run(
                [self.total_loss, self.train_op, self.global_step,
                 self.images_batch, self.labels_batch,
                 self.tf_predicted_classes,
                 self.tf_accuracy, self.tf_top5_accuracy, self.summary_op],
                feed_dict=feed_dict)

            self.summary_writer.add_summary(summary_data, step)

            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss: %2.6f\tAccuracy: %.3f' %
                  (epoch, batch_number + 1, nrof_batches, duration, loss, accuracy_b))

            batch_number += 1
            nrof_used_samples += batch_size

        assert nrof_used_samples == nrof_examples

        return step

example = StaticGraphExample()
example.run_train()