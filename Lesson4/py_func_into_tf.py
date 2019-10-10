import tensorflow as tf
import numpy as np
import threading
import time
import cv2

NROF_THREADS = 2
NROF_ITERATIONS = 2
BATCH_SIZE = 2
USE_GPU = True
GPU_MEMORY_FRACTION = 1.0

IMAGE_EXTENSION = 'jpg'
IMAGE_CHANNEL_SIZE = 3
IMAGE_WIDTH = 218
IMAGE_HEIGHT = 178

class TestFIFOQueueDequeue:
    def __init__(self, images_path):

        self.images_path = images_path
        self.images_labels = np.arange(len(images_path))

        self.__create_graph()

        return

    def __create_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.image_path_ph = tf.placeholder(shape=(None), dtype=tf.string)
            self.image_label_ph = tf.placeholder(shape=(None), dtype=tf.int64)
            self.batch_size_ph = tf.placeholder(shape=(), dtype=tf.int32)

            self.enqueue_op, self.paths_batch, self.images_batch, self.labels_batch = \
                self.__create_FIFO(self.image_path_ph, self.image_label_ph)

        self.tensorboard_writer = tf.summary.FileWriter('./logs', self.graph )

        if USE_GPU:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEMORY_FRACTION)
            session_conf = tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=True,
                                          gpu_options=gpu_options)
        else:
            session_conf = tf.ConfigProto(
                device_count={'CPU': 1, 'GPU': 0},
                allow_soft_placement=True,
                log_device_placement=True
            )
        self.session = tf.Session(graph=self.graph, config=session_conf)
        self.coordinator = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coordinator, sess=self.session)

        return

    def __read_cv2(self, path_):
        image = cv2.imread(str(np.core.defchararray.decode(path_)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def __create_FIFO(self, image_path, image_label):
        self.queue = tf.queue.FIFOQueue(capacity=100, dtypes=[tf.string, tf.int64], shapes=[(), ()])

        enqueue_op = self.queue.enqueue_many([image_path, image_label])

        paths_images_and_labels = []
        for _ in range(NROF_THREADS):
            filename, label = self.queue.dequeue()
            image = tf.py_func(self.__read_cv2, [filename], np.uint8)
            image.set_shape((IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL_SIZE))
            paths_images_and_labels.append([filename, image, label])

        # Fill Queue to create a batch of examples
        path_batch, image_batch, label_batch = tf.train.batch_join(paths_images_and_labels,
                                                      batch_size=self.batch_size_ph,
                                                      enqueue_many=False)

        return enqueue_op, path_batch, image_batch, label_batch

    def run_FIFO(self):
        with self.session.as_default():
            self.session.run(self.enqueue_op, feed_dict={self.image_path_ph: self.images_path,
                                                         self.image_label_ph: self.images_labels})
            nrof_examples = len(self.images_path)
            nrof_batches = int(np.ceil(nrof_examples / BATCH_SIZE))
            i = 0
            while i < nrof_batches:
                batch_size = min(nrof_examples - i * BATCH_SIZE, BATCH_SIZE)

                images_out, labels_out = self.session.run([self.images_batch, self.labels_batch],
                                                          feed_dict={self.batch_size_ph: batch_size})
                print(labels_out)
                i += 1

            self.coordinator.request_stop()
            self.session.run(self.queue.close(cancel_pending_enqueues=True))
            self.coordinator.join(self.threads)
            self.session.close()

        return


images_path = ['/home/firiuza/PycharmProjects/TensorflowPractice1/data/000001.jpg'] * 12

queue_obj = TestFIFOQueueDequeue(images_path)
queue_obj.run_FIFO()