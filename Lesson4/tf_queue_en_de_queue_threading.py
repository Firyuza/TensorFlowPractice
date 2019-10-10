import tensorflow as tf
import numpy as np
import threading
import time

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

NROF_THREADS = 2
NROF_ITERATIONS = 2
BATCH_SIZE = 2
USE_GPU = True
GPU_MEMORY_FRACTION = 1.0

IMAGE_EXTENSION = 'jpg'
IMAGE_CHANNEL_SIZE = 3
IMAGE_WIDTH = 218
IMAGE_HEIGHT = 178

class TestFIFOQueueEnDeQueue:
    def __init__(self, images_path):

        self.images_path = images_path
        self.images_labels = list(np.arange(len(images_path)))
        self.nrof_examples = len(images_path)

        self.__create_graph()

        return

    def __create_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.image_path_ph = tf.placeholder(shape=(), dtype=tf.string)
            self.image_label_ph = tf.placeholder(shape=(), dtype=tf.int64)
            self.batch_size_ph = tf.placeholder(shape=(), dtype=tf.int32)

            self.queue = tf.queue.FIFOQueue(capacity=500, dtypes=[tf.string, tf.int64], shapes=[(), ()])

            self.enqueue_op = self.queue.enqueue([self.image_path_ph, self.image_label_ph])

            # self.paths_batch, self.labels_batch = self.queue.dequeue_many(n=self.batch_size_ph)
            paths_images_and_labels = []
            for _ in range(NROF_THREADS):
                filename, label = self.queue.dequeue()
                image = self.__read_image_from_disk(filename)
                image.set_shape((IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL_SIZE))
                paths_images_and_labels.append([filename, image, label])

            self.paths_batch, self.images_batch, self.labels_batch = tf.train.batch_join(paths_images_and_labels,
                                                                       batch_size=self.batch_size_ph,
                                                                       enqueue_many=False)

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

        self.threads_enqueue = [threading.Thread(target=self.__load_data) for _ in range(NROF_THREADS)]

        self.threads_dequeue = tf.train.start_queue_runners(coord=self.coordinator, sess=self.session)

        return

    def __read_image_from_disk(self, path):
        file_contents = tf.io.read_file(path)
        if IMAGE_EXTENSION in ['jpeg', 'jpg']:
            example = tf.image.decode_jpeg(file_contents, channels=IMAGE_CHANNEL_SIZE)
        elif IMAGE_EXTENSION == 'png':
            example = tf.image.decode_png(file_contents, channels=IMAGE_CHANNEL_SIZE)

        return example

    def __extract_data(self):
        return self.images_path.pop(), self.images_labels.pop()

    def __load_data(self):
        try:
            while not self.coordinator.should_stop():
                image, label = self.__extract_data()

                print(label)

                self.session.run(self.enqueue_op,
                                 feed_dict={
                                     self.image_path_ph: image,
                                     self.image_label_ph: label
                                 })
        except IndexError:
            print('Enqueue op is finished')

        return

    def run_FIFO(self):
        with self.session.as_default():
            [thread.start() for thread in self.threads_enqueue]

            nrof_batches = int(np.ceil(self.nrof_examples / BATCH_SIZE))
            i = 0
            while i < nrof_batches:
                batch_size = min(self.nrof_examples - i * BATCH_SIZE, BATCH_SIZE)

                path_out, labels_out = self.session.run([self.paths_batch, self.labels_batch],
                                                          feed_dict={self.batch_size_ph: batch_size})
                print(labels_out)
                i += 1

            self.coordinator.request_stop()

            self.session.run(self.queue.close(cancel_pending_enqueues=True))

            self.coordinator.join(self.threads_dequeue + self.threads_enqueue, stop_grace_period_secs=5)

            self.session.close()

        return


images_path = ['/home/firiuza/PycharmProjects/TensorflowPractice1/data/000001.jpg'] * 12

queue_obj = TestFIFOQueueEnDeQueue(images_path)
queue_obj.run_FIFO()