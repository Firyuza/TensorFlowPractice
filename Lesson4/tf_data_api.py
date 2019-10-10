import tensorflow as tf
import os
import numpy as np
import cv2

tf.compat.v1.enable_eager_execution()

IMAGE_PATHS = ['/home/firiuza/PycharmProjects/TensorflowPractice1/data/000001.jpg'] * 12
IMAGE_LABELS = [1] * len(IMAGE_PATHS)
BATCH_SIZE = 4
PREFETCH_BUFFER_SIZE = BATCH_SIZE
IMAGE_CHANNEL_SIZE = 3

def read_and_augmment_image_via_python(filename, label):
    def read_cv2(path_):
        image = cv2.imread(str(np.core.defchararray.decode(path_)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    img_tensor = tf.py_func(read_cv2, [filename], np.uint8)
    img_tensor.set_shape((None, None, 3))
    img_final = tf.image.resize(img_tensor, [256, 256])
    img_final = img_final / 255.0

    return img_final, label

def read_and_augmment_image(filename, label):
    file_contents = tf.io.read_file(filename)
    img_tensor = tf.image.decode_jpeg(file_contents, channels=IMAGE_CHANNEL_SIZE)
    img_tensor.set_shape((None, None, IMAGE_CHANNEL_SIZE))
    img_final = tf.image.resize(img_tensor, [256, 256])
    img_final = img_final / 255.0

    return img_final, label

def test_dataset_api():
    dataset = tf.data.Dataset.from_tensor_slices((IMAGE_PATHS, IMAGE_LABELS))

    dataset = dataset.map(map_func=read_and_augmment_image,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size=len(IMAGE_PATHS))
    dataset = dataset.batch(batch_size=BATCH_SIZE)


    dataset = dataset.prefetch(buffer_size=PREFETCH_BUFFER_SIZE)

    for image, label in dataset:
        print(label.numpy())

    return

test_dataset_api()