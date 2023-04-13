import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time


CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'deer', 'cat', 'dog', 'frog', 'horse', 'ship', 'truck']


def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (227, 227))
    return image , label


if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

    validation_images, validation_labels = train_images[:5000], train_labels[:5000]
    train_images, train_labels = train_images[5000:], train_labels[5000:]

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

    plt.figure(figsize=(20, 20))
    for i, (image, label) in enumerate(train_ds.take(5)):
        ax = plt.subplot(5, 5, i+1)
        plt.imshow(image)
        print(label.numpy())
        plt.title(CLASS_NAMES[label.numpy()[0]])
        plt.axis('off')
    plt.show()

    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
    validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
    print("Training data size: ", train_ds_size)
    print("Test data size: ", test_ds_size)
    print("Validation data size: ", validation_ds_size)

    train_ds = (train_ds
                      .map(process_images)
                      .shuffle(buffer_size=train_ds_size)
                      .batch(batch_size=32, drop_remainder=True))
    test_ds = (test_ds
                      .map(process_images)
                      .shuffle(buffer_size=train_ds_size)
                      .batch(batch_size=32, drop_remainder=True))
    validation_ds = (validation_ds
                      .map(process_images)
                      .shuffle(buffer_size=train_ds_size)
                      .batch(batch_size=32, drop_remainder=True))
