import mnist
import numpy as np


def mnist_train_test_sets():
    print('Loading data, requires network connection on first run')
    # [60000, 28, 28]
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    # [10000, 28, 28]
    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    print('Normalizing pixel values to range 0-1')
    train_images -= np.mean(np.int16(train_images), dtype=np.int16)
    train_images = train_images/np.int16(np.std(train_images))
    test_images -= np.mean(np.int16(test_images), dtype=np.int16)
    test_images = test_images/np.int16(np.std(test_images))

    print("Reshaping data for model, which expects a color channel")
    # MNIST is black and white, 1 color channel
    train_images = train_images.reshape(60000, 1, 28, 28)
    train_labels = np.eye(10)[train_labels]
    test_images = test_images.reshape(10000, 1, 28, 28)
    test_labels = np.eye(10)[test_labels]

    return (train_images, train_labels, test_images, test_labels)
