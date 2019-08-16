from model.network import LeNet5
from saliency.vanilla_gradient import save_vanilla_gradient
from model.data import mnist_train_test_sets
import numpy as np

# Get MNIST dataset, preprocessed
train_images, train_labels, test_images, test_labels = mnist_train_test_sets()

# Load net and 98% acc weights
net = LeNet5(weights_path="15epoch_weights.pkl")

# Uncomment if you want to train or test
# net.train(training_data=train_images, training_labels=train_labels,
#          batch_size=32, epochs=3, weights_path='weights.pkl')
# net.test(test_images, test_labels)

# Uncomment if you want to filter by class
# target_image_class = 7
# target_image_indexes = [i for i in range(len(test_labels))
#                        if np.argmax(test_labels[i]) == target_image_class]
# target_images = [test_images[index] for index in target_image_indexes]
# target_labels = [test_labels[index] for index in target_image_indexes]

# Generate saliency maps for the first 10 images
target_images = train_images[:10]
target_labels = train_labels[:10]
save_vanilla_gradient(network=net, data=target_images, labels=target_labels)
