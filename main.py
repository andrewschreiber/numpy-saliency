import numpy as np
import mnist
from model.network import LeNet
from saliency.vanilla_gradient import save_vanilla_gradient


print('Loading data...')
# [60000, 28, 28]
train_images = np.int16(mnist.train_images())
train_labels = np.int16(mnist.train_labels())
# [10000, 28, 28]
test_images = np.int16(mnist.test_images())
test_labels = np.int16(mnist.test_labels())

print('Normalizing data...')
train_images -= np.int16(np.mean(train_images))
train_images = train_images/np.int16(np.std(train_images))
test_images -= np.int16(np.mean(test_images))
test_images = test_images/np.int16(np.std(test_images))

print("Shaping data...")
training_data = train_images.reshape(60000, 1, 28, 28)
training_labels = np.eye(10)[train_labels]
testing_data = test_images.reshape(10000, 1, 28, 28)
testing_labels = np.eye(10)[test_labels]

net = LeNet()

# net.load("weights.pkl")

net.train(training_data[:100], training_labels[:100], 32, 3, 'weights.pkl')

net.test(testing_data[:100], testing_labels[:100])

# save_vanilla_gradient(net, training_data[:25], training_labels[:25], 5)
