import numpy as np
import pickle
from model.loss import cross_entropy
from model.layers import Conv2D, Maxpool2D, Dense, Flatten, ReLu, Softmax


class LeNet5:
    """Implementation of LeNet 5 for MNIST
       http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
    """

    def __init__(self, weights_path=None):
        lr = 0.01
        layers = []
        layers.append(Conv2D(n_filter=6, n_channel=1,
                             kernel_size=5, padding=2, stride=1,
                             learning_rate=lr, name='conv1'))
        layers.append(ReLu())
        layers.append(Maxpool2D(
            pool_size=2, stride=2, name='maxpool2'))
        layers.append(Conv2D(n_filter=16, n_channel=6,
                             kernel_size=5, padding=0, stride=1,
                             learning_rate=lr, name='conv3'))
        layers.append(ReLu())
        layers.append(Maxpool2D(
            pool_size=2, stride=2, name='maxpool4'))
        layers.append(Conv2D(n_filter=120, n_channel=16,
                             kernel_size=5, padding=0, stride=1,
                             learning_rate=lr, name='conv5'))
        layers.append(ReLu())
        layers.append(Flatten())
        layers.append(Dense(
            num_inputs=120, num_outputs=84, learning_rate=lr, name='dense6'))
        layers.append(ReLu())
        layers.append(Dense(
            num_inputs=84, num_outputs=10, learning_rate=lr, name='dense7'))
        layers.append(Softmax())
        self.layers = layers
        if weights_path is not None:
            self._load(weights_path)

    def _load(self, weights_path):
        with open(weights_path, 'rb') as handle:
            b = pickle.load(handle)
        self.layers[0].load(b[0]['conv1.weights'], b[0]['conv1.bias'])
        self.layers[3].load(b[3]['conv3.weights'], b[3]['conv3.bias'])
        self.layers[6].load(b[6]['conv5.weights'], b[6]['conv5.bias'])
        self.layers[9].load(b[9]['dense6.weights'], b[9]['dense6.bias'])
        self.layers[11].load(b[11]['dense7.weights'], b[11]['dense7.bias'])

    def train(self, training_data, training_labels, batch_size, epochs,
              weights_path):
        print("Training LeNet...")
        total_acc = 0
        for epoch in range(epochs):
            # batch training data
            for batch_index in range(0, training_data.shape[0], batch_size):
                loss = 0
                acc = 0

                data = training_data[batch_index:batch_index+batch_size]
                labels = training_labels[batch_index:batch_index+batch_size]

                # iterate over batch
                for b in range(len(data)):
                    x = data[b]
                    y = labels[b]

                    # forward pass
                    output = self.forward(x)
                    if np.argmax(output) == np.argmax(y):
                        acc += 1
                        total_acc += 1
                    loss += cross_entropy(output, y)

                    # backward pass
                    # update network on each datapoint for simplicity
                    dy = y
                    for l in range(len(self.layers)-1, -1, -1):
                        dout = self.layers[l].backward(dy)
                        dy = dout

                # print performance
                loss /= len(data)
                batch_acc = float(acc)/float(len(data))
                train_acc = float(total_acc) / \
                    float((batch_index+len(data)+epoch*len(training_data)))

                print(('| Epoch: {0:d}/{1:d} | Iter:{2:d} | Loss: {3:.2f} | ' +
                       'BatchAcc: {4:.2f} | TrainAcc: {5:.2f} |')
                      .format(epoch+1, epochs, batch_index+len(data),
                              loss, batch_acc, train_acc))

            # save parameters after each epoch
            print("Saving model to", weights_path)
            layers = [layer.parameters() for layer in self.layers]
            with open(weights_path, 'wb') as handle:
                pickle.dump(layers, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def forward(self, x):
        for l in range(len(self.layers)):
            output = self.layers[l].forward(x)
            x = output
        return output

    def predict(self, x):
        output = self.forward(x)
        digit = np.argmax(output)
        probability = output[0, digit]
        return digit, probability

    def test(self, data, labels):
        print("Testing LeNet...")
        total_acc = 0
        test_size = len(data)
        for i in range(test_size):
            x = data[i]
            y = labels[i]
            if np.argmax(self.forward(x)) == np.argmax(y):
                total_acc += 1

        print("== Correct: {}/{}. Accuracy: {} =="
              .format(total_acc, test_size, total_acc/test_size))
