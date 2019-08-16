import numpy as np
from abc import ABCMeta, abstractmethod


class Layer(metaclass=ABCMeta):
    """ The parent class for all layers """
    @abstractmethod
    def forward(self, inputs):
        """ Feedforward pass """
        raise NotImplementedError()

    @abstractmethod
    def backward(self, dy):
        raise NotImplementedError()

    @abstractmethod
    def parameters(self):
        return NotImplementedError()

    @abstractmethod
    def load(self, weights, bias):
        return NotImplementedError()


class Conv2D(Layer):
    def __init__(self, n_filter, n_channel, kernel_size,
                 padding, stride, learning_rate, name):
        self.n_filter = n_filter
        # n_channel  # 1 if grayscale, 3 if RGB
        self.kernal_size = kernel_size  # height and width of convolution
        self.weights = np.zeros((n_filter, n_channel,
                                 self.kernal_size, self.kernal_size))
        self.bias = np.zeros((self.n_filter, 1))

        # Generate random values for each convolutional filter
        for filter_index in range(0, self.n_filter):
            self.weights[filter_index, :, :, :] = \
                np.random.normal(loc=0,
                                 scale=np.sqrt(
                                     1./(n_channel*kernel_size**2)),
                                 size=(n_channel, kernel_size, kernel_size))

        self.padding = padding
        self.stride = stride
        self.lr = learning_rate
        self.name = name

    def forward(self, inputs):
        # input size: (C, W, H)
        # output size: (n_filter, map_w, map_h)
        C = inputs.shape[0]
        W = inputs.shape[1] + 2*self.padding
        H = inputs.shape[2] + 2*self.padding
        self.inputs = np.zeros((C, W, H))
        for c in range(inputs.shape[0]):
            self.inputs[c, :, :] = self._zero_padding(inputs[c, :, :],
                                                      self.padding)
        map_w = int((W-self.kernal_size)/self.stride + 1)
        map_h = int((H-self.kernal_size)/self.stride + 1)

        feature_maps = np.zeros((self.n_filter, map_w, map_h))
        for filter_i in range(self.n_filter):
            for w in range(map_w):
                for h in range(map_h):
                    inputs = self.inputs[:,
                                         w: w+self.kernal_size,
                                         h: h+self.kernal_size]
                    weights = self.weights[filter_i, :, :, :]
                    feature_maps[filter_i, w, h] = np.sum(inputs * weights) + \
                        self.bias[filter_i]
        return feature_maps

    def _zero_padding(self, inputs, size):
        w, h = inputs.shape[0], inputs.shape[1]
        new_w = w + 2*size
        new_h = h + 2*size
        padded_input = np.zeros((new_w, new_h))
        padded_input[size:w+size, size:h+size] = inputs
        return padded_input

    def backward(self, dy):
        C, W, H = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        dw = np.zeros(self.weights.shape)
        db = np.zeros(self.bias.shape)

        F, W, H = dy.shape
        for f in range(F):
            for w in range(W):
                for h in range(H):
                    width = slice(w, w+self.kernal_size)
                    height = slice(h, h+self.kernal_size)
                    delta_y = dy[f, w, h]

                    dw[f, :, :, :] += delta_y * self.inputs[:, width, height]
                    dx[:, width, height] += delta_y * self.weights[f, :, :, :]

        for f in range(F):
            db[f] = np.sum(dy[f, :, :])

        self.weights -= self.lr * dw
        self.bias -= self.lr * db
        return dx

    def parameters(self):
        return {self.name+'.weights': self.weights,
                self.name+'.bias': self.bias}

    def load(self, weights, bias):
        self.weights = weights
        self.bias = bias


class Maxpool2D(Layer):
    def __init__(self, pool_size, stride, name):
        self.pool = pool_size
        self.stride = stride
        self.name = name

    def forward(self, inputs):
        self.inputs = inputs
        C, W, H = inputs.shape
        new_width = int((W - self.pool)/self.stride + 1)
        new_height = int((H - self.pool)/self.stride + 1)
        out = np.zeros((C, new_width, new_height))
        for c in range(C):
            for w in range(int(W/self.stride)):
                for h in range(int(H/self.stride)):
                    width = slice(w*self.stride, w*self.stride+self.pool)
                    height = slice(h*self.stride, h*self.stride+self.pool)
                    out[c, w, h] = np.max(self.inputs[c, width, height])
        return out

    def backward(self, dy):
        C, W, H = self.inputs.shape
        dx = np.zeros(self.inputs.shape)

        for c in range(C):
            for w in range(0, W, self.pool):
                for h in range(0, H, self.pool):
                    i = np.argmax(self.inputs[c, w:w+self.pool, h:h+self.pool])
                    idx, idy = np.unravel_index(i, (self.pool, self.pool))
                    dx[c, w+idx, h+idy] = dy[c,
                                             int(w/self.pool),
                                             int(h/self.pool)]
        return dx

    def parameters(self):
        return

    def load(self, weights, bias):
        return


class Dense(Layer):
    def __init__(self, num_inputs, num_outputs, learning_rate, name):
        self.weights = 0.01*np.random.rand(num_inputs, num_outputs)
        self.bias = np.zeros((num_outputs, 1))
        self.lr = learning_rate
        self.name = name

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(self.inputs, self.weights) + self.bias.T

    def backward(self, dy):
        if dy.shape[0] == self.inputs.shape[0]:
            dy = dy.T
        dw = dy.dot(self.inputs)
        db = np.sum(dy, axis=1, keepdims=True)
        dx = np.dot(dy.T, self.weights.T)

        self.weights -= self.lr * dw.T
        self.bias -= self.lr * db

        return dx

    def parameters(self):
        return {self.name+'.weights': self.weights,
                self.name+'.bias': self.bias}

    def load(self, weights, bias):
        self.weights = weights
        self.bias = bias


class Flatten(Layer):
    def __init__(self):
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        C, W, H = inputs.shape
        return inputs.reshape(1, C * W * H)

    def backward(self, dy):
        C, W, H = self.inputs.shape
        return dy.reshape(C, W, H)

    def parameters(self):
        return

    def load(self, weights, bias):
        return


class ReLu(Layer):
    def __init__(self):
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        x = inputs.copy()
        x[x < 0] = 0
        return x

    def backward(self, dy):
        dx = dy.copy()
        dx[self.inputs < 0] = 0
        return dx

    def parameters(self):
        return

    def load(self, weights, bias):
        return


class Softmax(Layer):
    def __init__(self):
        self.out = None

    def forward(self, inputs):
        exp = np.exp(inputs, dtype=np.float128)
        self.out = exp/np.sum(exp)
        return self.out

    def backward(self, dy):
        return self.out.T - dy.reshape(dy.shape[0], 1)

    def parameters(self):
        return

    def load(self, weights, bias):
        return
