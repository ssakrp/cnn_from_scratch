import numpy as np
import random
import math
from keras.datasets import mnist
from matplotlib import pyplot


class ConvolutionalLayer:
    def __init__(self, old_depth, old_height, old_width, fc, fs):       # входные размеры, fc - кол-во фильтров, fs - длина и ширина фильтра
        self.old_depth = old_depth
        self.old_height = old_height
        self.old_width = old_width
        self.new_depth = fc
        self.new_height = self.old_height - fs + 1
        self.new_width = self.old_width - fs + 1
        self.filter_count = fc
        self.filter_size = fs
        self.filter_depth = old_depth
        self.W = np.zeros((fc, self.filter_depth, self.filter_size, self.filter_size))      # веса ядер
        self.dW = np.zeros((fc, self.filter_depth, self.filter_size, self.filter_size))     # градиенты весов ядер
        self.b = np.zeros((self.filter_count,))                                             # веса смещений
        self.db = np.zeros((self.filter_count,))                                            # градиенты весов смещений

        for f in range(self.filter_count):
            for i in range(self.filter_size):
                for j in range(self.filter_size):
                    for k in range(self.filter_depth):
                        (self.W[f])[k][i][j] = random.uniform(-0.1, 0.1)

            self.b[f] = 0.1



    def forward(self, input_tensor):
        output_tensor = np.zeros((self.new_depth, self.new_height, self.new_width))
        for f in range(self.filter_count):
            for y in range(self.new_height):
                for x in range(self.new_width):
                    sum = self.b[f]

                    for i in range(self.filter_size):
                        for j in range(self.filter_size):
                            i0 = y + i
                            j0 = x + j
                            for k in range(self.filter_depth):
                                sum += input_tensor[k, i0, j0] * (self.W[f])[k, i, j]
                    output_tensor[f, y, x] = sum
        return output_tensor

    def backward(self, input_tensor, deltas_tensor):
        for f in range(self.filter_count):
            for y in range(self.new_height):
                for x in range(self.new_width):
                    self.db[f] += deltas_tensor[f, y, x]

        for f in range(self.filter_count):
            for y in range(self.new_height):
                for x in range(self.new_width):
                    delta = deltas_tensor[f, y, x]
                    for i in range(self.filter_size):
                        for j in range(self.filter_size):
                            i0 = i + y
                            j0 = j + x
                            if i0 < 0 or i0 >= self.old_height or j0 < 0 or j0 >= self.old_width:
                                continue
                            for k in range(self.filter_depth):
                                (self.dW[f])[k, i, j] += delta * input_tensor[k, i0, j0]

        dX = np.zeros((self.old_depth, self.old_height, self.old_width))
        for y in range(self.old_height):
            for x in range(self.old_width):
                for k in range(self.filter_depth):
                    sum = 0
                    for i in range(self.filter_size):
                        for j in range(self.filter_size):
                            i0 = y + i - (self.filter_size - 1)
                            j0 = x + j - (self.filter_size - 1)
                            if i0 < 0 or i0 >= self.new_height or j0 < 0 or j0 >= self.new_width:
                                continue
                            for f in range(self.filter_count):
                                sum += (self.W[f])[k, self.filter_size - 1 - i, self.filter_size - 1 - j] * \
                                       deltas_tensor[f, i0, j0]
                    dX[k, y, x] = sum
        return dX

    def update_weights(self, learning_rate):
        for f in range(self.filter_count):
            for i in range(self.filter_size):
                for j in range(self.filter_size):
                    for k in range(self.filter_depth):
                        (self.W[f])[k, i, j] -= learning_rate * (self.dW[f])[k, i, j]
                        (self.dW[f])[k, i, j] = 0
            self.b[f] -= learning_rate * self.db[f]
            self.db[f] = 0

    def activation_forward(self, tensor):
        result_tensor = np.zeros_like(tensor, dtype='float')
        for i in range(tensor.shape[1]):
            for j in range(tensor.shape[2]):
                for k in range(tensor.shape[0]):
                    if tensor[k, i, j] > 0:
                        result_tensor[k, i, j] = tensor[k, i, j]
                    else:
                        result_tensor[k, i, j] = 0
        return result_tensor

    def activation_backward(self, tensor, deltas):
        dX = np.zeros_like(tensor, dtype='float')
        for i in range(tensor.shape[1]):
            for j in range(tensor.shape[2]):
                for k in range(tensor.shape[0]):
                    dX[k, i, j] = deltas[k, i, j]
                    if tensor[k, i, j] > 0:
                        dX[k, i, j] *= 1
                    else:
                        dX[k, i, j] *= 0
        return dX


class PoolingLayer:
    def __init__(self, old_depth, old_height, old_width, scale):
        self.old_depth = old_depth
        self.old_height = old_height
        self.old_width = old_width
        self.scale = scale
        self.new_depth = self.old_depth
        self.new_height = self.old_height // self.scale
        self.new_width = self.old_width // self.scale
        self.mask = np.zeros((self.old_depth, self.old_height, self.old_width))

    def forward(self, input_tensor):
        output_tensor = np.zeros((self.new_depth, self.new_height, self.new_width))
        for k in range(self.old_depth):
            for i in range(0, self.old_height, self.scale):
                for j in range(0, self.old_width, self.scale):
                    i_max = i
                    j_max = j
                    max = input_tensor[k, i, j]
                    for y in range(i, i + self.scale, 1):
                        for x in range(j, j + self.scale, 1):
                            value = input_tensor[k, y, x]
                            self.mask[k, y, x] = 0
                            if value > max:
                                max = value
                                i_max = y
                                j_max = x
                    output_tensor[k, i // self.scale, j // self.scale] = max
                    self.mask[k, i_max, j_max] = 1
        return output_tensor

    def backward(self, deltas, input_tensor):
        dX = np.zeros_like(input_tensor, dtype='float')
        for k in range(self.old_depth):
            for i in range(self.old_height):
                for j in range(self.old_width):
                    dX[k, i, j] = deltas[k, i // self.scale, j // self.scale] * self.mask[k, i, j]
        return dX


class FullyConnectedLayer:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.b = np.zeros((self.outputs,))
        self.db = np.zeros((self.outputs,))
        self.W = np.zeros((self.outputs, self.inputs))
        self.dW = np.zeros((self.outputs, self.inputs))
        self.df = np.zeros((self.outputs,))
        for i in range(self.outputs):
            for j in range(self.inputs):
                self.W[i, j] = random.uniform(-0.1, 0.1)
            self.b[i] = 0.1

    def activation(self, output):
        for i in range(self.outputs):
            output[i] = 1 / (1 + np.exp(-output[i]))
            self.df[i] = output[i] * (1 - output[i])

    def forward(self, inputed):
        output = np.zeros((self.outputs,))
        for i in range(self.outputs):
            sum = self.b[i]
            for j in range(self.inputs):
                sum += self.W[i, j] * inputed[j]
            output[i] = sum
        self.activation(output)
        return output

    def backward(self, deltas, inputed):
        for i in range(self.outputs):
            self.df[i] *= deltas[i]
        for i in range(self.outputs):
            for j in range(self.inputs):
                self.dW[i, j] = self.df[i] * inputed[j]
            self.db[i] = self.df[i]
        dX = np.zeros_like(inputed, dtype='float')
        for j in range(self.inputs):
            sum = 0
            for i in range(self.outputs):
                sum += self.W[i, j] * self.df[i]
            dX[j] = sum
        return dX

    def backward_final(self, outputed, inputed, need):
        global error
        deltas = np.zeros((self.outputs,))
        for i in range(self.outputs):
            deltas[i] = outputed[i] - need[i]
            error += (deltas[i] * deltas[i]) / 2
            self.df[i] *= deltas[i]
        for i in range(self.outputs):
            for j in range(self.inputs):
                self.dW[i, j] = self.df[i] * inputed[j]
            self.db[i] = self.df[i]
        dX = np.zeros_like(inputed, dtype='float')
        for j in range(self.inputs):
            sum = 0
            for i in range(self.outputs):
                sum += self.W[i, j] * self.df[i]
            dX[j] = sum
        return dX

    def update_weights(self, learning_rate):
        for i in range(self.outputs):
            for j in range(self.inputs):
                self.W[i, j] -= learning_rate * self.dW[i, j]
            self.b[i] -= learning_rate * self.db[i]



(train_X, train_Y), (test_X, test_Y) = mnist.load_data()
vector_train_Y = np.zeros((len(train_Y), 10))
vector_test_Y = np.zeros((len(test_Y), 10))
for i in range(len(train_Y)):
    vector_train_Y[i, train_Y[i]] = 1
for i in range(len(test_Y)):
    vector_test_Y[i, test_Y[i]] = 1
safe = np.zeros((len(train_X), 1, train_X.shape[1], train_X.shape[2]))
for i in range(len(train_X)):
    safe[i][0] = train_X[i]/255
train_X = safe.copy()
safe = np.zeros((len(test_X), 1, test_X.shape[1], test_X.shape[2]))
for i in range(len(test_X)):
    safe[i][0] = test_X[i]/255
test_X = safe.copy()

layer1 = ConvolutionalLayer(1, 28, 28, 16, 3)
layer2 = ConvolutionalLayer(layer1.new_depth, layer1.new_height, layer1.new_width, 16, 3)
layer3 = PoolingLayer(layer2.new_depth, layer2.new_height, layer2.new_width, 2)
layer4 = ConvolutionalLayer(layer3.new_depth, layer3.new_height, layer3.new_width, 32, 3)
layer5 = ConvolutionalLayer(layer4.new_depth, layer4.new_height, layer4.new_width, 32, 3)
layer6 = PoolingLayer(layer5.new_depth, layer5.new_height, layer5.new_width, 2)
layer7 = FullyConnectedLayer(layer6.new_depth * layer6.new_height * layer6.new_width, 128)
layer8 = FullyConnectedLayer(layer7.outputs, 10)


epoches = 10
epoch_error = np.zeros((epoches,))
for ep in range(1, epoches + 1):
    print("The epoch is ", ep)
    for t in range(len(train_X) - 59900):
        print(t)
        layer1_input = train_X[t].copy()
        instance = layer1.forward(train_X[t])
        layer1_activat_input = instance.copy()
        instance = layer1.activation_forward(instance)
        layer2_input = instance.copy()
        instance = layer2.forward(instance)
        layer2_activat_input = instance.copy()
        instance = layer2.activation_forward(instance)
        layer3_input = instance.copy()
        instance = layer3.forward(instance)
        layer4_input = instance.copy()
        instance = layer4.forward(instance)
        layer4_activat_input = instance.copy()
        instance = layer4.activation_forward(instance)
        layer5_input = instance.copy()
        instance = layer5.forward(instance)
        layer5_activat_input = instance.copy()
        instance = layer5.activation_forward(instance)
        layer6_input = instance.copy()
        instance = layer6.forward(instance)

        instance_array = np.zeros((instance.shape[0] * instance.shape[1] * instance.shape[2],))
        index = -1
        for k in range(instance.shape[0]):
            for i in range(instance.shape[1]):
                for j in range(instance.shape[2]):
                    index += 1
                    instance_array[index] = instance[k, i, j]

        layer7_input = instance_array.copy()
        instance_array = layer7.forward(instance_array)
        layer8_input = instance_array.copy()
        instance_array = layer8.forward(instance_array)
        print(instance_array)
        error = 0
        my_deltas = layer8.backward_final(instance_array, layer8_input, vector_train_Y[t])
        epoch_error[ep - 1] += error
        layer8.update_weights(0.4)
        my_deltas = layer7.backward(my_deltas, layer7_input)
        layer7.update_weights(0.4)

        deltas = np.zeros_like(instance, dtype='float')
        index = -1
        for k in range(instance.shape[0]):
            for i in range(instance.shape[1]):
                for j in range(instance.shape[2]):
                    index += 1
                    deltas[k, i, j] = my_deltas[index]

        deltas = layer6.backward(deltas, layer6_input)
        deltas = layer5.activation_backward(layer5_activat_input, deltas)
        deltas = layer5.backward(layer5_input, deltas)
        layer5.update_weights(0.4)
        deltas = layer4.activation_backward(layer4_activat_input, deltas)
        deltas = layer4.backward(layer4_input, deltas)
        layer4.update_weights(0.4)
        deltas = layer3.backward(deltas, layer3_input)
        deltas = layer2.activation_backward(layer2_activat_input, deltas)
        deltas = layer2.backward(layer2_input, deltas)
        layer2.update_weights(0.4)
        deltas = layer1.activation_backward(layer1_activat_input, deltas)
        deltas = layer1.backward(layer1_input, deltas)
        layer1.update_weights(0.4)
        print('MISTAKE')
        print(epoch_error)