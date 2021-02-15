import numpy as np
import math

def sigmoid_function(val):
    return (1 / (1 + math.exp(-val)))

def htan_function(val):
    return ((math.exp(val) - math.exp(-val)) / (math.exp(val) + math.exp(-val)))

def rec_lin_function(val):
    if val > 0:
        return 1
    else:
        return 0

class DenseLayer:
    '''
    @ weights : np.array of dimension [# of perceptrons][# of variables] holding weights
    @ inputs : np.array --> same dim as weights --> holding data inputs
    '''
    def __init__(self, weights):
        self.weights = weights
        self.bias = np.random.rand(weights.shape[1], 1)

    def call(self, inputs):
        for i in range(len(self.weights)):
            result = htan_function((np.matmul(inputs[i], weights[i].transpose()) + self.bias[i]))
            print(f"Result for preceptron {i + 1}: {result}")

if __name__ == "__main__":
    weights = np.random.rand(3,4)
    inputs = np.random.rand(3,4)#.reshape(3,4)
    test_layer = DenseLayer(weights)
    test_layer.call(inputs)
