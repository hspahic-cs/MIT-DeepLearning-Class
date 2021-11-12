import numpy as np
import random
import math

class SinglePerceptron:
    def __init__(self, input, weights, bias, activation):
        self.input = np.asarray(input)
        self.weights = np.asarray(weights)
        self.bias = bias
        self.activation = activation
    
    def __str__(self):
        return f"Perceptron Values:\n INPUT = {self.input}\n WEIGHTS = \n{self.weights}\n BIAS = {self.bias}\n ACTIVATION = {self.activation}\n RESULT = {self.runPerceptron()}\n"
    
    def runPerceptron(self):
        result = np.dot(self.input, self.weights) + self.bias
        return chooseActivation(self.activation, result)


def chooseActivation(activationFx, x):
    match activationFx:
        case "sigmoid":
            return sigmoidFunction(x)
        case "hyperbolic":
            return hyperbolicTangent(x)
        case "RLU":
            return RLU(x)
        case _:
            raise ValueError("Invalid activation function, please enter one of the following: [sigmoid, hyperbolic, RLU] ")

def sigmoidFunction(x):
    return 1 / (1 + math.exp(-x))

def hyperbolicTangent(x):
    return ((math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)))

def RLU(x):
    return math.max(0, x)



if __name__ == "__main__":
    myValues = [i for i in range(10)]
    myWeights = [random.random() for i in range(10)]
    bias = -25
    activationfx = "sigmoid"
    sample = SinglePerceptron(myValues, myWeights, bias, activationfx)
    print(sample)
    # print("Inputs: ", sample.input)
    # print("Weights: " , sample.weights)
    # print("Bias: " , sample.bias)
    # print("Activation: " , sample.activation)
    # print("Perceptron: " , sample.runPerceptron())
    
