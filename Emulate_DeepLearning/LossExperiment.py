import DenseLayer as DL
from mpl_toolkits import mplot3d


'''
The goal of the "LossExperiment.py" file is to play around with determining optimal loss 
with very small numbers of weights.

# Experiment 1: visualize loss space given 2 weights, plot loss space
'''

w1, w2, loss = [], [], []

def meanSquaredErrorLoss(actual, predicted):
    if(actual.size() != predicted.size()):
        print("Sizes of test set & predicted are not the same")    
    return sum(list(map(lambda a, b: a - b, actual, predicted))) / actual.size()

