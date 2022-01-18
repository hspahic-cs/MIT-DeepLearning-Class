import csv
import DenseLayer as DL
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

'''
The goal of the "LossExperiment.py" file is to play around with determining optimal loss 
with very small numbers of weights.

# Experiment 1: visualize loss space given 2 weights, plot loss space
'''

w1_list, w2_list = np.linspace(0, 1, 100), np.linspace(0, 1, 100)

def parseData(file):
    with open(file, newline='') as f: 
        cocoa, year, rating  = [], [], []
        line_reader = csv.DictReader(f, delimiter = ',')
        for line in line_reader:
            cocoa.append(float(line['Cocoa Percent'][:-1])/10)
            #Seems like numbers in the thousands overwhelm weights of step 0.01 --> divide by 100
            year.append((int(line['Review Date']) - 2000) / 10)
            rating.append(float(line['Rating']))
    
    return cocoa, year, rating

def meanSquaredErrorLoss(actual, predicted):
    if(len(actual) != len(predicted)):
        print("Sizes of test set & predicted are not the same")
    return sum(list(map(lambda a, b: (a - b)**2, actual, predicted))) / len(actual)

# data in form of truple --> ([cocoa], [year], [rating])
def calcLoss(w1, w2, data):
    actual = data[2]
    predicted = []
    weights = [w1, w2]

    for x in range(len(data[0])):
        input = [data[0][x], data[1][x]]

        # Minimum rating is 1 --> bias is 1 
        # Range of ratings is 3 (max = 4, min = 1) thus multiply result of activation by 3
        temp = DL.SinglePerceptron(input, weights, 1, "sigmoid").runPerceptron() * 3 + 1
        predicted.append(temp)
        
    return meanSquaredErrorLoss(actual, predicted)

def plotLoss(w1_list, w2_list):
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})

    data = parseData('chocolate_ratings.csv')
    loss = []
    for w1 in w1_list:
        for w2 in w2_list:
            loss.append(calcLoss(w1, w2, data))
    
    X, Y = np.meshgrid(w1_list, w2_list)
    
    Loss = (np.asarray(loss).reshape(len(w1_list), len(w2_list)))
    surf = ax.plot_surface(X, Y, Loss, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    
    ax.set_xlabel('W1')
    ax.set_ylabel('W2')
    ax.set_zlabel('Loss')
    
    plt.show()


