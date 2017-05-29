from math import sqrt
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

input = np.loadtxt("./data/wine_input.asc")
target = np.loadtxt("./data/wine_desired.asc")

def getTargetList(target):
    t = np.zeros(len(target), dtype=int)
    for index, i in enumerate(target):
        num = list(i).index(1.)
        #num = 3 - num
        t[index] = num
    return t


t = getTargetList(target)

EPOCHS = 100
LEARNING_RATE = 0.5
NEIGHBOURHOOD_RADIUS = 1
#SIGMA = 1
MAP_SIZE = [50, 50] # rows x columns
input_len = 13

data = input

def decayLR(epoch):
    return LEARNING_RATE/(1+ epoch/(EPOCHS/2))

def calculateActivationMap(one_data, activation_map):
    for i in range(MAP_SIZE[0]):
        for j in range(MAP_SIZE[1]):
            s = one_data - weightsMap[i][j]
            sum = 0
            for k in range(len(s)):
                sum += s[k]*s[k]
            activation_map[i][j] = sqrt(sum)
            
def findIndexOfBestMatch(one_data):
    activation_map = np.zeros((MAP_SIZE[0], MAP_SIZE[1]))
    calculateActivationMap(one_data, activation_map)
    return np.unravel_index(activation_map.argmin(), activation_map.shape)

def updateWeights(epoch, min_index, data_i):
    i_min, j_min = min_index
    for i in range((i_min - NEIGHBOURHOOD_RADIUS), (i_min + NEIGHBOURHOOD_RADIUS)):
        for j in range((j_min - NEIGHBOURHOOD_RADIUS), (j_min + NEIGHBOURHOOD_RADIUS)):
            weightsMap[i][j] = weightsMap[i][j] + decayLR(epoch)*(data_i - weightsMap[i][j]) 


def train(epoch, data):
    global weightsMap
    for data_i in data:
        min_index = findIndexOfBestMatch(data_i)
        updateWeights(epoch, min_index, data_i)

weightsMap = np.random.rand(MAP_SIZE[0], MAP_SIZE[1], input_len)

for epoch in range(EPOCHS):
    train(epoch, data)

for x,i in zip(input,t): # scatterplot
    w = findIndexOfBestMatch(x)
    plt.text(w[0], w[1], str(i), color=plt.cm.Dark2(i / 4.), fontdict={'weight': 'bold', 'size': 11})
plt.axis([0,MAP_SIZE[0]+2,0, MAP_SIZE[1]+2])
plt.show()
