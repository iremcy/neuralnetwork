#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import random
import math
from termcolor import colored

### requirements ###
data = []
train_data = []
test_data = []
norm_data = []
weights_ih = []
weights_ho = []
hidden = []
output = []
min_max_array = 0,0,0,0,0,0
n = 0.5 # learnin rate

### set data ###
content = open("data.txt", "r")
lines = content.readlines()

for line in lines:
    data.append(line)

random.shuffle(data)

train_data = data[:int(len(data)*80/100)]
test_data = data[int(len(data)*80/100):]

### initialize random weigths between input nodes - hidden layer ###
a, b = 3, 3
weights_ih = [[0 for x in range(a)] for y in range(b)]

for i in range(0,3):
    for j in range(0,3):
        weights_ih[i][j] = random.random()

### initialize random weigths between hidden layer - output node ###
a, b = 1, 3
weights_ho = [[0 for x in range(a)] for y in range(b)]

for i in range(0,3):
    for j in range(0,1):
        weights_ho[i][j] = random.random()

### sigmoid function ###
def sigmoid(x):
    x = 1 / (1 + math.exp(-x))
    return x

### back propagation algorithm between hidden layer and output layer ###
def back_propagation_ho(old_weight, err, y_out, y_hid):
    new_weight = old_weight - (n * err * (-1) * (y_out * (1 - y_out)) * y_hid)
    return new_weight

### back propagation algorithm between input layer and hidden layer ###
def back_propagation_ih(old_weight, err, y_out, y_hid, i, w):
    new_weight = old_weight - (n * err * (-1) * (y_out * (1 - y_out)) * weights_ho[w][0] * (y_hid * (1 - y_hid)) * i)
    return new_weight

### min_max function ###
def min_max_func(lines):
    min1 = 100
    min2 = 100
    min3 = 100
    max1 = 0
    max2 = 0
    max3 = 0
    for line in lines:
        a, b, c, d = line.split(",")
        if(int(a) < min1):
            min1 = int(a)
        if(int(a) > max1):
            max1 = int(a)
        if(int(b) < min2):
            min2 = int(b)
        if(int(b) > max2):
            max2 = int(b)
        if(int(c) < min3):
            min3 = int(c)
        if(int(c) > max3):
            max3 = int(c)
    min_max_array = min1, max1, min2, max2, min3, max3
    return min_max_array

min_max_array = min_max_func(lines)

### normalization function ###
def norm_func(value, order_num, min_max_array):
    if(order_num == 1):
        norm = (float(value) - float(min_max_array[0]))/float((min_max_array[1]) - float(min_max_array[0]))
    if(order_num == 2):
        norm = (float(value) - float(min_max_array[2]))/float((min_max_array[3]) - float(min_max_array[2]))
    if(order_num == 3):
        norm = (float(value) - float(min_max_array[4]))/float((min_max_array[5]) - float(min_max_array[4]))
    return norm

def train_func(train_data, min_max_array):
    ### train neural network ###
    for j in range(0, len(train_data)):
        split_data = train_data[j].replace("\n", "").split(",")  # split data
        norm_data.insert(0, norm_func(split_data[0], 1, min_max_array))
        norm_data.insert(1, norm_func(split_data[1], 2, min_max_array))
        norm_data.insert(2, norm_func(split_data[2], 3, min_max_array))

        ### input * weight ###
        Vh = 0
        for i in range(0, 3):
            for j in range(0, 3):
                Vh += weights_ih[j][i] * norm_data[j]
            Yh = sigmoid(Vh)
            hidden.insert(i, Yh)

        Vo = 0
        for i in range(0, 3):
            Vo += weights_ho[i][0] * hidden[i]
        Yo = sigmoid(Vo)
        output.insert(0, Yo)

        ### error rate ###
        e = float(split_data[3]) - float(output[0])

        for j in range(0, 3):
            for i in range(0, 3):
                weights_ih[i][j] = back_propagation_ih(weights_ih[i][j], e, output[0], hidden[i], norm_data[i], j)

        ### apply back propagation algoritm ###
        for i in range(0, 3):
            weights_ho[i][0] = back_propagation_ho(weights_ho[i][0], e, output[0], hidden[i])



    return e

def test_funct():
    trueness = 0
    ### perform test data ###
    for j in range(0, len(test_data)):
        split_data = test_data[j].replace("\n", "").split(",")  # split data
        norm_data.insert(0, norm_func(split_data[0], 1, min_max_array))
        norm_data.insert(1, norm_func(split_data[1], 2, min_max_array))
        norm_data.insert(2, norm_func(split_data[2], 3, min_max_array))

        ### input * weight ###
        Vh = 0
        for i in range(0, 3):
            for j in range(0, 3):
                Vh += weights_ih[j][i] * norm_data[j]
            Yh = sigmoid(Vh)
            hidden.insert(i, Yh)

        Vo = 0
        for i in range(0, 3):
            Vo += weights_ho[i][0] * hidden[i]
        Yo = sigmoid(Vo)
        output.insert(0, Yo)

        ### error rate ###
        e = float(split_data[3]) - float(output[0])

        if int(split_data[3]) == 1 and output[0] > 0.6:
            trueness += 1
        elif int(split_data[3]) == 0 and output[0] <= 0.4:
            trueness += 1

    print "Data:", len(test_data)
    print len("True: ")*"-"
    print colored("True: ","green"), trueness
    print colored(len("True: ")*"-","green")
    print colored("False: ", "red"), len(test_data) - trueness
    print colored(len("False: ")*"-","red")
    print colored("Accuracy: ", "yellow"), "%", int(float(float(trueness)/float(len(test_data))*100))
    print colored(len("Accuracy: ")*"-", "yellow")

def main():
    for i in range(0, 25):
        print "[" + colored("+","green") + "] ", i+1, ". epoch"
        print colored("------------------------\n", "red")
        random.shuffle(train_data)
        e_son = train_func(train_data, min_max_array)

    test_funct()

main()
