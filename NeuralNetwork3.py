# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 09:31:40 2020

@author: queek
"""

# =============================================================================
#  1. Split the dataset into batches 
#  2. Initialize weights and bias 
#  3. Select one batch of data​ ​and calculate forward pass​ - 
#     follow the basic structure of neural network to compute output for each layer,
#     you might need to cache output of each layer for the convenience of backward propagation. 
#  4. Compute loss function​ - you need to use cross-entropy (logistic loss - see references above) as loss function  
#  5. Backward propagation ​- use backward propagation (your implementation)  to update hidden weights 
#  6. Updates weights using optimization algorithms​ - there are many ways to update 
#     weights you can use plain SGD or advanced methods such as Momentum and Adam.
#     (but you can get full credit easily without any advanced methods) 
#  7. Repeat 2,3,4,5,6 for all batches​ - after finishing this process for all 
#     batches (it just iterates all data points of dataset), it is called ‘one epoch’.  
#  8. Repeat 2,3,4,5,6,7 number of epochs times​- You might need to train many epochs to get
#     a good result. As an option, you may want to print out the accuracy of your network at the end of each epoch. 
# =============================================================================
import numpy as np
import time
import matplotlib.pyplot as plt
import random

    

def main():
    start_time = time.time()
    print("Start!")
    
    numOfHiddenNodes = 88
    numOfOutputNodes = 10
    numOfHidden2Nodes = 20
    
    learningRate = float(1/100)#float(1/50000)
    trainImages = np.genfromtxt('train_image.csv',delimiter=',',dtype=float)
    trainImages = np.divide(trainImages,(784*255/6))
    numOfInputNodes = len(trainImages[0])
    numOfSplits = int(len(trainImages)/10)
    #trainImages = [2,1,3,4,53]
    imageBatch = np.array_split(trainImages, numOfSplits)
    #print(len(batch))
    txtTrainLabels = np.genfromtxt('train_label.csv',delimiter=',',dtype=float)
    txtTestImages = np.genfromtxt('test_image.csv',delimiter=',',dtype=float)
    txtTestLabels = np.genfromtxt('test_label.csv',delimiter=',',dtype=float)
    
    trainLabels = []
    for trainLabel in txtTrainLabels:
        row = [float(0) for i in range(numOfOutputNodes)]
        row[int(trainLabel)] = 1
        trainLabels.append(row)
        
        
    #trainLabels = [3,4,4,26,6,4,22]
    labelBatch = np.array_split(trainLabels, numOfSplits)


    bias = float(-(0))
    hiddenWeights = [[random.random() for i in range(numOfInputNodes)] for i in range(numOfHiddenNodes)]
    hidden2Weights = []#[[random.random() for i in range(numOfHiddenNodes)] for i in range(numOfHidden2Nodes)]
    outputWeights = [[random.random() for j in range(numOfHiddenNodes)]for i in range(numOfOutputNodes)]
    CES = []
    epochs = []
    print("--- %s minutes to load data ---" % ((time.time() - start_time)/60))
    start_time = time.time()
    epoch = 0
    while epoch <20:
        CE = float(0.0)
        for batch in range(numOfSplits):
            #first feed forward for batch
            

            #h2_weightChange = np.array([[float(0.0) for j in range(numOfInputNodes)] for i in range(numOfHiddenNodes)])
            h_weightChange = np.array([[float(0.0) for j in range(numOfInputNodes)] for i in range(numOfHiddenNodes)])
            o_weightChange = np.array([[float(0.0) for j in range(numOfHiddenNodes)]for i in range(numOfOutputNodes)])
            #for image in range(len(imageBatch[batch])):

            hidden_node_values, o_node_values,softmax,tLabels, hidden2_node_values, CE = feedforward(numOfInputNodes,numOfHiddenNodes,numOfHidden2Nodes, numOfOutputNodes, hiddenWeights, hidden2Weights, outputWeights, bias, labelBatch[batch],\
                        imageBatch[batch],0)
            
            #CES = CES.transpose()
            #hiddenWeights = [hiddenNodes[i].n_weight for i in range(len(hiddenNodes))]
            #outputWeights = [outputNodes[i].n_weight for i in range(len(outputNodes))]
            ##then backpropogate
            #hiddenErrors = []
            h_weightChange, h2_weightChange, o_weightChange = backpropogate(hidden_node_values, hidden2_node_values, o_node_values, softmax,tLabels,hiddenWeights, hidden2Weights, outputWeights,learningRate,\
                          imageBatch[batch],numOfInputNodes, numOfHiddenNodes,h_weightChange, o_weightChange)
            
            hiddenWeights = np.subtract(hiddenWeights, (h_weightChange))
            
            #hidden2Weights = np.add(hidden2Weights, -(h2_weightChange))

            outputWeights = np.subtract(outputWeights,(o_weightChange))

            if batch == 0:
                CES.append(CE)
                epochs.append(epoch)
                print("epoch %s" % epoch)
                print("error %s" % CE)
            #print("batch num: %s" % batch)
            #print("label was %s" %tLabels[0])
            #print(softmax[0])"""
        epoch +=1
    plt.plot(epochs,CES)
    plt.show()
    print("--- %s minutes to learn---" % ((time.time() - start_time)/60))
    start_time = time.time()
            #figure out step size for updating weights
            #store new weights
        #get the accuracy% at the end of epoch
    #repeat"""
    
    hidden_node_values, o_node_values,softmax,tLabels, hidden2_node_values, CE = feedforward(numOfInputNodes,numOfHiddenNodes,numOfHidden2Nodes, numOfOutputNodes, hiddenWeights, hidden2Weights, outputWeights, bias, trainLabels,\
    np.divide(txtTestImages,784*255/6),1) 

    
    f = open("test_predictions.csv","w")
    string =''
    output = np.apply_along_axis(returnLabels, 1, o_node_values)
    string = '\n'.join([str(elem) for elem in output])
    f.write(string)
    f.close()
            
    
class Node:
    def __init__(self, in_value, out_value, n_weight):
        self.in_value = in_value
        self.out_value = out_value
        self.n_weight = n_weight
        
    def sigmoid_fxn(self, x, y):
        return 1/(1+np.exp(-(x/y)))
    
    def resetNode(self):
        self.in_value = float(0.0)
        self.out_value = float(0.0)

def softmax_fxn(x):
        """Compute softmax values for each sets of scores in x."""
        if np.sum(np.exp(x), axis=0) == 0:
            print("softmax sum is 0")
        return np.exp(x) / np.sum(np.exp(x), axis=0) 

def feedforward(numOfInputNodes,numOfHiddenNodes, numOfHidden2Nodes, numOfOutputNodes, hiddenWeights, hidden2Weights, outputWeights, bias, label,image, runType):
    #if not last node calculate output based on sigmoid function
    #if last node calculate output based on softmax function
    hidden_node_inputs = np.dot(hiddenWeights,np.transpose(image))
    hidden_node_inputs = np.transpose(hidden_node_inputs)
    #get output nodes 
    hidden_node_inputs = np.add(hidden_node_inputs,bias)
    vsigmoid = np.vectorize(sigmoid_fxn)
    hidden_node_values = vsigmoid(hidden_node_inputs,1)
    
    hidden2_node_values = []#vsigmoid(hidden2_node_inputs,1)
    
    o_node_values = np.dot(outputWeights, np.transpose(hidden_node_values))
    o_node_values = np.transpose(o_node_values)
    #store softmax
    #o_node_outputs.append(o_node_values)
    #hidden_node_outputs.append(hidden_node_values)
    softmax = np.apply_along_axis(softmax_fxn, 1, o_node_values)
    
    
    
    #calculate cross entropy for each row 
    #o_node_inputs = o_node_inputs.transpose()
    #softmax = softmax.transpose()
    tLabels = np.array(label)
    #tLabelBatch = tLabelBatch.transpose()
    #CES = np.array([])
    CE = 0
    if runType == 0:    
        CE = -np.multiply(tLabels,np.log(softmax))
        CE= np.sum(CE,axis=1)
        CE = np.average(np.transpose(CE))
    o_node_values = np.asarray(o_node_values)
    softmax = np.asarray(softmax)
    tLabels = np.asarray(tLabels)
    hidden_node_values = np.asarray(hidden_node_values)
    hidden2_node_values = np.asarray(hidden2_node_values)
    
    return hidden_node_values, o_node_values,softmax,tLabels, hidden2_node_values, CE

def sigmoid_fxn(x, y):
    return 1/(1+np.exp(-(x)))

def cross_entropy_fxn(desiredOutput, actualOutput):
    return -np.multiply(desiredOutput, np.log(actualOutput))
    
    

def backpropogate(hidden_node_values, hidden2_node_values, o_node_values, softmax,tLabels,hiddenWeights, hidden2Weights, outputWeights,learningRate, image,\
                  numOfInputNodes, numOfHiddenNodes, h_weightChange, o_weightChange):
    #if the first node calculate 
    
    
     #figure out weights into output
    temp = []
    outputErrorSums = []
    outputError = np.subtract(softmax,tLabels)
    outputErrorSums = outputError
    
    #np.multiply(np.transpose(hidden2_node_values),outputError)
    #take each output array in samples multiply by each value in erorrs
    temp = np.transpose(np.dot(learningRate,np.dot(np.transpose(hidden_node_values),outputError)))
    
    
    temp = np.asarray(temp)        
    o_weightChange = temp
    
    
    #figure out weights into hidden2 layer
 
    h2_weightChange = []#temp
    
    
    temp = []
    #hiddenError = np.sum(hidden2ErrorSums)
    hiddenError = np.dot(np.transpose(outputWeights), np.transpose(outputErrorSums))
    #hiddenError = np.transpose(hiddenError)
    #hiddenError = np.multiply(hidden_node_values, np.transpose(hiddenError))
    temp = np.transpose(np.dot(learningRate, np.dot(np.transpose(image),np.multiply(np.multiply(hidden_node_values, np.transpose(hiddenError)),1-hidden_node_values))))
    
    temp = np.asarray(temp)
    #temp = np.sum(temp,axis=2)
    
    h_weightChange = temp
       
    
    return h_weightChange, h2_weightChange, o_weightChange

def returnLabels(x):
    
    return str(x.tolist().index(np.amax(x)))

    

if __name__ == '__main__':
    main()

