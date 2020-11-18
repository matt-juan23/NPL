#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import numpy as np
# import sklearn

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split()

    return processed

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """

    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

stopWords = {}
wordVectors = GloVe(name='6B', dim=300)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """

    #softmax = tnn.Softmax(dim=1)
    return torch.round(ratingOutput).long, torch.argmax(categoryOutput, 1)
    #return torch.round(torch.sigmoid(ratingOutput)).long(), torch.argmax(softmax(categoryOutput), 1)
    #return torch.argmax(ratingOutput, 1), torch.argmax(categoryOutput, 1)
    #return np.argmax(ratingOutput), np.argmax(categoryOutput)

################################################################################
###################### The following determines the model ######################
################################################################################
'''
# REVISION 1
class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()
        self.r1 = tnn.RNN(50, 200)
        self.r2 = tnn.RNN(50, 200)

        self.l1 = tnn.Linear(200, 1)
        self.l2 = tnn.Linear(200, 5)

    def forward(self, input, length):
        input = input.float()
        #print(input.shape, length.shape)
        # reshape it to form [sentence length, batch size, embedding size]
        out1, hid1 = self.r1(input.reshape([-1, length.shape[0], 50]))
        out2, hid2 = self.r2(input.reshape([-1, length.shape[0], 50]))
        #print(hid1.shape, hid2.shape)
        softmax = tnn.Softmax(dim=1)
        #sigmoid = tnn.Sigmoid()
        #print((self.l1(hid1.squeeze(0))))
        return self.l1(hid1.squeeze(0)), softmax(self.l2(hid2.squeeze(0)))
'''

class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        #ratingOutput, categoryOutput, ratingTarget, categoryTarget = ratingOutput.float(), categoryOutput.float(), ratingTarget.float(), categoryTarget.float()
        ratingLoss = tnn.BCEWithLogitsLoss()
        catLoss = tnn.CrossEntropyLoss()
        #print(ratingOutput.dtype, ratingTarget.dtype, categoryOutput.dtype, categoryTarget.dtype)
        #print(ratingOutput.shape, categoryOutput.shape)
        loss1 = ratingLoss(ratingOutput.squeeze(1), ratingTarget.float())
        loss2 = catLoss(categoryOutput, categoryTarget)
        return loss1 + loss2


class network(tnn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.rnn1 = tnn.LSTM(300,
                            200,
                            num_layers=2,
                            bidirectional=True,
                            dropout=0.5)
        self.fc1 = tnn.Linear(200*2, 1)
        self.dropout1 = tnn.Dropout(0.5)

        self.rnn2 = tnn.LSTM(300,
                            200,
                            num_layers=2,
                            bidirectional=True,
                            dropout=0.5)
        self.fc2 = tnn.Linear(200*2, 5)
        self.dropout2 = tnn.Dropout(0.5)

    def forward(self, input, length):
        input = input.float() # shape [batch size, sentence length, embedding size]
        #print(input[0].shape)
        # reshape it to form [sentence length, batch size, embedding size]
        #input = input.reshape([-1, length.shape[0], 50])
        input = torch.transpose(input, 0, 1)
        dropout1 = self.dropout1(input)
        dropout2 = self.dropout2(input)

        output1, (hidden1, cell1) = self.rnn1(dropout1)
        output2, (hidden2, cell2) = self.rnn2(dropout2)
        #print(hidden1.shape)
        hidden1 = self.dropout1(torch.cat((hidden1[-2,:,:], hidden1[-1,:,:]), dim=1))
        hidden2 = self.dropout2(torch.cat((hidden2[-2,:,:], hidden2[-1,:,:]), dim=1))
        #print(hidden1.shape, hidden2.shape)

        sigmoid = tnn.Sigmoid()
        softmax = tnn.Softmax(dim=1)
        return self.fc1(hidden1), self.fc2(hidden2)

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
#optimiser = toptim.SGD(net.parameters(), lr=0.01)
optimiser = toptim.Adam(net.parameters(), lr=0.005)

# 79.26 lr=0.005 dropout=0.5 arch=lstm->linear


