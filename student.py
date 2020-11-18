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
import math
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

    for i in range(len(sample)):
        #sample[i] = sample[i].replace(".", "")
        #sample[i] = sample[i].replace(",", "")
        #sample[i] = sample[i].replace("!", "")
        #sample[i] = sample[i].replace("-", "")
        sample[i] = ''.join([letter for letter in sample[i] if letter.isalpha()])
    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

stopWords = {"a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"}
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

    softmax = tnn.Softmax(dim=1)
    sigmoid = tnn.Sigmoid()
    #return torch.round(ratingOutput).long(), torch.argmax(categoryOutput, 1)
    return torch.round(sigmoid(ratingOutput)).long(), torch.argmax(softmax(categoryOutput), 1)
    #return torch.argmax(ratingOutput, 1), torch.argmax(categoryOutput, 1)
    #return np.argmax(ratingOutput), np.argmax(categoryOutput)

################################################################################
###################### The following determines the model ######################
################################################################################

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
        #ratingLoss = tnn.NLLLoss()
        catLoss = tnn.CrossEntropyLoss()
        loss1 = ratingLoss(ratingOutput.squeeze(1), ratingTarget.float())
        loss2 = catLoss(categoryOutput, categoryTarget)
        #print(loss1, loss2)
        # return (loss1 + loss2) 
        # return (loss1 + loss2)/2
        return torch.log(loss1 + loss2)

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

'''
#pretty cool
class network(tnn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.rnn1 = tnn.LSTM(50,
                            128,
                            num_layers=2,
                            bidirectional=True,
                            dropout=0.5)
        self.fc1 = tnn.Linear(128*2, 1)
        self.dropout1 = tnn.Dropout(0.5)

        self.rnn2 = tnn.LSTM(50,
                            128,
                            num_layers=2,
                            bidirectional=True,
                            dropout=0.5)
        self.fc2 = tnn.Linear(128*2, 5)
        self.dropout2 = tnn.Dropout(0.5)

    def forward(self, input, length):
        input = input.float() # shape [batch size, sentence length, embedding size]
        #print(input[0].shape)
        # reshape it to form [sentence length, batch size, embedding size]
        #input = input.reshape([-1, length.shape[0], 50])
        input = torch.transpose(input, 0, 1)
        dropout1 = self.dropout1(input)
        dropout2 = self.dropout2(input)

        output1, (hidden1, _) = self.rnn1(dropout1)
        output2, (hidden2, _) = self.rnn2(dropout2)
        #print(hidden1.shape)
        hidden1 = self.dropout1(torch.cat((hidden1[-2,:,:], hidden1[-1,:,:]), dim=1))
        hidden2 = self.dropout2(torch.cat((hidden2[-2,:,:], hidden2[-1,:,:]), dim=1))
        #print(hidden1.shape, hidden2.shape)

        #return sigmoid(self.fc1(hidden1)), softmax(self.fc2(hidden2))
        return self.fc1(hidden1), self.fc2(hidden2)
        #return self.fc1(torch.cat((hidden1[-2,:,:], hidden1[-1,:,:]), dim=1)), self.fc2(torch.cat((hidden2[-2,:,:], hidden2[-1,:,:]), dim=1))
'''
#pretty cool
class network(tnn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.rnn1 = tnn.LSTM(300,
                            256,
                            num_layers=3,
                            bidirectional=True,
                            dropout=0.6)
        self.fc1 = tnn.Linear(256*2, 1)
        self.dropout1 = tnn.Dropout(0.6)

        self.rnn2 = tnn.LSTM(300,
                            256,
                            num_layers=3,
                            bidirectional=True,
                            dropout=0.6)
        self.fc2 = tnn.Linear(256*2, 5)
        self.dropout2 = tnn.Dropout(0.6)

    def forward(self, input, length):
        input = input.float() # shape [batch size, sentence length, embedding size]
        #print(input[0].shape)
        # reshape it to form [sentence length, batch size, embedding size]
        #input = input.reshape([-1, length.shape[0], 50])
        input = torch.transpose(input, 0, 1)
        dropout1 = self.dropout1(input)
        dropout2 = self.dropout2(input)

        output1, (hidden1, _) = self.rnn1(dropout1)
        output2, (hidden2, _) = self.rnn2(dropout2)
        #print(hidden1.shape)
        hidden1 = self.dropout1(torch.cat((hidden1[-1,:,:], hidden1[-2,:,:]), dim=1))
        hidden2 = self.dropout2(torch.cat((hidden2[-1,:,:], hidden2[-2,:,:]), dim=1))
        #print(hidden1.shape, hidden2.shape)

        #return sigmoid(self.fc1(hidden1)), softmax(self.fc2(hidden2))
        return self.fc1(hidden1), self.fc2(hidden2)
        #return self.fc1(torch.cat((hidden1[-2,:,:], hidden1[-1,:,:]), dim=1)), self.fc2(torch.cat((hidden2[-2,:,:], hidden2[-1,:,:]), dim=1))


'''
class convNet(cnn.Module):
    pass
    def __init__(self):
        super(convNet, self).__init__()

    def forward(self):
        pass

'''
'''
# pretty shit ngl
class network(tnn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.ratingRNN = tnn.RNN(input_size=50, hidden_size=200, num_layers=2, bidirectional=True)
        self.ratingFc = tnn.Linear(800, 1)

        self.catRNN = tnn.RNN(input_size=50, hidden_size=200, num_layers=2, bidirectional=True)
        self.catFc = tnn.Linear(800, 5)

    def forward(self, input, length):
        input = input.float() # shape [batch size, sentence length, embedding size]
        # reshape it to form [sentence length, batch size, embedding size]
        input = torch.transpose(input, 0, 1)
        ratingOutput, ratingHidden = self.ratingRNN(input)
        catOutput, catHidden = self.catRNN(input)
        #print(ratingHidden.shape, catHidden.shape, length.shape)
        sigmoid = tnn.Sigmoid()
        softmax = tnn.Softmax(dim=1)
        ratingHidden = torch.cat((ratingHidden[-4,:,:], ratingHidden[-3,:,:], ratingHidden[-2,:,:], ratingHidden[-1,:,:]), dim=1)
        catHidden = torch.cat((catHidden[-4,:,:], catHidden[-3,:,:], catHidden[-2,:,:], catHidden[-1,:,:]), dim=1)
        #print(ratingHidden.shape, catHidden.shape)
        return self.ratingFc(ratingHidden.squeeze(0)), self.catFc(catHidden.squeeze(0))
        #return sigmoid(self.ratingFc(ratingHidden)), softmax(self.catFc(catHidden))
'''
net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.9
batchSize = 64
epochs = 20
#optimiser = toptim.SGD(net.parameters(), lr=0.01)
#optimiser = toptim.Adam(net.parameters(), lr=0.0005)
optimiser = toptim.Adam(net.parameters(), lr=0.001)

# 79.26 lr=0.005 dropout=0.5 arch=lstm->linear
# 83.91 lr=0.0005 dropout=0.5 same arch + preprocessing 20 epochs
# 40 epochs went about 0.5 better score

# TrainSplit: 0.9,300d, 10 epoches, adam 0.001 lr --> 86.9



# layers
# Adam lr
# loss func
# trainValSplit
# epochs
