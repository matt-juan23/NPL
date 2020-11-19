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

        sample[i] = ''.join([letter for letter in sample[i] if letter.isalpha() or letter.isnumeric()])
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
    return torch.round(sigmoid(ratingOutput)).long(), torch.argmax(softmax(categoryOutput), 1)

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
        ratingLoss = tnn.BCEWithLogitsLoss()
        catLoss = tnn.CrossEntropyLoss()
        loss1 = ratingLoss(ratingOutput.squeeze(1), ratingTarget.float())
        loss2 = catLoss(categoryOutput, categoryTarget)
        return loss1 + loss2
''' Justification
LSTM
Loss function
'''

class network(tnn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.rnn1 = tnn.LSTM(300,
                            256,
                            num_layers=2,
                            bidirectional=True,
                            batch_first=True,
                            dropout=0.6)

        self.rnn1fc1 = tnn.Linear(256*2, 128)
        self.rnn1fc2 = tnn.Linear(128, 1)

        self.dropout1 = tnn.Dropout(0.6)




        self.rnn2 = tnn.LSTM(300,
                            256,
                            num_layers=2,
                            bidirectional=True,
                            batch_first=True,
                            dropout=0.6)

        self.rnn2fc1 = tnn.Linear(256*2, 256*2)
        self.rnn2fc2 = tnn.Linear(256*2, 5)

        self.dropout2 = tnn.Dropout(0.6)

    def forward(self, input, length):
        input = input.float() # shape [batch size, sentence length, embedding size]

        dropout1 = self.dropout1(input)
        dropout2 = self.dropout2(input)

        relu = tnn.ReLU()

        # network 1
        output1, (hidden1, _) = self.rnn1(dropout1)
        hidden1 = self.dropout1(torch.cat((hidden1[-1,:,:], hidden1[-2,:,:]), dim=1))
        hidden1 = self.rnn1fc1(hidden1)

        # network 2
        output2, (hidden2, _) = self.rnn2(dropout2)
        hidden2 = self.dropout2(torch.cat((hidden2[-1,:,:], hidden2[-2,:,:]), dim=1))
        hidden2 = self.rnn2fc1(hidden2)

        return self.rnn1fc2(self.dropout1(relu(hidden1))), self.rnn2fc2(self.dropout2(relu(hidden2)))


net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 128
epochs = 20
optimiser = toptim.Adam(net.parameters(), lr=0.0005)