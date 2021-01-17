# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
"""
This is the main entry point for MP6. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class TrainDataSet():
    def __init__(self, trainset, trainlabels):
        self.train_set = trainset
        self.labels = trainlabels
    def __len__(self):
        return len(self.train_set)
    def __getitem__(self, idx):
        return (self.train_set[idx], self.labels[idx])


class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network

        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param yhat - an (N,out_size) tensor
            @param y - an (N,) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param in_size: Dimension of input
        @param out_size: Dimension of output

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.hidden1 = nn.Linear(in_size, 120)
        self.hidden2 = nn.Linear(120, 32)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(32, out_size)
        self.optimizer = torch.optim.SGD(self.parameters(), lr = 0.001, momentum = 0.8)


    def set_parameters(self, params):
        """ Set the parameters of your network
        @param params: a list of tensors containing all parameters of the network
        """

        pass

    def get_parameters(self):
        """ Get the parameters of your network
        @return params: a list of tensors containing all parameters of the network
        """
        return self.parameters()


    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """
        x = F.relu(self.hidden1(x))
        hidden = self.hidden2(x)
        sig = self.sigmoid(hidden)
        output = self.out(sig)
        return F.softmax(output)

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """

        outputs = self.forward(x)
        loss = self.loss_fn(outputs, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()


def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of epochs of training
    @param batch_size: The size of each batch to train on. (default 100)

    # return all of these:

    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: A NeuralNet object

    # NOTE: This must work for arbitrary M and N
    """
    
    loss_fn = nn.CrossEntropyLoss()
    net = NeuralNet(0.001, loss_fn, 3072, 2)

    running_loss = []
    train_dat = TrainDataSet(train_set, train_labels)
    trainloader = torch.utils.data.DataLoader(train_dat, batch_size= batch_size, shuffle = True)


    for i in (range(n_iter)):
        curr_loss = 0
        for i, data in enumerate(trainloader,0) :
            train, labels = data
            net.optimizer.zero_grad()
            curr_loss += net.step(train, labels)
        running_loss.append(curr_loss/float(batch_size))
    
    y_hats = net.forward(dev_set)
    _, predicted = torch.max(y_hats, 1)
    return(running_loss, predicted.numpy(), net)


