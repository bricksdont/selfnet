#! /bin/python3
# -*- coding: utf-8 -*-

from selfnet.tensor import Tensor
from selfnet.net import Network
from selfnet.loss import Loss, SumSquaredError
from selfnet.optimizer import Optimizer, SGD
from selfnet.data import DataIterator, BatchIterator


class Trainer(object):

    def __init__(self,
                 net: Network,
                 loss: Loss = SumSquaredError(),
                 optimizer: Optimizer = SGD(),
                 iterator: DataIterator = BatchIterator(),
                 max_epochs: int = 1000) -> None:
        self.net = net
        self.loss = loss
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.iterator = iterator

    def __repr__(self) -> str:

        return "Trainer(net=%s, loss=%s, optimizer=%s, iterator=%s, max_epochs=%s)" % \
               (self.net, self.loss, self.optimizer, self.iterator, self.max_epochs)

    def train(self, inputs: Tensor, targets: Tensor) -> None:

        for epoch in range(self.max_epochs):
            epoch_loss = 0.0

            for batch in self.iterator(inputs, targets):
                outputs = self.net.forward(batch.inputs)
                loss = self.loss.forward(outputs, batch.targets)
                epoch_loss += loss

                grad = self.loss.backward(outputs, batch.targets)
                self.net.backward(grad)
                self.optimizer.step(self.net)

            print("Epoch=%s, loss=%f" % (epoch, epoch_loss))
