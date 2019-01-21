#! /bin/python3
# -*- coding: utf-8 -*-

from selfnet.net import Network


class Optimizer(object):

    def __init__(self,
                 learning_rate: float = 0.001) -> None:

        self.learning_rate = learning_rate

    def step(self, net: Network) -> None:
        raise NotImplementedError


class SGD(Optimizer):

    def step(self, net: Network) -> None:

        for param, grad in net.params_and_grads():
            param -= self.learning_rate * grad

    def __repr__(self):

        return "Optimizer(type=SGD, learning_rate=%s)" % (self.learning_rate)
