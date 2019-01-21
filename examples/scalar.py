#! /bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from selfnet.train import Trainer
from selfnet.net import Network
from selfnet.layers import LinearLayer, ActivationLayerRelu


# create training data

def output_y(x):
    return x * 3.0


X = np.random.sample((100, 1))
y = output_y(X)

# create network

net = Network()
net.add(LinearLayer(1, 5))
net.add(ActivationLayerRelu())
net.add(LinearLayer(5, 1))

# train
trainer = Trainer(net, max_epochs=100)

print(trainer)

trainer.train(X, y)

print(net.forward(np.array([[1], [2]])))
