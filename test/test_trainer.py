#! /bin/python3
# -*- coding: utf-8 -*-

import numpy as np

import unittest

from selfnet.train import Trainer
from selfnet.net import Network
from selfnet.layers import LinearLayer, ActivationLayerRelu


class TestTrainer(unittest.TestCase):

    def test_trainer_call(self):
        num_samples = 100
        hidden_dim = 5
        input_dim = 2
        output_dim = 3

        inputs = np.random.sample((num_samples, input_dim))
        outputs = np.random.sample((num_samples, output_dim))

        net = Network()
        net.add(LinearLayer(input_dim, hidden_dim))
        net.add(ActivationLayerRelu())
        net.add(LinearLayer(hidden_dim, output_dim))

        trainer = Trainer(net, max_epochs=10)
        print(trainer)
        trainer.train(inputs, outputs)


if __name__ == "__main__":
    unittest.main()