#! /bin/python3
# -*- coding: utf-8 -*-

import numpy as np

import unittest

from selfnet.layers import LinearLayer, ActivationLayerRelu
from selfnet.loss import SumSquaredError


class TestLayers(unittest.TestCase):

    def test_linear_layer_forward_shape(self):
        batch_size = 10
        input_dim = 2
        output_dim = 3

        inputs = np.random.sample((batch_size, input_dim))
        layer = LinearLayer(input_dim, output_dim)

        outputs = layer.forward(inputs)

        self.assertEqual((batch_size, output_dim), outputs.shape)

    def test_linear_layer_backward_shape(self):
        batch_size = 10
        input_dim = 2
        output_dim = 3

        inputs = np.random.sample((batch_size, input_dim))
        layer = LinearLayer(input_dim, output_dim)

        predicted = layer.forward(inputs)
        actual = np.random.sample((batch_size, output_dim))

        loss = SumSquaredError()
        loss_grad = loss.backward(predicted, actual)
        grad = layer.backward(loss_grad)

        self.assertEqual(inputs.shape, grad.shape)

        self.assertEqual((input_dim, output_dim), layer.grads["W"].shape)

        self.assertEqual((output_dim,), layer.grads["b"].shape)


    def test_activation_layer_relu_forward_shape(self):

        batch_size = 10
        input_dim = 2

        X = np.random.sample((batch_size, input_dim))
        a = ActivationLayerRelu()

        outputs = a.forward(X)

        # activation should not change the shape at all
        self.assertEqual(X.shape, outputs.shape)


if __name__ == "__main__":
    unittest.main()