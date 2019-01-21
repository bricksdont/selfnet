#! /bin/python3
# -*- coding: utf-8 -*-

import numpy as np

import unittest

from selfnet.loss import SumSquaredError

class TestLoss(unittest.TestCase):

    def test_loss_sse_forward_is_scalar(self):
        batch_size = 10
        output_dim = 3

        predicted = np.random.sample((batch_size, output_dim))
        actual = np.random.sample((batch_size, output_dim))

        loss = SumSquaredError()
        loss_value = loss.forward(predicted, actual)

        self.assertTrue(isinstance(loss_value, float))

    def test_loss_sse_backward_shape(self):
        batch_size = 10
        output_dim = 3

        predicted = np.random.sample((batch_size, output_dim))
        actual = np.random.sample((batch_size, output_dim))

        loss = SumSquaredError()
        grad = loss.backward(predicted, actual)

        self.assertEqual((batch_size, output_dim), grad.shape)


if __name__ == "__main__":
    unittest.main()