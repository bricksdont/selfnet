#! /bin/python3
# -*- coding: utf-8 -*-

import numpy as np

import unittest

from selfnet.data import BatchIterator


class TestData(unittest.TestCase):

    def test_batch_shape(self):
        num_samples = 100
        batch_size = 10
        input_dim = 2
        output_dim = 3

        inputs = np.random.sample((num_samples, input_dim))
        outputs = np.random.sample((num_samples, output_dim))

        iterator = BatchIterator(batch_size=batch_size)

        batch = next(iterator(inputs, outputs))

        self.assertEqual((batch_size, input_dim), batch.inputs.shape)
        self.assertEqual((batch_size, output_dim), batch.targets.shape)

    def test_num_batches(self):
        num_samples = 100
        batch_size = 10
        input_dim = 2
        output_dim = 3

        inputs = np.random.sample((num_samples, input_dim))
        outputs = np.random.sample((num_samples, output_dim))

        iterator = BatchIterator(batch_size=batch_size, shuffle=False)

        batches = [batch for batch in iterator(inputs, outputs)]

        self.assertEqual(num_samples / batch_size, len(batches))

if __name__ == "__main__":
    unittest.main()