#! /bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from selfnet.tensor import Tensor


def linear(inputs: Tensor) -> Tensor:
    return inputs

def linear_prime(inputs: Tensor) -> Tensor:

    return np.ones_like(inputs)

def relu(inputs: Tensor) -> Tensor:
    return np.maximum(0, inputs)

def relu_prime(outputs: Tensor) -> Tensor:

    grad = np.ones_like(outputs)
    grad[outputs <= 0] = 0

    return grad

def tanh(inputs: Tensor) -> Tensor:
    return np.tanh(inputs)
