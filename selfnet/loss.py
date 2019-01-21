#! /bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from selfnet.tensor import Tensor

class Loss(object):

    def forward(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def backward(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class SumSquaredError(Loss):

    def forward(self, predicted: Tensor, actual: Tensor) -> float:

        return np.sum((predicted - actual) ** 2)

    def backward(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)

    def __repr__(self):
        return "Loss(type=SumSquaredError)"