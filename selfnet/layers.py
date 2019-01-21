#! /bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from typing import Callable, Dict

from selfnet.tensor import Tensor
from selfnet.activation import relu, relu_prime


class Layer(object):

    def __init__(self) -> None:
        self.params = {} # type: Dict[str, Tensor]
        self.grads = {} # type: Dict[str, Tensor]

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError


class LinearLayer(Layer):

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()

        self.params["W"] = np.random.random_sample(size=(input_dim, output_dim))
        self.params["b"] = np.zeros(output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        :param inputs: shape (batch_size, input_dim)
        :return: shape (batch_size, output_dim)
        """

        # remember inputs for backward pass
        self.inputs = inputs

        return np.dot(inputs, self.params["W"]) + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:

        # gradient for weights: outer product of head gradient with inputs
        self.grads["W"] = np.dot(self.inputs.T, grad)

        # gradient for biases: head gradient, sum across batch
        # summing across rows is a short form for:
        # np.dot(vector_of_ones, head_gradient)
        self.grads["b"] = np.sum(grad, axis=0)

        # return gradient on inputs
        return np.dot(grad, self.params["W"].T)

    def __repr__(self):
        return "Layer(type=linear, params=%s, grads=%s)" % (self.params, self.grads)


class ActivationLayer(Layer):

    def __init__(self, f: Callable, f_prime: Callable) -> None:
        super().__init__()

        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.outputs = self.f(inputs)

        return self.outputs

    def backward(self, grad: Tensor) -> Tensor:

        return grad * self.f_prime(self.outputs)

    def __repr__(self):
        return "Layer(type=activation, f=%s, f_prime=%s)" % (self.f, self.f_prime)


class ActivationLayerRelu(ActivationLayer):

    def __init__(self):
        super().__init__(relu, relu_prime)