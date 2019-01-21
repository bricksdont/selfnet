#! /bin/python3
# -*- coding: utf-8 -*-

from typing import List, Iterator, Tuple

from selfnet.tensor import Tensor
from selfnet.layers import Layer


class Network(object):

    def __init__(self) -> None:
        self.layers = [] # type: List[Layer]

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    def __repr__(self) -> str:
        return "Network(%s)" % self.layers

    def forward(self, inputs: Tensor) -> Tensor:

        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs

    def backward(self, grad: Tensor) -> None:

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:

        for layer in self.layers:
            for name, param in layer.params.items():
                yield param, layer.grads[name]
