from math import tanh
from random import random
from typing import List


class Neuron:
    def __init__(self, weights_count: int, learning_rate: float) -> None:
        self.inputs = []
        self.weights = [random() * 2 - 1 for i in range(weights_count)]
        self.learning_rate = learning_rate

    def set_inputs(self, inputs: List[float]) -> None:
        self.inputs = inputs

    def restart(self) -> None:
        self.inputs = []
        self.weights = [random() * 2 - 1 for i in range(len(self.weights))]

    def evaluate(self) -> float:
        return tanh(
            sum([value * weight for value, weight in zip(self.inputs, self.weights)])
        )

    def train(self, error: float) -> None:
        delta = error * (1 - (self.evaluate() ** 2))
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * delta * self.inputs[i]

    def __str__(self) -> str:
        return f"Neuron(w:{self.weights} | i:{self.inputs})"

    def __repr__(self) -> str:
        return self.__str__()
