from neuron import Neuron
from typing import List, Tuple


class Network:
    def __init__(
        self,
        training_data: List[Tuple[List[float], List[float]]],
        learning_rate: float,
        max_epoch: int = 10000,
        bias: float = 1.0,
    ) -> None:
        self.training_data = training_data
        self.max_epoch = max_epoch
        self.bias = bias

        input_count = len(training_data[0][0])
        output_count = len(training_data[0][1])
        hidden_count = input_count * 2 + 1

        self.hidden_layer = [
            Neuron(input_count + 1, learning_rate) for i in range(hidden_count)
        ]
        self.output_layer = [
            Neuron(hidden_count + 1, learning_rate) for i in range(output_count)
        ]

    def set_inputs(self, inputs: List[float]) -> None:
        new_inputs = [self.bias]
        new_inputs.extend(inputs)

        for neuron in self.hidden_layer:
            neuron.set_inputs(new_inputs)

    def feed_forward(self, values: List[float]) -> None:
        self.set_inputs(values)

        new_inputs = [self.bias]
        new_inputs.extend([neuron.evaluate() for neuron in self.hidden_layer])

        for neuron in self.output_layer:
            neuron.set_inputs(new_inputs)

    def back_propagate(self, expected: List[float]) -> None:
        deltas = []

        for i in range(len(self.output_layer)):
            result = self.output_layer[i].evaluate()
            error = expected[i] - result

            self.output_layer[i].train(error)
            deltas.append(error * (1 - result ** 2))

        for i in range(len(self.hidden_layer)):
            error = sum(
                [
                    neuron.weights[i + 1] * deltas[index]
                    for index, neuron in enumerate(self.output_layer)
                ]
            )

            self.hidden_layer[i].train(error)

    def evaluate(self, index: int) -> float:
        return self.output_layer[index].evaluate()

    def restart(self) -> None:
        for neuron in self.hidden_layer:
            neuron.restart()

        for neuron in self.output_layer:
            neuron.restart()

    def train(self) -> float:
        mean_squared_error = 0.0

        epoch = 0
        while epoch < self.max_epoch:
            aux = 0.0
            for inputs, expected in self.training_data:
                self.feed_forward(inputs)

                results = []

                for i in range(len(expected)):
                    results.append((expected[i] - self.evaluate(i)) ** 2)

                aux += sum(results)

                self.back_propagate(expected)
            mean_squared_error = aux / len(self.training_data)

            if epoch > self.max_epoch / 5 and mean_squared_error > 1:
                self.restart()
                epoch = 0

            epoch += 1

        return mean_squared_error

    def predict(self, input: List[float]) -> List[float]:
        self.feed_forward(input)

        return [self.evaluate(i) for i in range(len(self.output_layer))]

    def save(self, filename: str) -> None:
        with open(filename, "w") as file:
            file.write(
                f"{self.max_epoch}\n"
                f"{self.bias}\n"
                f"{len(self.hidden_layer)}\n"
                f"{len(self.output_layer)}\n"
            )

            for neuron in self.hidden_layer:
                file.write(f"{len(neuron.weights)}\n")
                for weight in neuron.weights:
                    file.write(f"{weight}\n")

            for neuron in self.output_layer:
                file.write(f"{len(neuron.weights)}\n")
                for weight in neuron.weights:
                    file.write(f"{weight}\n")

    def load(self, filename: str) -> None:
        with open(filename, "r") as file:
            self.max_epoch = int(file.readline())
            self.bias = float(file.readline())
            hidden_count = int(file.readline())
            output_count = int(file.readline())

            for i in range(hidden_count):
                weights_count = int(file.readline())
                self.hidden_layer[i].weights = [
                    float(file.readline()) for i in range(weights_count)
                ]

            for i in range(output_count):
                weights_count = int(file.readline())
                self.output_layer[i].weights = [
                    float(file.readline()) for i in range(weights_count)
                ]
