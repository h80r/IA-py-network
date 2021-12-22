from typing import List, Tuple
from network import Network


def read_training_data(filename: str) -> List[Tuple[List[str], List[str]]]:
    inputs = []
    training_data = []

    with open(filename, "r") as file:
        for line in file:
            for word in line.split(","):
                inputs.append(line)
                expected = [x for x in line.split(",")][-1]

    print(inputs)

    training_data.append((inputs, expected))

    return training_data


def main() -> None:
    training_data = read_training_data("car.data")

    network = Network(training_data, 0.5, 1000, 1)

    # print(network.predict([0, 0]))
    # print(network.predict([0, 1]))
    # print(network.predict([1, 0]))
    # print(network.predict([1, 1]))


if __name__ == "__main__":
    main()
