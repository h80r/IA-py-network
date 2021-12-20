from network import Network


def main() -> None:
    training_data = [
        ([0, 0], [-1]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [-1]),
    ]

    network = Network(training_data, 0.5, 1000, 1)
    print(f"Mean Squared Error: {network.train()}")

    print(network.predict([0, 0]))
    print(network.predict([0, 1]))
    print(network.predict([1, 0]))
    print(network.predict([1, 1]))


if __name__ == "__main__":
    main()
