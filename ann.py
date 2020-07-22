import numpy as np
import matplotlib.pyplot as plt

X = np.array(([0, 0, 0, 0, 0], [0, 0, 0, 0, 1]
              , [0, 0, 0, 1, 0], [0, 0, 0, 1, 1]
              , [0, 0, 1, 0, 0], [0, 0, 1, 0, 1]
              , [0, 0, 1, 1, 0], [0, 0, 1, 1, 1]
              , [0, 1, 0, 0, 0], [0, 1, 0, 0, 1]
              , [0, 1, 0, 1, 0], [0, 1, 0, 1, 1]
              , [0, 1, 1, 0, 0], [0, 1, 1, 0, 1]
              , [0, 1, 1, 1, 0], [0, 1, 1, 1, 1]
              , [1, 0, 0, 0, 0], [1, 0, 0, 0, 1]
              , [1, 0, 0, 1, 0], [1, 0, 0, 1, 1]
              , [1, 0, 1, 0, 0], [1, 0, 1, 0, 1]
              , [1, 0, 1, 1, 0], [1, 0, 1, 1, 1]
              , [1, 1, 0, 0, 0], [1, 1, 0, 0, 1]
              , [1, 1, 0, 1, 0], [1, 1, 0, 1, 1]
              , [1, 1, 1, 0, 0], [1, 1, 1, 0, 1]
              , [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]))

Y = np.array(([0, 0, 1], [0, 0, 0]
              , [0, 1, 1], [0, 1, 0]
              , [0, 1, 1], [0, 1, 0]
              , [0, 1, 1], [0, 1, 0]
              , [1, 0, 1], [1, 0, 0]
              , [1, 1, 1], [1, 1, 0]
              , [1, 1, 1], [1, 1, 0]
              , [1, 1, 1], [1, 1, 0]
              , [1, 0, 1], [1, 0, 0]
              , [1, 1, 1], [1, 1, 0]
              , [1, 1, 1], [1, 1, 0]
              , [1, 1, 1], [1, 1, 0]
              , [1, 0, 1], [1, 0, 0]
              , [1, 1, 1], [1, 1, 0]
              , [1, 1, 1], [1, 1, 0]
              , [1, 1, 1], [1, 1, 0]))

arr_rand = np.random.rand(X.shape[0])
split = arr_rand < np.percentile(arr_rand, 80)
X_TRAIN = X[split]
Y_TRAIN = Y[split]
X_TEST = X[~split]
Y_TEST = Y[~split]


class artificial_neural_network(object):

    def __init__(self):
        self.input_size = 5
        self.hidden_size = 4
        self.output_size = 3

        self.weight1 = np.random.uniform(low=-1, high=1, size=(self.input_size, self.hidden_size))
        self.weight2 = np.random.uniform(low=-1, high=1, size=(self.hidden_size, self.output_size))

    def sigmoid(self, S, deriv=False):
        if deriv:
            return S * (1 - S)
        return 1 / (1 + np.exp(-S))

    def feedforward(self, X, Y):
        self.netH = np.dot(X, self.weight1)
        outH = self.sigmoid(self.netH)
        self.netO = np.dot(outH, self.weight2)
        outO = self.sigmoid(self.netO)

        self.errorList = np.zeros((3, 1))
        i = 0
        for num in outO:
            self.update = np.subtract(Y[i], num)
            self.errorList[i] = self.update
            i += 1

        return outH, outO, self.errorList

    def summation_delta_weight(self, out_deltas, weight):
        product = 0
        for x, out_delta in enumerate(out_deltas):
            product += out_delta * weight[x]
        return product

    def backword_propogation(self, X, Y, outH, outO, learning_rate):
        self.out_delta = self.sigmoid(outO, deriv=True) * (Y - outO)

        for i in range(self.hidden_size):
            for j in range(self.output_size):
                self.weight2[i][j] += learning_rate * self.out_delta[j] * outH[i]

        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.hidden_delta = self.sigmoid(outH, deriv=True) \
                                    * self.summation_delta_weight(self.out_delta, self.weight2[j])

                self.weight1[i][j] += learning_rate * self.hidden_delta[j] * X[i]

    def is_bad_fact(self, error_list):
        mu = 0.2
        for error in error_list:
            if abs(error) > mu:
                return False
        return True

    def plot_graph_bad_facts_vs_epochs(self, graph):

        plt.plot(graph)
        plt.xlabel("Epochs")
        plt.ylabel("Bad Facts")
        print(plt.show())

    def train(self, X, Y, epochs):
        learning_rate = 0.2
        epoch_number = 0
        ending = False
        epochs_vs_bad_facts_graph = np.zeros(epochs)
        # np.zeros(epochs_vs_bad_facts_graph)
        while epoch_number < epochs:
            bad_fact_number = 0
            if not ending:
                for j in range(len(X)):
                    output = self.feedforward(X[j], Y[j])

                    bad_fact = self.is_bad_fact(output[2])

                    if not bad_fact:
                        bad_fact_number += 1
                        self.backword_propogation(X[j], Y[j], output[0], output[1], learning_rate)

                    print("Bad Facts in epoch:" + str(epoch_number) + " | "
                          + str(bad_fact_number))

                    epochs_vs_bad_facts_graph[epoch_number] = bad_fact_number

                epoch_number += 1

        self.plot_graph_bad_facts_vs_epochs(epochs_vs_bad_facts_graph)


ANN = artificial_neural_network()
ANN.train(X_TRAIN, Y_TRAIN, 999)
for x in range(len(X_TEST)):
    output = ANN.feedforward(X_TEST[x], Y_TEST[x])
    print(" Input " + str(X_TEST[x]) + " Expected Output " +
          str(Y_TEST[x]) + " Actual Output" + str(np.around(output[1])))
