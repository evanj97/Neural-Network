"""

Evan Johnson
Single Hidden Layer Neural Network
Made from scratch with only NumPy (and pandas for convenient .csv reading)


"""

import pandas as pd
import numpy as np
import math
import time


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


#  use numpy sorcery to transform sigmoid() into a function capable of accepting arrays
sigmoid = np.vectorize(sigmoid)


# Single [hidden] Layer Neural Network
class SLNN:

    def __init__(self, num_input_nodes, num_hidden_nodes, num_output_nodes):

        rng = np.random.default_rng()

        # create vector of input "nodes"
        self.input_nodes = np.zeros(num_input_nodes + 1, dtype=float)
        self.input_nodes[0] = 1  # set bias value

        # create vector of hidden nodes
        self.hidden_nodes = np.zeros(num_hidden_nodes + 1, dtype=float)
        self.hidden_nodes[0] = 1  # set bias value

        # create vector of output nodes
        self.output_nodes = np.zeros(num_output_nodes, dtype=float)

        # create vector of input->hidden weights
        self.input_hidden_weights = rng.random(len(self.input_nodes) * (len(self.hidden_nodes) - 1)) - 0.5  # -1 is for the bias node

        # create vector of hidden->output weights
        self.hidden_output_weights = rng.random(len(self.hidden_nodes) * len(self.output_nodes)) - 0.5  # no -1 because output layer lacks a bias node

    # Predict the data's label
    def predict(self, data):
        self.forward_propagate(data=data)

        return np.argmax(self.output_nodes)

    # Forward propagate through the network
    def forward_propagate(self, data):  # todo WORKING FLAWLESSLY!

        #  populate input layer
        self.input_nodes[1:] = np.array(data)

        # calculate hidden layer node values
        self.hidden_nodes[1:] = sigmoid(np.sum(self.input_hidden_weights.reshape(len(self.hidden_nodes) - 1, -1) * self.input_nodes, axis=1))

        # calculate output layer node values
        self.output_nodes = sigmoid(np.sum(self.hidden_output_weights.reshape(len(self.output_nodes), -1) * self.hidden_nodes, axis=1))

    # Back propagate through the network
    def back_propagate(self, label, learning_rate):
        target_values = np.zeros(len(self.output_nodes))
        target_values[label] = 1

        #  determine errors of output nodes
        output_errors = self.output_nodes * (1 - self.output_nodes) * (target_values - self.output_nodes)

        #  determine errors of hidden nodes
        hidden_errors = self.hidden_nodes * (1 - self.hidden_nodes) * np.sum(
            output_errors * self.hidden_output_weights.reshape([len(self.hidden_nodes), -1], order='F'), axis=1)

        #  update hidden -> output weights
        self.hidden_output_weights = self.hidden_output_weights + (
                learning_rate * (np.repeat(output_errors, len(self.hidden_nodes)).reshape(len(output_errors), -1) * self.hidden_nodes).reshape(-1))

        #  update input -> hidden weights
        self.input_hidden_weights = self.input_hidden_weights + (
                learning_rate * (np.repeat(hidden_errors[1:], len(self.input_nodes)).reshape(len(hidden_errors[1:]), -1) * self.input_nodes).reshape(-1))

    # Train the model on a set of training data
    def train(self, training_data, training_labels, learning_rate, epochs=1):
        # self.dt = np.array(training_data)

        for x in range(0, epochs):
            for i in range(0, len(training_data)):
                self.forward_propagate(data=training_data[i])
                self.back_propagate(label=training_labels[i][0], learning_rate=learning_rate)

    # Test the model's accuracy on a set of training data
    def test(self, testing_data, testing_labels):
        correctly_classified = 0
        incorrectly_classified = 0

        for i in range(0, len(testing_data)):
            self.forward_propagate(testing_data[i])

            if np.argmax(self.output_nodes) == testing_labels[i][0]:
                correctly_classified = correctly_classified + 1
            else:
                incorrectly_classified = incorrectly_classified + 1

        # print("DEBUG correctly_classified:", correctly_classified)
        # print("DEBUG incorrectly_classified:", incorrectly_classified)
        # print("DEBUG total classified:", correctly_classified + incorrectly_classified)

        return correctly_classified / (incorrectly_classified + correctly_classified)

    # Save the weights to a text file
    def save_weights(self):
        filename = str(len(self.input_nodes) - 1) + "-" + str(len(self.hidden_nodes) - 1) + "-" + str(len(self.output_nodes)) + "_nodes.csv"
        np.savetxt("input-hidden_" + filename, self.input_hidden_weights, delimiter=",")
        np.savetxt("hidden-output_" + filename, self.hidden_output_weights, delimiter=",")

    # load weights from a text file
    def load_weights(self):
        filename = str(len(self.input_nodes) - 1) + "-" + str(len(self.hidden_nodes) - 1) + "-" + str(len(self.output_nodes)) + "_nodes.csv"
        self.input_hidden_weights = np.loadtxt("input-hidden_" + filename, delimiter=",")
        self.hidden_output_weights = np.loadtxt("hidden-output_" + filename, delimiter=",")

    # for debugging
    def debug_nodes(self):
        print("\n\n----------------------------NODE-DEBUGGER----------------------------")
        print("DEBUG INPUT NODES:\n", self.input_nodes)
        print("\nDEBUG INPUT->HIDDEN WEIGHTS:\n", self.input_hidden_weights)
        print("\nDEBUG HIDDEN NODES:\n", self.hidden_nodes)
        print("\nDEBUG HIDDEN->OUTPUT WEIGHTS:\n", self.hidden_output_weights)
        print("\nDEBUG OUTPUT NODES:\n", self.output_nodes)
        print("----------------------------NODE-DEBUGGER----------------------------\n\n")


#  PARAMETERS
hidden_nodes = 256
learning_rate = 0.05
TRAIN = True

# Load data
print("Loading data...")

test_labels = np.array(pd.read_csv("testing10000_labels.csv", header=None))
test_data = np.array(pd.read_csv("testing10000.csv", header=None))

train_labels = np.array(pd.read_csv("training60000_labels.csv", header=None))
train_data = np.array(pd.read_csv("training60000.csv", header=None))

print("Data loaded.\n")

nn = SLNN(num_input_nodes=784, num_hidden_nodes=hidden_nodes, num_output_nodes=10)

#  load saved weights and test accuracy
if not TRAIN:
    nn.load_weights()
    accuracy = nn.test(testing_data=test_data, testing_labels=test_labels)
    print("Accuracy:", accuracy)

#  train new weights and test accuracy
if TRAIN:
    print("\nNumber of hidden nodes:", hidden_nodes)
    print("TRAINING START:\n")

    total_time = time.time()

    for i in range(5):
        t = time.time()
        nn.train(training_data=train_data, training_labels=train_labels, learning_rate=learning_rate)
        accuracy = nn.test(testing_data=test_data, testing_labels=test_labels)
        nn.save_weights()
        print("Epoch:", i + 1)
        print("Accuracy:", accuracy)
        print("Training rate: %.2f" % learning_rate)
        print("time: %.2f seconds\n" % (time.time() - t))

    print("Total training time: %.2f seconds" % (time.time() - total_time))
