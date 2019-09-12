import numpy as np
import math


class NeuralNetwork:

    def __init__(self, input_dim, number_labels, intermediate_layers=1, intermediate_neurons=1000):
        self.input_dim = input_dim
        self.output_dim = number_labels
        self.create_layers(intermediate_layers, intermediate_neurons)

    def create_layers(self, intermediate_layers, intermediate_neurons):
        self.layers = []

        # create the initial layer
        first_layer = np.random.standard_normal(size=(self.input_dim + 1, intermediate_neurons))
        self.layers.append(first_layer)

        # create the hidden layers
        for _ in range(intermediate_layers):
            created_layer = np.random.standard_normal(size=(intermediate_neurons + 1, intermediate_neurons))
            self.layers.append(created_layer)

        # create the final layer
        last_layer = np.random.standard_normal(size=(intermediate_neurons + 1, self.output_dim))
        self.layers.append(last_layer)


    def forward_pass(self, input_vector):

        # add 1 to the input vector to handle the bias term
        input_vector = np.concatenate((input_vector, [1]))

        layer_output = np.dot(input_vector.T, self.layers[0])


        for layer in self.layers[1:]:
            # add the bias term to layer_output
            layer_output = np.append(layer_output, [1])
            layer_output = np.dot(layer_output.T, layer)

        return layer_output


    def compute_error(self, actual_result, expected_result):
        total_error = 0
        for actual, expected in zip(actual_result, expected_result):
            error = 0.5 * (actual - expected) ** 2
            total_error += error



    def backpropagate_pass(self, X, y):
        network_y = self.forward_pass(X)
        error = self.compute_error(network_y, y)



    def sigmoid_activation_fct(self, z_vector):
        return 1 / (1 + math.e**(-1 * z_vector))
