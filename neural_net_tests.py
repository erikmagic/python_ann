import unittest
import numpy as np

from neural_net import NeuralNetwork

class NeuralNetTester(unittest.TestCase):

    def test_layer_shape_expected(self):
        input_dim = 32 * 32
        number_labels = 10
        intermediate_layers = 1
        intermediate_neurons = 1000

        neural_net = NeuralNetwork(input_dim=input_dim, number_labels=number_labels,
                                   intermediate_layers=intermediate_layers, intermediate_neurons=intermediate_neurons)

        self.assertEqual(len(neural_net.layers), 3)

        self.assertEqual(neural_net.layers[0].shape, (input_dim + 1, intermediate_neurons))

        self.assertEqual(neural_net.layers[1].shape, (intermediate_neurons + 1, intermediate_neurons))

        self.assertEqual(neural_net.layers[2].shape, (intermediate_neurons + 1, number_labels))


    def test_sigmoid_activation_fct_expected(self):
        test_vector = np.array([1.1, 0.4, 0.9, -0.7])

        expected_vector = np.array([0.75, 0.6, 0.711, 0.33])

        neural_net = NeuralNetwork(1000, 10)

        actual_vector = neural_net.sigmoid_activation_fct(test_vector)

        for expected, actual in zip(expected_vector, actual_vector):
            self.assertAlmostEqual(expected, actual, 2)

    def test_forward_pass(self):
        input_dim = 2
        number_labels = 2
        intermediate_neurons = 3
        intermediate_layers = 1

        neural_net = NeuralNetwork(input_dim=input_dim,
                                   number_labels=number_labels,
                                   intermediate_neurons=intermediate_neurons,
                                   intermediate_layers=intermediate_layers)

        # the layers in the neural net are randomly initialized
        # modify them to fixed numbers in order to test
        first_layer = np.array([[1,1, 1], [2,2, 2], [3,3, 3]])

        intermediate_layer = np.array([[1,1,1], [2,2,2], [3,3,3]])
        intermediate_layer = np.vstack((intermediate_layer, np.array([1,1,1]).T))

        last_layer = np.array([[0.5, 1], [0.5, 1], [0.5, 1], [1, 1]])

        neural_net.layers = [first_layer, intermediate_layer, last_layer]

        expected_result = [92.5, 184]

        actual_result = neural_net.forward_pass(np.array([1, 3]))

        for actual, expected in zip(actual_result, expected_result):
            self.assertEqual(expected, actual)

    def test_forward_pass_with_weights_randomly_initialized(self):
        input_dim = 32 * 32
        number_labels = 10
        intermediate_neurons = 4000
        intermediate_layers = 2

        neural_net = NeuralNetwork(input_dim=input_dim,
                                   number_labels=number_labels,
                                   intermediate_neurons=intermediate_neurons,
                                   intermediate_layers=intermediate_layers)


        input_vector = np.random.uniform(0, 1, input_dim)

        forward_res = neural_net.forward_pass(input_vector)

        self.assertEqual(number_labels, len(forward_res))


    def test_errors_computed_correctly(self):
        # mock the layers just as in the forward pass case
        input_dim = 2
        number_labels = 2
        intermediate_neurons = 3
        intermediate_layers = 1

        neural_net = NeuralNetwork(input_dim=input_dim,
                                   number_labels=number_labels,
                                   intermediate_neurons=intermediate_neurons,
                                   intermediate_layers=intermediate_layers)

        # the layers in the neural net are randomly initialized
        # modify them to fixed numbers in order to test
        first_layer = np.array([[1,1, 1], [2,2, 2], [3,3, 3]])

        intermediate_layer = np.array([[1,1,1], [2,2,2], [3,3,3]])
        intermediate_layer = np.vstack((intermediate_layer, np.array([1,1,1]).T))

        last_layer = np.array([[0.5, 1], [0.5, 1], [0.5, 1], [1, 1]])

        neural_net.layers = [first_layer, intermediate_layer, last_layer]

