import numpy as np
import math
import random
from mnist_loader import load_data_wrapper

class Network():
    # Layers is an array of neurons in each layer
    # for example, [784,30,30,10] creates a network with
    # 784 input values, two hidden layers with 30 neurons each and 
    # an output layer with ten values
    def __init__(self, layers):
        self.layer_count = len(layers)

        # Xavier initialization
        glorot_product = math.sqrt(2 / (layers[0] * layers[-1]))
        self.weights = [np.random.randn(layers[i+1], layer) * glorot_product for i, layer in enumerate(layers[:-1])]
        self.biases = [np.random.randn(layer, 1) * glorot_product for i, layer in enumerate(layers[1:])]

    def sgd(self, mini_batch, learning_rate):
        # one column per training example. n rows equals batch_size
        training_data = np.squeeze(np.array([np.array(training_values) for training_values, _ in mini_batch])).T
        expected_values = np.squeeze(np.array([np.array(labels) for _, labels in mini_batch])).T
        (activations, outputs) = self.feedforward(input=training_data)

        (weight_gradients, bias_gradients) = self.backpropagate(activations=activations, outputs=outputs, expected_values=expected_values)

        # average the bias gradients over the training examples in the batch
        bias_gradients = [np.expand_dims(np.mean(bg, axis=1), axis=1) for bg in bias_gradients]

        self.weights = [w - (learning_rate / len(mini_batch)) * g for w, g in zip(self.weights, weight_gradients)]
        self.biases = [b - (learning_rate / len(mini_batch)) * g for b, g in zip(self.biases, bias_gradients)]

    # propagate back through the network and 
    # return a tuple of lists of gradients for weights and biases
    def backpropagate(self, activations, outputs, expected_values):
        weight_gradients = [np.zeros(weights_in_layer.shape) for weights_in_layer in self.weights]
        bias_gradients = [np.zeros(biases_in_layer.shape) for biases_in_layer in self.biases]

        loss = (activations[-1] - expected_values) * relu_prime(outputs[-1])
        weight_gradients[-1] = np.dot(loss, activations[-2].T)
        bias_gradients[-1] = loss

        for l in reversed(range(self.layer_count - 1)):
            loss = loss if l == self.layer_count - 2 else np.dot(self.weights[l + 1].T, loss) * relu_prime(outputs[l])
            weight_gradients[l] = np.dot(loss, activations[l].T)
            bias_gradients[l] = loss

        return (weight_gradients, bias_gradients)


    # Go through the network and 
    # return a tuple of lists for activations and outputs per layer
    def feedforward(self, input):
        outputs = []
        activations = [input]
        for i, (biases, weights) in enumerate(zip(self.biases, self.weights)):
            output = np.dot(weights, activations[i]) + biases
            outputs.append(output)
            activations.append(relu(output))

        return (activations, outputs)

    def train(self, number_of_epochs, learning_rate, batch_size):
        training_data, _, test_data = load_data_wrapper()
        training_data = list(training_data)
        test_data = list(test_data)
        random.shuffle(training_data)
        
        for e in range(number_of_epochs):
            batches = [training_data[i:i + batch_size] for i in range(0, len(training_data), batch_size)]
            for batch in batches: self.sgd(batch, learning_rate)
            print(f"Epoch {e} finished. Accuracy: {self.evaluate(test_data=test_data)}")

    def evaluate(self, test_data):
        predictions = [(np.argmax(self.feedforward(values)[0][-1]), expected) for values, expected in test_data]
        return (sum(int(predicted == expected) for predicted, expected in predictions)) / len(predictions)

def relu(x):
    return np.maximum(x, 0)

def relu_prime(x):
    return np.heaviside(x, 1)

def test():
    net = Network(layers=[784, 30, 30, 10])
    net.train(number_of_epochs=30, learning_rate=.03, batch_size=10)

if __name__ == "__main__":
    test()





