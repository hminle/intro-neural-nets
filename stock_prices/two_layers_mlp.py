import numpy as np

class TwoLayersNeuralNet():
    def __init__(self, input_layer_dimension, output_layer_dimension, number_of_nodes, seed_num=7):
        # Initialize the parameters
        np.random.seed(seed_num)
        self.W1 = np.random.randn(input_layer_dimension, number_of_nodes) / np.sqrt(input_layer_dimension)
        self.b1 = np.zeros((1, number_of_nodes))
        self.W2 = np.random.randn(number_of_nodes, output_layer_dimension) / np.sqrt(number_of_nodes)
        self.b2 = np.zeros((1, output_layer_dimension))
        self.loss = []

    # This function learns parameters for the neural network and returns the model.
    # - X: input data
    # - y: output target
    # - epochs: Number of passes through the training data for gradient descent
    def train(self, X, y, epochs, learning_rate=0.000000001, regularization_strength=0.01):
        mean = (np.amax(y) - np.amin(y))/2
        self.min_value = np.amin(y)
        self.max_value = np.amax(y)
        normFunction = np.vectorize(self.normalize)

        # Gradient descent. For each batch...
        for i in range(0, epochs):
            # Forwardpropagation
            z1 = X.dot(self.W1) + self.b1
            a1 = np.tanh(z1)
            a1_norm = normFunction(a1, self.min_value, self.max_value)
            z2 = a1_norm.dot(self.W2) + self.b2
            output = np.around(z2)

            # Backpropagation
            delta3 = output - y
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(self.W2.T) * (1 - np.power(a1, 2)) * mean # Derivative of "tanh"
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms
            dW2 += regularization_strength * self.W2
            dW1 += regularization_strength * self.W1

            # Update gradient descent parameters
            self.W1 += -learning_rate * dW1
            self.b1 += -learning_rate *db1
            self.W2 += -learning_rate *dW2
            self.b2 += -learning_rate *db2
            loss = self.calculate_error(output, y)
            self.loss.append(loss)
            print("RMSError after iteration %i: %f" %(i, loss))

    def mean_squared_error(self, predicted, actual):
        return np.average((actual - predicted) ** 2, axis=0)

    def calculate_error(self, X, y):
        output = X
        num_examples = output.shape[0]
        output = np.nan_to_num(output)
        y = np.nan_to_num(y)
        data_loss = np.sqrt(1.0 / num_examples * self.mean_squared_error(output, y))
        return np.log(data_loss)

    def normalize(self, x, min_value, max_value):
        mean = (max_value - min_value)/2.
        if x >= -1 and x <= 0:
            return -1*((x + 1) * mean - min_value )
        else:
            return (x - 1) * mean + max_value

    # Helper function to predict an output (0 or 1)
    def predict(self, X):
        normFunction = np.vectorize(self.normalize)
        # Forwardpropagation
        z1 = X.dot(self.W1) + self.b1
        a1 = np.tanh(z1)
        a1_norm = normFunction(a1, self.min_value, self.max_value)
        z2 = a1_norm.dot(self.W2) + self.b2
        return np.round(z2)
