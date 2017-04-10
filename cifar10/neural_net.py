import numpy as np
import matplotlib.pyplot as plt

import sys
import math

class TwoLayerNeuralNet(object):
    def __init__(self, number_of_inputs, number_of_hidden, number_of_outputs):
        self.model = {}
        self.model['W1'] = np.random.randn(number_of_inputs, number_of_hidden) / np.sqrt(number_of_inputs)
        self.model['b1'] = np.zeros((1, number_of_hidden))
        self.model['W2'] = np.random.randn(number_of_hidden, number_of_outputs) / np.sqrt(number_of_hidden)
        self.model['b2'] = np.zeros((1, number_of_outputs))

    def calculate_error(self, X, y, probs, regularization_strength):
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        number_of_examples = X.shape[0]

        # Calculate the error
        correct_logprobs = -np.log(probs[range(number_of_examples), y])
        data_loss = np.sum(correct_logprobs)

        # Add regularization term
        data_loss += regularization_strength / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1.0 / number_of_examples * data_loss

    # Train this neural network using stochastic gradient descent
    def train(self, X, y, X_val, y_val,
              learning_rate, learning_rate_decay,
              regularization_strength, iterations,
              batch_size, print_loss=False):
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use Stochastic Gradient Descent to optimize self.model
        loss_history = []
        training_history = []
        validation_history = []

        for i in range(iterations):
            # Get a random batch
            sample_indices = np.random.choice(np.arange(num_train), batch_size)
            X_batch = X[sample_indices]
            y_batch = y[sample_indices]

            W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
            number_of_examples = X.shape[0]

            # Forwardpropagation
            z1 = X.dot(W1) + b1
            a1 = np.maximum(0, z1) # Use ReLU as activation function
            scores = a1.dot(W2) + b2

            # Compute and return the class probabilities
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Compute and save the loss
            error = self.calculate_error(X, y, probs, regularization_strength)
            loss_history.append(error)

            # Backpropagation
            delta3 = probs
            delta3[range(number_of_examples), y] -= 1

            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)

            delta2 = delta3.dot(W2.T)   # Derivate of ReLU is just a constant
            delta2[a1 <= 0] = 0         # Backprop the ReLU non-linearity

            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization term
            dW2 += regularization_strength * W2
            dW1 += regularization_strength * W1

            # Update model
            self.model['W1'] += -learning_rate * dW1
            self.model['b1'] += -learning_rate * db1
            self.model['W2'] += -learning_rate * dW2
            self.model['b2'] += -learning_rate * db2

            # Print current loss
            if print_loss and i % 100 == 0:
                print("Loss after iteration %i / %i: %f" %(i, iterations, error))

            # Every epoch save training and validation accuracy and decay the learning rate
            if i % iterations_per_epoch == 0:
                # Save accuracy
                training_accuracy = (self.predict(X_batch) == y_batch).mean()
                validation_accuracy = (self.predict(X_val) == y_val).mean()
                training_history.append(training_accuracy)
                validation_history.append(validation_accuracy)
                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'training_history': training_history,
            'validation_history': validation_history,
        }

    # Predict an output based on current model
    def predict(self, X):
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.maximum(0, z1) # Again use ReLU as activation function
        scores = a1.dot(W2) + b2
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        prediction = np.argmax(probs, axis=1)
        return prediction
