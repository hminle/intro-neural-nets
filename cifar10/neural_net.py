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

    def calculate_probabilities(self, X, y):
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']

        # Forwardpropagation
        z1 = X.dot(W1) + b1
        a1 = np.maximum(0, z1) # Use ReLU as activation function
        scores = a1.dot(W2) + b2

        # Compute and return the class probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs, a1

    def calculate_error(self, X, y, probs, regularization_strength):
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']

        # Calculate the error
        correct_logprobs = -np.log(probs[range(X.shape[0]), y])
        data_loss = np.sum(correct_logprobs)

        # Add regularization term
        data_loss += regularization_strength / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        # if math.isnan(1.0 / X.shape[0] * data_loss):
        #     print(np.square(W1))
        #     print(np.square(W2))
        #     print(np.sum(correct_logprobs))
        #     print(data_loss)
        #     sys.exit()
        return 1.0 / X.shape[0] * data_loss

    # Train this neural network using stochastic gradient descent
    def train(self, X, y, X_val, y_val,
              learning_rate=0.001, learning_rate_decay=0.95,
              regularization_strength=0.00001, iterations=100,
              batch_size=200, print_loss=False):
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for i in range(iterations):
            # Get a random batch
            sample_indices = np.random.choice(np.arange(num_train), batch_size)
            X_batch = X[sample_indices]
            y_batch = y[sample_indices]

            W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']

            # Forwardpropagation
            probs, a1 = self.calculate_probabilities(X, y)
            print(a1)

            # Compute and save the loss
            error = self.calculate_error(X, y, probs, regularization_strength)
            loss_history.append(error)

            # Backpropagation
            delta3 = probs
            delta3[range(X.shape[0]), y] -= 1

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
            if print_loss and i % 10 == 0:
                print("Loss after iteration %i / %i: %f" %(i, iterations, error))

            # Every epoch, check training and validation accuracy and decay learning rate
            if i % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
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
