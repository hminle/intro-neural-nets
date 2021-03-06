import csv
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def read_dataset():
    coordinates = []
    classes = []
    with open('two_moon.txt') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for idx, row in enumerate(reader):
            if idx in range(0, 4):
                pass
            else:
                coordinates.append([float(row[0]), float(row[1])])
                classes.append(int(row[2]))
    return (np.array(coordinates), np.array(classes))

def calculate_error(model, X, y):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    number_of_examples = X.shape[0]

    # Forward propagation
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    scores = a1.dot(W2) + b2
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Calculate the error
    correct_logprobs = -np.log(probs[range(number_of_examples), y])
    data_loss = np.sum(correct_logprobs)

    # Add regularization term
    data_loss += Parameters.regularization_strength / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1.0 / number_of_examples * data_loss

# This function learns parameters for the neural network and returns the model.
# - number_of_nodes: Number of nodes in the hidden layer
# - epochs: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def train_neural_network(X, y, number_of_nodes, epochs=20000, print_loss=False):
    # Initialize weights to random numbers
    np.random.seed
    W1 = np.random.randn(Parameters.input_layer_dimension, number_of_nodes) / np.sqrt(Parameters.input_layer_dimension)
    b1 = np.zeros((1, number_of_nodes))
    W2 = np.random.randn(number_of_nodes, Parameters.output_layer_dimension) / np.sqrt(number_of_nodes)
    b2 = np.zeros((1, Parameters.output_layer_dimension))

    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2 }
    result = {
        'best_model': model, 'last_validation_loss': 10, 'best_iterations': 0,
        'training_loss': [], 'validation_loss': []
    }

    # Gradient descent. For each batch...
    for i in range(0, epochs):
        # Forwardpropagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        scores = a1.dot(W2) + b2

        # Compute the class probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        delta3[range(Parameters.number_of_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2)) # Derivative of "tanh"
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization term
        dW2 += Parameters.regularization_strength * W2
        dW1 += Parameters.regularization_strength * W1

        # Update gradient descent parameters
        W1 += -Parameters.learning_rate * dW1
        b1 += -Parameters.learning_rate * db1
        W2 += -Parameters.learning_rate * dW2
        b2 += -Parameters.learning_rate * db2

        # Update model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2 }

        loss = calculate_error(model, X, y)
        validation_loss = calculate_error(model, Parameters.coordinates[100:140,:], Parameters.classes[100:140])

        result['training_loss'].append(loss)
        result['validation_loss'].append(validation_loss)

        if validation_loss < result['last_validation_loss']:
            result['best_model'] = model
            result['last_validation_loss'] = validation_loss
            result['best_iterations'] = i

        if print_loss and i % 100 == 0:
            print("Loss after iteration %i: %f. Validation loss: %f" %(i, loss, validation_loss))

    return result

# Helper function to predict an output (0 or 1)
def predict(model, X):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    scores = a1.dot(W2) + b2
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

def count_mismatches(predicted_labels, actual_labels):
    diffs = { 'false_0': 0, 'false_1': 0 }
    for i in range(0, len(predicted_labels)):
        if predicted_labels[i] != actual_labels[i]:
            if predicted_labels[i] == 0:
                diffs['false_0'] += 1
            elif predicted_labels[i] == 1:
                diffs['false_1'] += 1
    return diffs

def visualize(X, y, stats):
    plot_decision_boundary(lambda x:predict(stats['best_model'],x), X, y)
    plot_loss(stats)

def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the coordinate system and the data
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

def plot_loss(stats):
    plt.title('Loss History')
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.plot(stats['validation_loss'][:1000], 'g-', label='Validation Loss')
    plt.plot(stats['training_loss'][:1000], 'b-', label='Training Loss')
    plt.legend()
    plt.show()

class Parameters:
    coordinates, classes = read_dataset()
    number_of_examples = int(len(coordinates) * 0.5)

    input_layer_dimension = 2
    output_layer_dimension = 2

    # Hyperparameters
    learning_rate = 0.02
    regularization_strength = 0.01


# Train the model
print("Training model with 2000 iterations...")
stats = train_neural_network(
    Parameters.coordinates[:100,:],
    Parameters.classes[:100], 3, epochs=2000, print_loss=True)
model = stats['best_model']

# Print result for validation
print("")
print("Best validation loss at %i iterations!" %(stats['best_iterations']))
print("Result of validation...")
validation_results = predict(model, Parameters.coordinates[100:140,:])
validation_count = count_mismatches(validation_results, Parameters.classes[100:140])
print(validation_count)

# Test the model
print("")
print("Testing with model with best validation loss...")
test_results = predict(model, Parameters.coordinates[140:,:])
test_count = count_mismatches(test_results, Parameters.classes[140:])
print(test_count)

# Visualize results
print("")
print("Visualizing...")
visualize(Parameters.coordinates, Parameters.classes, stats)
