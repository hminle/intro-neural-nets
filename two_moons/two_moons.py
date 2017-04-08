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

# Helper function to evaluate the total loss on the dataset
def calculate_loss(model, X, y):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)

    # print(len(exp_scores))

    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Calculating the loss
    correct_logprobs = -np.log(probs[range(Parameters.number_of_examples), y])
    data_loss = np.sum(correct_logprobs)

    # Add regulation term to loss (optional)
    data_loss += Parameters.regulation_strength / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1.0 / Parameters.number_of_examples * data_loss

    # predicted_labels = np.argmax(probs, axis=1)
    #
    # counters = Counter(Parameters.classes)
    # accuracy = {}
    # for i in range(0, len(predicted_labels)):
    #     if predicted_labels[i] == Parameters.classes[i]:
    #         if Parameters.classes[i] in accuracy:
    #             accuracy[Parameters.classes[i]] = accuracy[Parameters.classes[i]] + 1
    #         else:
    #             accuracy[Parameters.classes[i]] = 1
    #     else:
    #         if Parameters.classes[i] in accuracy:
    #             pass
    #         else:
    #             accuracy[Parameters.classes[i]] = 0
    #
    # # Calculating the loss
    # for i in range(0, len(accuracy)):
    #     if i in accuracy:
    #         accuracy[i] = accuracy[i] / counters[i]
    #
    # print(accuracy)
    # return accuracy

# This function learns parameters for the neural network and returns the model.
# - number_of_nodes: Number of nodes in the hidden layer
# - epochs: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def train_neural_network(X, y, number_of_nodes, epochs=20000, print_loss=False):
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(Parameters.input_layer_dimension, number_of_nodes) / np.sqrt(Parameters.input_layer_dimension)
    b1 = np.zeros((1, number_of_nodes))
    W2 = np.random.randn(number_of_nodes, Parameters.output_layer_dimension) / np.sqrt(number_of_nodes)
    b2 = np.zeros((1, Parameters.output_layer_dimension))

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for i in range(0, epochs):
        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        delta3[range(Parameters.number_of_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += Parameters.regulation_strength * W2
        dW1 += Parameters.regulation_strength * W1

        # Gradient descent parameter update
        W1 += -Parameters.learning_rate * dW1
        b1 += -Parameters.learning_rate * db1
        W2 += -Parameters.learning_rate * dW2
        b2 += -Parameters.learning_rate * db2

        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
          print("Loss after iteration %i: %f" %(i, calculate_loss(model, X, y)))
    # calculate_loss(model, X, y)
    return model

# Helper function to predict an output (0 or 1)
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

def count_mismatches(predicted_labels, actual_labels):
    diffs = { 'false_0': 0, 'false_1': 0 }
    for i in range(0, len(predicted_labels)):
        if predicted_labels[i] == actual_labels[i]:
            pass
        else:
            if predicted_labels[i] == 0:
                diffs['false_0'] = diffs['false_0'] + 1
            else:
                diffs['false_1'] = diffs['false_1'] + 1
    return diffs

def visualize(X, y, model):
    plot_decision_boundary(lambda x:predict(model,x), X, y)
    plt.title("Logistic Regression")

def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

def main():
    print("Training model...")
    model = train_neural_network(
        Parameters.coordinates[:100,:],
        Parameters.classes[:100], 3, print_loss=True)

    print("")
    print("Validating...")
    validation_results = predict(model, Parameters.coordinates[100:40,:])
    validation_count = count_mismatches(validation_results, Parameters.classes[100:40])
    print(validation_count)

    print("")
    print("Testing...")
    test_results = predict(model, Parameters.coordinates[140:,:])
    test_count = count_mismatches(test_results, Parameters.classes[140:])
    print(test_count)

    print("")
    print("Visualizing...")
    visualize(Parameters.coordinates, Parameters.classes, model)

class Parameters:
    coordinates, classes = read_dataset()
    number_of_examples = int(len(coordinates) * 0.5)
    # number_of_examples = len(coordinates)

    input_layer_dimension = 2
    output_layer_dimension = 2

    # Gradient descent parameters
    learning_rate = 0.01
    regulation_strength = 0.01

if __name__ == "__main__":
    main()
