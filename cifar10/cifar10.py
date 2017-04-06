import numpy as np
from collections import Counter

def unpickle(file):
    import _pickle as cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo, encoding='latin1')
    fo.close()
    return dict

# Helper function to evaluate the total loss on the dataset
def calculate_loss(model, X, y):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    predicted_labels = np.argmax(probs, axis=1)

    counters = Counter(Parameters.data['labels'])
    accuracy = {}
    for i in range(0, len(predicted_labels)):
        if predicted_labels[i] == Parameters.data['labels'][i]:
            if Parameters.data['labels'][i] in accuracy:
                accuracy[Parameters.data['labels'][i]] = accuracy[Parameters.data['labels'][i]] + 1
            else:
                accuracy[Parameters.data['labels'][i]] = 1
        else:
            if Parameters.data['labels'][i] in accuracy:
                pass
            else:
                accuracy[Parameters.data['labels'][i]] = 0

    # Calculating the loss
    for i in range(0, len(accuracy)):
        if i in accuracy:
            accuracy[i] = accuracy[i] / counters[i]

    return accuracy

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
        if print_loss and i % 10 == 0:
          print("Accuracy after iteration %i: %a" %(i, calculate_loss(model, X, y)))

    if print_loss:
        print("Accuracy after all iterations: %a" %(calculate_loss(model, X, y)))

    return model

def main():
    # print(Parameters.label_names)
    # print(Parameters.data['labels'])
    # print(Parameters.data['data'][0])
    # print(Parameters.data)
    model = train_neural_network(Parameters.data['data'], Parameters.data['labels'], 3, epochs=100, print_loss=True)

class Parameters:
    label_names = unpickle('./images/batches.meta')
    data = unpickle('./images/data_batch_1')
    number_of_examples = len(data['data'])

    input_layer_dimension = 3072
    output_layer_dimension = 10

    # Gradient descent parameters
    learning_rate = 0.01
    regulation_strength = 0.01

if __name__ == "__main__":
    main()
    # pass
