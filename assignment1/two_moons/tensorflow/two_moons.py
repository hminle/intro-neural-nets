from utils import read_dataset
import tensorflow as tf

# Multilayer Perceptron Model
def multilayer_perceptron(x, model):
    # Hidden layer with tanh activation
    layer_1 = tf.add(tf.matmul(x, model['W1']), model['b1'])
    layer_1 = tf.nn.tanh(layer_1)
    # Output layer with softmax activation
    out_layer = tf.add(tf.matmul(layer_1, model['W2']), model['b2'])
    return out_layer


# Prepare data
coordinates, labels = read_dataset()
training_data = coordinates[:100,:]
training_labels = labels[:100]
# training_labels = tf.one_hot(labels[:100], 2)

test_data = coordinates[100:,:]
test_labels = labels[100:]

# Neural Network Parameters
n_hidden = 32       # number of neurons in hidden layer
n_input = 2         # two moons input (x + y coordinates)
n_classes = 2       # two moons classes

# Store model
model = {
    'W1': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'b1': tf.Variable(tf.zeros([n_hidden])),
    'W2': tf.Variable(tf.random_normal([n_hidden, n_classes])),
    'b2': tf.Variable(tf.zeros([n_classes]))
}

# This is where training samples and labels are fed to the graph.
# These placeholder nodes will be fed a batch of training data at each
# training step using the {feed_dict} argument to the Run() call below.
x = tf.placeholder(tf.float32, shape=(100, 2))
y = tf.placeholder(tf.int32, shape=(100, 2))

# Construct model
pred = multilayer_perceptron(x, model)

# Define loss and optimization function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

# Launch the session
with tf.Session() as session:
    # Initializing the variables
    session.run(tf.global_variables_initializer())
    iterations = 1000

    training_labels = session.run(tf.one_hot(training_labels, 2))
    test_labels = session.run(tf.one_hot(test_labels, 2))

    # Training cycle
    for epoch in range(iterations):
        _, loss_value = session.run([optimizer, loss], feed_dict={ x: training_data, y: training_labels })

        # Display logs every once in a while
        if epoch % 100 == 0:
            print("Iteration: %d, Loss: %f" % (epoch, loss_value))

    # Calculate accuracy
    # predicted_class = tf.argmax(pred, axis=1)
    correct_prediction = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", session.run(accuracy, feed_dict={ x: test_data, y: test_labels }))
