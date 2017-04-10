from utils import read_and_prepare_images
from neural_net import TwoLayerNeuralNet

# Read input images
print("Reading input data...")
training_data, training_labels, validation_data, validation_labels, testing_data, testing_labels = read_and_prepare_images()
print("Training data shape: ", training_data.shape)
print("Training labels shape: ", training_labels.shape)
print("Validation data shape: ", validation_data.shape)
print("Validation labels shape: ", validation_labels.shape)
print("Testing data shape: ", testing_data.shape)
print("Testing labels shape: ", testing_labels.shape)

# Initialize net
image_size = 32 * 32 * 3
neurons_per_layer = 50
output_classes = 10
net = TwoLayerNeuralNet(image_size, neurons_per_layer, output_classes)

# Train the network
print("")
print("Training...")
stats = net.train(training_data, training_labels, validation_data, validation_labels,
            iterations=1000, batch_size=200,
            learning_rate=0.00001, learning_rate_decay=0.95,
            regularization_strength=0.00001, print_loss=True)

# Predict on the validation set
print("")
print("Validating...")
validation_accuracy = (net.predict(validation_data) == validation_labels).mean()
print("Validation accuracy: ", validation_accuracy)
