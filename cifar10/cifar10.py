from collections import Counter
from utils import read_and_prepare_images, count_matches
from neural_net import TwoLayerNeuralNet

# Read input images
print("Reading input data...")
training_data, training_labels, validation_data, validation_labels, testing_data, testing_labels = read_and_prepare_images()

# Initialize net
image_size = 32 * 32 * 3 # Dimension of one image (width x height x rgb)
neurons_per_layer = 50
output_classes = 10
net = TwoLayerNeuralNet(image_size, neurons_per_layer, output_classes)

# Train the network
print("")
print("Training...")
stats = net.train(training_data, training_labels, validation_data, validation_labels,
            iterations=5000, batch_size=200,
            learning_rate=0.0006, learning_rate_decay=0.95,
            regularization_strength=0.0001, print_loss=True)

# Print result for validation
print("")
print("Best validation loss at %i iterations!" %(stats['best_iterations']))
print("Result of validation...")
validation_real_labels = Counter(validation_labels)
validation_predicted_labels = net.predict(validation_data)
validation_accuracy = (validation_predicted_labels == validation_labels).mean()
validation_class_accuracy = count_matches(validation_predicted_labels, validation_labels)
print("Validation accuracy: ", validation_accuracy)
print("Validation class accuracy: ", validation_class_accuracy)
print("Validation real labels: ", validation_real_labels)

# Test the model on the test data
print("")
print("Testing with model with best validation loss...")
testing_real_labels = Counter(testing_labels)
testing_predicted_labels = net.predict(testing_data)
testing_accuracy = (testing_predicted_labels == testing_labels).mean()
testing_class_accuracy = count_matches(testing_predicted_labels, testing_labels)
print("Testing accuracy: ", testing_accuracy)
print("Testing class accuracy: ", testing_class_accuracy)
print("Testing real labels: ", testing_real_labels)
