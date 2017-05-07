import os
from utils import read_and_prepare_images, get_accuracy_per_class, get_top3_per_class

import tensorflow as tf
from tensorflow.contrib import learn

# tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.set_verbosity(tf.logging.INFO)

DEFAULT_BATCH_SIZE = 200
DEFAULT_STEPS = 500

def process(model, model_dir=None, batch_size=DEFAULT_BATCH_SIZE, steps=DEFAULT_STEPS, use_gpu=False):
    # Read input images
    print("Reading input data...")
    training_data, training_labels, validation_data, validation_labels, test_data, test_labels = read_and_prepare_images()


    print("")
    print("Building model..")

    # Create the Estimator
    cifar10_classifier = learn.Estimator(model_fn=model, model_dir=model_dir)

    # Set up logger for validation error
    # Set up early stopping if validation loss does not decrease for 100 steps
    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        validation_data, validation_labels, every_n_steps=50,
        early_stopping_metric="loss", early_stopping_metric_minimize=True, early_stopping_rounds=100
    )

    print("")
    print("Training model...")

    # Choose right device
    device = '/cpu:0'
    if use_gpu:
        device = '/gpu:0'

    # Train the model
    with tf.device(device):
        cifar10_classifier.fit(
            x=training_data, y=training_labels,
            batch_size=batch_size, steps=steps, monitors=[validation_monitor]
        )

    # Configure the accuracy metric for evaluation
    metrics = {
        "accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes")
    }

    # Evaluate the model and print results
    results = cifar10_classifier.evaluate(
        x=test_data, y=test_labels, metrics=metrics
    )

    print("")
    print(results)

    print("")
    print("----------")
    print("")
    validation_predicted_labels = list(cifar10_classifier.predict(validation_data))
    validation_class_accuracy = get_accuracy_per_class(validation_predicted_labels, validation_labels)
    validation_top3_accuracy = get_top3_per_class(validation_predicted_labels, validation_labels)
    print("")
    print("Accuracy per class for validation", validation_class_accuracy)
    print("TOP3 Accuracy per class for validation", validation_top3_accuracy)

    test_predicted_labels = list(cifar10_classifier.predict(test_data))
    test_class_accuracy = get_accuracy_per_class(test_predicted_labels, test_labels)
    test_top3_accuracy = get_top3_per_class(test_predicted_labels, test_labels)
    print("")
    print("Accuracy per class for testing", test_class_accuracy)
    print("TOP3 Accuracy per class for testing", test_top3_accuracy)
