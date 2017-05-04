import os
from utils import read_and_prepare_images

import tensorflow as tf
from tensorflow.contrib import learn

tf.logging.set_verbosity(tf.logging.ERROR)

DEFAULT_BATCH_SIZE = 200
DEFAULT_STEPS = 500

def process(model, model_dir="/tmp/model", batch_size=DEFAULT_BATCH_SIZE, steps=DEFAULT_STEPS):
    # Read input images
    print("Reading input data...")
    training_data, training_labels, validation_data, validation_labels, test_data, test_labels = read_and_prepare_images()


    print("")
    print("Building model..")

    # Create the Estimator
    cifar10_classifier = learn.Estimator(model_fn=model, model_dir=model_dir)

    # Set up logging for predictions
    # Logging is deactivated though because we do not want to see deprecation messages
    tensors_to_log = { "probabilities": "softmax_tensor" }
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    print("")
    print("Training model...")

    # Train the model
    cifar10_classifier.fit(
        x=training_data, y=training_labels,
        batch_size=batch_size, steps=steps, monitors=[logging_hook]
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
