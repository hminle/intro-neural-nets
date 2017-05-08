import os
import sys
import pickle
from collections import Counter

pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, pwd + "../deps")

import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        images = pickle.load(fo, encoding='latin1')
    return images

def load_image_batch(filename):
    data = unpickle(filename)
    records = data['data']
    records = records.astype("float32")
    return records, data['labels']

def load_images():
    training_data = []
    training_labels = []
    for i in range(1, 6):
        filename = os.path.join('./images/', 'data_batch_%d' % (i, ))
        X_batch, Y_batch = load_image_batch(filename)
        training_data.append(X_batch)
        training_labels.append(Y_batch)
    training_data = np.concatenate(training_data)
    training_labels = np.concatenate(training_labels)
    test_data, test_labels = load_image_batch(os.path.join('./images/', 'test_batch'))
    return training_data, training_labels, np.array(test_data), np.array(test_labels)

def read_and_prepare_images(training_records=49000, validation_records=1000, test_records=10000):
    training_data, training_labels, testing_data, testing_labels = load_images()

    # Divide the whole data into the three parts (training, validation, test)
    mask = range(training_records, training_records + validation_records)
    validation_data = training_data[mask]
    validation_labels = training_labels[mask]
    mask = range(training_records)
    training_data = training_data[mask]
    training_labels = training_labels[mask]
    mask = range(test_records)
    testing_data = testing_data[mask]
    testing_labels = testing_labels[mask]

    # Normalize the data: Subtract the mean image
    mean_image = np.mean(training_data, axis=0)
    training_data -= mean_image
    validation_data -= mean_image
    testing_data -= mean_image

    return training_data, training_labels, validation_data, validation_labels, testing_data, testing_labels

def get_accuracy_per_class(predicted, actual):
    accuracies = []
    counter = Counter(actual)
    matches = _count_matches(predicted, actual)
    for i in range(10):
        class_accuracy = matches['correct_%i' %(i)] / counter[i]
        class_accuracy = round(class_accuracy, 2)
        accuracies.append(class_accuracy)
    return accuracies

def _count_matches(predicted_labels, actual_labels):
    matches = {}
    for i in range(10):
        matches['correct_%i' %(i)] = 0
    for i in range(0, len(predicted_labels)):
        if predicted_labels[i]['classes'] == actual_labels[i]:
            matches['correct_%i' %(predicted_labels[i]['classes'])] += 1
    return matches

def get_top3_per_class(probabilities_array, actual_labels):
    top3_accuracies = []
    counter = Counter(actual_labels)
    matches = _count_matches_within_top_3(probabilities_array, actual_labels)
    for i in range(10):
        top3_accuracy = matches['correct_%i' %(i)] / counter[i]
        top3_accuracy = round(top3_accuracy, 2)
        top3_accuracies.append(top3_accuracy)
    return top3_accuracies

def _count_matches_within_top_3(probabilities_array, actual_labels):
    matches = {}
    for i in range(10):
        matches['correct_%i' %(i)] = 0
    for i in range(0, len(probabilities_array)):
        current_probabilites = probabilities_array[i]['probabilities']
        top_3 = np.argpartition(current_probabilites, -3)[-3:]
        if actual_labels[i] in top_3:
            matches['correct_%i' %(actual_labels[i])] += 1
    return matches
