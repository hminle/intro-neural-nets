import os
import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def load_image_batch(filename):
    data = unpickle(filename)
    records = data['data']
    records = records.astype("float")
    # records = records.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    return records, data['labels']

def load_images():
    training_data = []
    training_labels = []
    for i in range(1,6):
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
