import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

class SampleArchitecture2(object):
    height = 32
    width = 32
    channels = 3
    learning_rate = 0.001

    @classmethod
    def get_model(cls, features, labels, mode):
        # Input Layer
        input_layer = tf.reshape(features, [
            -1, SampleArchitecture2.height, SampleArchitecture2.width, SampleArchitecture2.channels
        ])

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer, filters=32, kernel_size=[5, 5],
            padding="same", activation=tf.nn.relu
        )

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1, filters=32, kernel_size=[5, 5],
            padding="same", activation=tf.nn.relu
        )
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Convolutional Layer #3 (No Pooling)
        conv3 = tf.layers.conv2d(
            inputs=pool2, filters=64, kernel_size=[5, 5],
            padding="same", activation=tf.nn.relu
        )
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

        # Dense Layer
        pool3_flat = tf.reshape(pool3, [-1, 4 * 4 * 64]) # Flatten pool3 which has these dimensions
        dense = tf.layers.dense(inputs=pool3_flat, units=64)

        # Logits Layer
        logits = tf.layers.dense(inputs=dense, units=10)

        loss = None
        train_op = None

        # Calculate Loss (for both TRAIN and EVAL modes)
        if mode != learn.ModeKeys.INFER:
            onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
            loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == learn.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=SampleArchitecture2.learning_rate,
                optimizer="SGD"
            )

        # Generate Predictions
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        # Return a ModelFnOps object
        return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)
