import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

class SampleArchitecture4(object):
    height = 32
    width = 32
    channels = 3
    learning_rate = 0.001

    @classmethod
    def get_model(cls, features, labels, mode):
        # Input Layer
        input_layer = tf.reshape(features, [
            -1, SampleArchitecture4.height, SampleArchitecture4.width, SampleArchitecture4.channels
        ])

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer, filters=64, kernel_size=[5, 5],
            padding="same", activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=1e-4)
        )

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2, padding="same")

        # Normalization Layer #1
        norm1 = tf.nn.local_response_normalization(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        # Convolutional Layer #2 and Normalization Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=input_layer, filters=64, kernel_size=[5, 5],
            padding="same", activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2)
        )
        norm2 = tf.nn.local_response_normalization(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        pool2 = tf.layers.max_pooling2d(norm2, pool_size=[3, 3], strides=2, padding="same")

        # Dense Layer #1 with 384 neurons
        pool2_flat = tf.reshape(pool2, [-1, 16 * 16 * 64]) # Flatten pool2 which has these dimensions
        dense1 = tf.layers.dense(
            inputs=pool2_flat, units=384, activation=tf.nn.relu,
            kernel_initializer=tf.constant_initializer(0.1)
        )

        # Dense Layer #2 with 192 neurons
        dense2 = tf.layers.dense(
            inputs=dense1, units=192, activation=tf.nn.relu,
            kernel_initializer=tf.constant_initializer(0.1)
        )

        # Logits Layer
        logits = tf.layers.dense(inputs=dense2, units=10)

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
                learning_rate=SampleArchitecture4.learning_rate,
                optimizer="SGD"
            )

        # Generate Predictions
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(input=logits, name="softmax_tensor")
        }

        # Return a ModelFnOps object
        return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)
