import tensorflow as tf


class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__init__()

        # Initialize weight and bias:
        self.W = self.add_weight([input_dim, output_dim])
        self.b = self.add_weight([1, output_dim])

    def call(self, inputs):
        # Forward proparate the inputs
        z = tf.matmul(inputs, self.W) + self.b

        # Feed through a non-linear function
        output = tf.sigmoid(z)

        return output


