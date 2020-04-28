
import tensorflow as tf
from tensorflow.keras import layers

class RandomProjectionLayer(tf.keras.layers.Layer):
    def __init__(self, T, D, std):
        super(RandomProjectionLayer, self).__init__()
        self.num_outputs = T*D
        self.T = T
        self.D = D
        self.std = std


    def build(self, input_shape):
        # input_shape = (batch_size, seq_len)
        seq_len = input_shape[-1]
        # shape = (T, D, seq_len) 
        self.random_planes = tf.random.normal(
                                    (self.T, self.D, seq_len),
                                    mean=0.0,
                                    stddev=self.std,
                                    dtype=tf.dtypes.float32,
                                    seed=None,
                                    name=None
                                )
        self.input_spec = layers.InputSpec(min_ndim=2,
                                axes={-1: seq_len})


    def call(self, x):
        # input = (batch_size, seq_len)
        x = tf.cast(x, dtype=tf.float32)
        # projected_values = (T, D, seq_len) x (seq_len, batch_size) = (T, D, batch_size)
        projected_values = tf.matmul(self.random_planes,
                                     x,
                                     transpose_b=True)
        
        # projected_flat = (T, D, batch_size) -> (batch_size, T*D)
        projected_flat = tf.reshape(projected_values, (self.T*self.D, -1))
        projected_flat = tf.transpose(projected_flat)

        # projected_flat = (batch_size, T*D) and contains only values [0, 1]
        projected_bits = tf.sign(projected_flat)
        projected_bits = tf.maximum(projected_bits, 0)

        return projected_bits


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.self.num_outputs)



class RandomProjectionNet(tf.keras.Model):

    def __init__(self, output_dim, T, D, std, hidden_dims=None):
        super(RandomProjectionNet, self).__init__()

        self.projection = RandomProjectionLayer(T=T, D=D, std=std)
        self.hidden_layers = []
        if hidden_dims is not None:
            for dim in hidden_dims:
                self.hidden_layers.append(layers.Dense(dim))
        self.classifier = layers.Dense(output_dim)

    def call(self, inputs):
        x = self.projection(inputs)
        for dense in self.hidden_layers:
            x = dense(x)
        return self.classifier(x)