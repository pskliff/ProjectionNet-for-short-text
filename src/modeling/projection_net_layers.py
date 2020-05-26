from typing import Optional
import tensorflow as tf
from tensorflow.keras import layers



# class RandomProjectionLayer(tf.keras.layers.Layer):
#     def __init__(
#         self,
#         T: int,
#         D: int,
#         std: float,
#         trainable: bool = False
#         ):
#         super(RandomProjectionLayer, self).__init__()
#         self.num_outputs = T*D
#         self.T = T
#         self.D = D
#         self.std = std
#         self.trainable = trainable


#     def build(self, input_shape):
#         # input_shape = (batch_size, seq_len)
#         seq_len = input_shape[-1]
#         # shape = (T, D, seq_len)
#         w_init = tf.random_normal_initializer(
#                     mean=0.0, stddev=self.std, seed=None
#                     )
#         self.random_planes = self.add_weight("random_planes",
#                                   shape=(self.T, self.D, seq_len),
#                                   initializer=w_init,
#                                   trainable=self.trainable)
        
#         self.input_spec = layers.InputSpec(min_ndim=2,
#                                 axes={-1: seq_len})


#     def call(self, x):
#         # input = (batch_size, seq_len)
#         x = tf.cast(x, dtype=tf.float32)
#         # projected_values = (T, D, seq_len) x (seq_len, batch_size) = (T, D, batch_size)
#         projected_values = tf.matmul(self.random_planes,
#                                      x,
#                                      transpose_b=True)
        
#         # projected_flat = (T, D, batch_size) -> (batch_size, T*D)
#         projected_flat = tf.reshape(projected_values, (self.T * self.D, -1))
#         projected_flat = tf.transpose(projected_flat)

#         # projected_flat = (batch_size, T*D) and contains only values [0, 1]
#         projected_bits = tf.sign(projected_flat)
#         projected_bits = tf.maximum(projected_bits, 0)

#         return projected_bits


#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.num_outputs)



# class SGNNLayer(tf.keras.layers.Layer):
#     def __init__(
#         self,
#         num_word_grams: int,
#         num_char_grams: int,
#         T: int,
#         D: int,
#         std: float,
#         trainable: Optional[bool] = False):
#         super(SGNNLayer, self).__init__()
#         self.num_outputs = T * D
#         self.T = T
#         self.D = D
#         self.std = std
#         self.K = T // (num_word_grams + num_char_grams)
#         self.random_planes = self._create_random_projection_layers(
#             num_word_grams,
#             num_char_grams,
#             trainable=trainable
#             )


#     def _create_random_projection_layers(
#         self,
#         num_word_grams: int,
#         num_char_grams: int,
#         trainable: bool) -> list:
#         random_planes = []
#         for i in range(num_word_grams + num_char_grams):
#             random_planes.append(
#                 RandomProjectionLayer(
#                     T=self.K,
#                     D=self.D,
#                     std=self.std,
#                     trainable=trainable
#                 )
#             )
#         return random_planes


#     def call(self, x):
#         assert len(x) == len(self.random_planes), \
#          f"Wrong number of inputs: expected {len(self.random_planes)}, got {len(x)}"
#         # tf.Assert(len(x) == len(self.random_planes), data=[x])
#         # with tf.control_dependencies([assert_op]):
#         projected = []
#         for i in range(len(x)):
#             projected.append(self.random_planes[i](x[i]))

#         return tf.concat(projected, axis=1)
    








class RandomProjectionLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        T: int,
        D: int,
        std: float,
        trainable: bool = False,
        seed: int = 42
        ):
        super(RandomProjectionLayer, self).__init__()
        self.num_outputs = T*D
        self.T = T
        self.D = D
        self.std = std
        self.trainable = trainable
        self.seed = seed


    def build(self, input_shape):
        # input_shape = (batch_size, seq_len)
        self.seq_len = input_shape[-1]
        # shape = (T, D, seq_len)

        
        self.input_spec = layers.InputSpec(min_ndim=2,
                                axes={-1: self.seq_len})


    def _generate_hash_functions(self):
        tf.random.set_seed(self.seed)
        return tf.random.normal(
                                    (self.T, self.D, self.seq_len),
                                    mean=0.0,
                                    stddev=self.std,
                                    dtype=tf.dtypes.float32,
                                    seed=self.seed,
                                    name="random_planes"
                                )

    def call(self, x):
        # input = (batch_size, seq_len)
        x = tf.cast(x, dtype=tf.float32)
        
        random_planes = self._generate_hash_functions()
        
        # projected_values = (T, D, seq_len) x (seq_len, batch_size) = (T, D, batch_size)
        projected_values = tf.matmul(random_planes,
                                     x,
                                     transpose_b=True)

        # projected_flat = (T, D, batch_size) -> (batch_size, T*D)
        projected_flat = tf.reshape(projected_values, (self.T * self.D, -1))
        projected_flat = tf.transpose(projected_flat)

        # projected_flat = (batch_size, T*D) and contains only values [0, 1]
        projected_bits = tf.sign(projected_flat)
        projected_bits = tf.maximum(projected_bits, 0)

        return projected_bits


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_outputs)



class SGNNLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        num_word_grams: int,
        num_char_grams: int,
        T: int,
        D: int,
        std: float,
        trainable: Optional[bool] = False):
        super(SGNNLayer, self).__init__()
        self.num_outputs = T * D
        self.T = T
        self.D = D
        self.std = std
        self.K = T // (num_word_grams + num_char_grams)
        self.random_planes = self._create_random_projection_layers(
            num_word_grams,
            num_char_grams,
            trainable=trainable
            )


    def _create_random_projection_layers(
        self,
        num_word_grams: int,
        num_char_grams: int,
        trainable: bool) -> list:
        return RandomProjectionLayer(
                    T=self.K,
                    D=self.D,
                    std=self.std,
                    trainable=trainable
                )


    def call(self, x):
        projected = []
        for i in range(len(x)):
            projected.append(self.random_planes(x[i]))

        return tf.concat(projected, axis=1)
