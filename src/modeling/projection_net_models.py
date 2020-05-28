from typing import List, Optional
import tensorflow as tf
from tensorflow.keras import layers
from .projection_net_layers import RandomProjectionLayer, SGNNLayer



class ProjectionNetBase(tf.keras.Model):
    def __init__(
        self,
        output_dim: int,
        T: int,
        D: int,
        std: float,
        trainable_projection: Optional[bool] = False,
        hidden_dims: Optional[List[int]] = None
        ):
        super(ProjectionNetBase, self).__init__()

        self.T = T
        self.D = D
        self.std = std
        self.trainable_projection = trainable_projection
        self.projection = None
        self.hidden_layers = []
        self.dropout_layers = []
        self.classifier = None
        

    def _get_hidden_dims(self, hidden_dims):
        hidden_layers = []
        if hidden_dims is not None:
            for dim in hidden_dims:
                hidden_layers.append(layers.Dense(dim))
        return hidden_layers
    

    def _get_dropout_layers(self, hidden_dims):
        dropout_layers = []
        if hidden_dims is not None:
            for dim in hidden_dims:
                dropout_layers.append(layers.Dropout(0.25))
        return dropout_layers


    def _get_classifier(self, output_dim):
        classifier = layers.Dense(output_dim)
        return classifier


    def _get_projection_layer(self):
        raise NotImplementedError("You have to implement setting of projection layer")
    
    
    def call(self, inputs):
        x = self.projection(inputs)
        for dense, dropout in zip(self.hidden_layers, self.dropout_layers):
            x = dense(x)
            x = dropout(x)
        return self.classifier(x)



class RandomProjectionNet(ProjectionNetBase):
    def __init__(
        self,
        output_dim: int,
        T: int,
        D: int,
        std: float,
        trainable_projection: Optional[bool] = False,
        hidden_dims: Optional[List[int]] = None
        ):
        super(RandomProjectionNet, self).__init__(
            output_dim,
            T,
            D,
            std,
            trainable_projection,
            hidden_dims=hidden_dims
            )
        self.projection = self._get_projection_layer()
        self.hidden_layers = self._get_hidden_dims(hidden_dims)
        self.dropout_layers = self._get_dropout_layers(hidden_dims)
        self.classifier = self._get_classifier(output_dim)
    

    def _get_projection_layer(self):
        return RandomProjectionLayer(
            T=self.T,
            D=self.D,
            std=self.std,
            trainable=self.trainable_projection
            )



class SGNNNet(ProjectionNetBase):
    def __init__(
        self,
        output_dim: int,
        num_word_grams: int,
        num_char_grams: int,
        T: int,
        D: int,
        std: float,
        trainable_projection: Optional[bool] = False,
        hidden_dims: Optional[List[int]] = None
        ):
        super(SGNNNet, self).__init__(
            output_dim,
            T,
            D,
            std,
            trainable_projection,
            hidden_dims=hidden_dims
            )
        self.num_word_grams = num_word_grams
        self.num_char_grams = num_char_grams
        self.projection = self._get_projection_layer()
        self.hidden_layers = self._get_hidden_dims(hidden_dims)
        self.dropout_layers = self._get_dropout_layers(hidden_dims)
        self.classifier = self._get_classifier(output_dim)
        

    
    def _get_projection_layer(self):
        return SGNNLayer(
            num_word_grams=self.num_word_grams,
            num_char_grams=self.num_char_grams,
            T=self.T, D=self.D, std=self.std,
            trainable=self.trainable_projection
        )
