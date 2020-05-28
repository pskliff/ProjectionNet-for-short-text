
from typing import List, Dict, Tuple, Optional, Union


class BertTrainSettings:
    SEQ_LEN = 128


class DataSettings:
    DATA_PATH = ''


class TaskSettings:
    TASK = 'snips'


class TrainConfig(object):
    def __init__(
        self,
        batch_size: int,
        eval_batch_size: int,
        epochs: int
    ):
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.epochs = epochs
    

class TrainBertConfig(TrainConfig):

    def __init__(
        self,
        bert_model_name: str,
        batch_size: int,
        eval_batch_size: int,
        epochs: int
    ):
        super().__init__(
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            epochs=epochs
            )
        self.bert_model_name = bert_model_name


class TrainProjectionNetConfig(TrainConfig):

    def __init__(
        self,
        batch_size: int,
        eval_batch_size: int,
        epochs: int
    ):
        super().__init__(
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            epochs=epochs
            )


class TrainAdamOptimizerConfig(object):

    def __init__(
        self,
        learning_rate: float,
        epsilon: float
    ):
        self.learning_rate = learning_rate
        self.epsilon = epsilon


class ProjectionNetConfig(object):
    def __init__(
        self,
        T: int,
        D: int,
        std: float,
        trainable_projection: Optional[bool] = False,
        hidden_dims: Optional[List[int]] = None
    ):
        self.T = T
        self.D = D
        self.std = std
        self.hidden_dims = hidden_dims
        self.trainable_projection = trainable_projection


class SGGNNetConfig(ProjectionNetConfig):
    def __init__(
        self,
        T: int,
        D: int,
        std: float,
        num_word_grams: int,
        num_char_grams: int,
        word_ngram_range: Tuple[int, int],
        char_ngram_range: Tuple[int, int],
        trainable_projection: Optional[bool] = False,
        hidden_dims: Optional[List[int]] = None
    ):
        super().__init__(
            T, D, std,
            trainable_projection=trainable_projection,
            hidden_dims=hidden_dims
            )
        
        self.num_word_grams = num_word_grams
        self.num_char_grams = num_char_grams
        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range