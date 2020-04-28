
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
        epochs: int,
        T: int,
        D: int,
        std: float,
        hidden_dims: Optional[List[int]] = None
    ):
        super().__init__(
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            epochs=epochs
            )
        self.T = T
        self.D = D
        self.std = std
        self.hidden_dims = hidden_dims


class TrainAdamOptimizerConfig(object):

    def __init__(
        self,
        learning_rate: float,
        epsilon: float
    ):
        self.learning_rate = learning_rate
        self.epsilon = epsilon



