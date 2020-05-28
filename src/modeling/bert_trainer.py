import os
import sys
import tensorflow as tf
from os.path import dirname, abspath
from transformers import (
                            BertConfig,
                            TFBertForSequenceClassification,
                        )

SRC_PATH = dirname(dirname(abspath(__file__)))
sys.path.append(SRC_PATH)

from data_processors.slu_data_to_bert_converter import slu_processors
from settings import TrainBertConfig, TrainAdamOptimizerConfig


class BertModel(object):
    def __init__(
        self,
        task: str,
        train_config: TrainBertConfig,
        optimizer_config: TrainAdamOptimizerConfig
        ):

        self.train_config = train_config
        self.optimizer_config = optimizer_config
        self._model = self._create_model(
            bert_model_name=train_config.bert_model_name,
            task=task
            )
        self._compile_model(optimizer_config)
        print(self._model.summary())


    def _create_model(self, bert_model_name: str, task: str):
        num_labels = len(slu_processors[task]().get_labels())
        bert_config = BertConfig.from_pretrained(
                            bert_model_name,
                            num_labels=num_labels
                            )
        return TFBertForSequenceClassification.from_pretrained(
                                                    bert_model_name,
                                                    config=bert_config
                                                    )


    def _compile_model(self, optimizer_config: TrainAdamOptimizerConfig):
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=optimizer_config.learning_rate,
            epsilon=optimizer_config.epsilon
            )
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
        self._model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    
    def fit(
        self,
        train_dataset: tf.data.Dataset,
        validation_data: tf.data.Dataset,
        num_train_examples: int,
        num_valid_examples:  int
        ):
        train_steps = num_train_examples // self.train_config.batch_size
        valid_steps = num_valid_examples // self.train_config.eval_batch_size
        history = self._model.fit(
                    train_dataset,
                    epochs=self.train_config.epochs,
                    steps_per_epoch=train_steps,
                    validation_data=validation_data,
                    validation_steps=valid_steps,
                    )

    
    def predict(self, test_dataset: tf.data.Dataset):
        return self._model.predict(test_dataset, verbose=True)
