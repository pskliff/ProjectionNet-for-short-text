import os
import sys
from os.path import dirname, abspath
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import accuracy_score


SRC_PATH = dirname(dirname(abspath(__file__)))
sys.path.append(SRC_PATH)

from .projection_net_models import ProjectionNetBase
from data_processors.slu_data_to_bert_converter import slu_processors
from settings import TrainBertConfig, TrainAdamOptimizerConfig, TrainProjectionNetConfig


class ProjectionNetTrainerBase(object):
    def __init__(
        self,
        task: str,
        model: ProjectionNetBase,
        train_config: TrainProjectionNetConfig,
        optimizer_config: TrainAdamOptimizerConfig
        ):

        self.train_config = train_config
        self.optimizer_config = optimizer_config
        self._model = model
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=optimizer_config.learning_rate,
            epsilon=optimizer_config.epsilon
            )
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
        self.val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")


    def _create_model(self, task: str) -> tf.keras.Model:
        raise NotImplementedError("You have to implement model creation")

    
    def loss_fn(self, y_true, y_pred):
        return self.criterion(y_true, y_pred)


    def _run_train_epoch(
        self,
        epoch: int,
        train_dataset: tf.data.Dataset,
        steps_per_epoch: int,
        experiment = None
        ) -> List[float]:
        raise NotImplementedError("You have to implement training procedure")

    
    def _run_valid_epoch(
        self,
        epoch: int,
        valid_dataset: tf.data.Dataset,
        train_losses: List[float],
        experiment = None
        ):
        val_losses = []
        for x_batch_val, y_batch_val in valid_dataset:
            x_batch_val = tf.cast(
                        x_batch_val, tf.float32
                    )
            
            val_logits = self._model(x_batch_val)
            val_loss = self.criterion(y_batch_val, val_logits)

            val_losses.append(val_loss)
            self.val_acc_metric(y_batch_val, val_logits)
        
        avg_train_loss = np.mean(train_losses)
        avg_train_metric = self.train_acc_metric.result()
        avg_val_loss = np.mean(val_losses)
        avg_val_metric = self.val_acc_metric.result()

        self.train_acc_metric.reset_states()
        self.val_acc_metric.reset_states()

        if experiment is not None:
            metrics_dict = {'average_train_loss': avg_train_loss,
                            'average_train_metric': avg_train_metric,
                            'average_loss': avg_val_loss,
                            'average_accuracy': avg_val_metric}
            experiment.log_metrics(metrics_dict)

        metric_line = f"E = {epoch + 1}; train_loss: {avg_train_loss:.7f};\
        train_accuracy: {avg_train_metric:.7f}; valid_loss: {avg_val_loss:.7f};\
        valid_accuracy: {avg_val_metric:.7f}\n"
        print(metric_line)


    def fit_exp(
        self,
        train_dataset: tf.data.Dataset,
        validation_data: tf.data.Dataset,
        steps_per_epoch: int,
        experiment
        ):
        epochs = self.train_config.epochs
        for epoch in range(epochs):
            print(f'Epoch {epoch}/{epochs}')
            experiment.set_epoch(epoch)
            with experiment.train():
                train_losses = self._run_train_epoch(epoch, train_dataset, steps_per_epoch=steps_per_epoch, experiment=experiment)
            
            with experiment.validate():
                self._run_valid_epoch(epoch, validation_data, train_losses, experiment=experiment)


    def fit(
        self,
        train_dataset: tf.data.Dataset,
        validation_data: tf.data.Dataset,
        steps_per_epoch: int,
        ):
        epochs = self.train_config.epochs
        for epoch in range(epochs):
            print(f'Epoch {epoch}/{epochs}')    
            train_losses = self._run_train_epoch(epoch, train_dataset, steps_per_epoch=steps_per_epoch)
            self._run_valid_epoch(epoch, validation_data, train_losses)

    
    def predict_proba(self, test_dataset: tf.Tensor) -> np.ndarray:
        # return self._model.predict(test_dataset, verbose=True)
        return self._model(test_dataset, training=False).numpy()
    

    def predict(self, test_dataset: tf.Tensor) -> np.ndarray:
        y_pred_proba = self.predict_proba(test_dataset=test_dataset)
        y_pred = y_pred_proba.argmax(-1)
        return y_pred


    def evaluate(
        self,
        test_dataset: tf.Tensor,
        test_labels: np.array,
        ):
        y_pred = self.predict(test_dataset=test_dataset)
        acc = accuracy_score(test_labels, y_pred)
        return acc
    

    def evaluate_exp(
        self,
        test_dataset: tf.Tensor,
        test_labels: np.array,
        experiment
        ):
        with experiment.test():
            acc = self.evaluate(test_dataset, test_labels)
            experiment.log_metric(name='Accuracy', value=acc)
        
        return acc


    def save(self, filepath: str):
        self._model.save(filepath)
