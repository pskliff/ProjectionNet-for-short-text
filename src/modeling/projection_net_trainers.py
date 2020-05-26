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
from .projection_net_trainer_base import ProjectionNetTrainerBase
from data_processors.slu_data_to_bert_converter import slu_processors
from settings import TrainAdamOptimizerConfig, TrainProjectionNetConfig



class ProjectionNetTrainer(ProjectionNetTrainerBase):
    def __init__(
        self,
        task: str,
        model: ProjectionNetBase,
        train_config: TrainProjectionNetConfig,
        optimizer_config: TrainAdamOptimizerConfig
        ):
        super().__init__(task, model, train_config, optimizer_config)
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    

    def loss_fn(self, y_true, y_pred):
        return self.criterion(y_true, y_pred)


    def _run_train_epoch(
        self,
        epoch: int,
        train_dataset: tf.data.Dataset,
        steps_per_epoch: int,
        experiment = None
        ) -> List[float]:

        dataset_tqdm = tqdm(train_dataset, total=steps_per_epoch)
        train_losses = []
        train_metrics = []
        for step, (x_batch_train, y_batch_train) in enumerate(dataset_tqdm):
            # if experiment is not None:
            #     experiment.set_step(step * (1 + epoch))

            # x_batch_train = tf.cast(
            #                         x_batch_train, tf.float32
            #                     )
                
            with tf.GradientTape() as tape:
                logits = self._model(x_batch_train, training=True) 
                loss_value = self.loss_fn(y_batch_train, logits)
                
            grads = tape.gradient(loss_value, self._model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self._model.trainable_weights))
            train_metric = self.train_acc_metric(y_batch_train, logits)

            train_losses.append(loss_value)
            avg_loss = np.mean(train_losses)
            train_metrics.append(train_metric)
            avg_metric = np.mean(train_metrics)

        if experiment is not None:
            metrics_dict = {'batch_loss': loss_value,
                            'average_loss': avg_loss,
                            'average_accuracy': avg_metric}
            experiment.log_metrics(metrics_dict)

            dataset_tqdm.set_description(f'E{epoch + 1}; loss: {loss_value:.5f}; avl: {avg_loss:.5f}; av_acc: {avg_metric:.5f}')
        return train_losses




class ProjectionNetDistillTrainer(ProjectionNetTrainerBase):
    def __init__(
        self,
        task: str,
        model: ProjectionNetBase,
        train_config: TrainProjectionNetConfig,
        optimizer_config: TrainAdamOptimizerConfig,
        alpha: float
        ):
        super().__init__(task, model, train_config, optimizer_config)
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.soft_criterion = tf.keras.losses.MeanSquaredError()
        self.alpha = alpha


    def loss_fn(self, y_true, y_soft, y_pred):

        hard_loss_value = self.criterion(y_true, y_pred)
        soft_loss_value = self.soft_criterion(y_soft, y_pred)
        loss_value = self.alpha * hard_loss_value + (1 - self.alpha) * soft_loss_value
        return hard_loss_value, soft_loss_value, loss_value


    def _run_train_epoch(
        self,
        epoch: int,
        train_dataset: tf.data.Dataset,
        steps_per_epoch: int,
        experiment = None
        ) -> List[float]:

        dataset_tqdm = tqdm(train_dataset, total=steps_per_epoch)
        train_losses = []
        train_hard_losses = []
        train_soft_losses = []
        train_metrics = []
        for step, (x_batch_train, y_batch_train, y_batch_soft) in enumerate(dataset_tqdm):
            # if experiment is not None:
            #     experiment.set_step(step * (1 + epoch))

            # x_batch_train = tf.cast(
            #                         x_batch_train, tf.float32
            #                     )
                
            with tf.GradientTape() as tape:
                logits = self._model(x_batch_train, training=True) 
                hard_loss_value, soft_loss_value, loss_value = self.loss_fn(y_batch_train, y_batch_soft, logits)
                
            grads = tape.gradient(loss_value, self._model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self._model.trainable_weights))
            train_metric = self.train_acc_metric(y_batch_train, logits)

            train_losses.append(loss_value)
            train_hard_losses.append(hard_loss_value)
            train_soft_losses.append(soft_loss_value)

            avg_hard_loss = np.mean(train_hard_losses)
            avg_soft_loss = np.mean(train_soft_losses)
            avg_loss = np.mean(train_losses)
            train_metrics.append(train_metric)
            avg_metric = np.mean(train_metrics)

        if experiment is not None:
            metrics_dict = {'batch_loss': loss_value,
                            'average_loss': avg_loss,
                            'average_hard_loss': avg_hard_loss,
                            'average_soft_loss': avg_soft_loss,
                            'average_accuracy': avg_metric}
            experiment.log_metrics(metrics_dict)

            dataset_tqdm.set_description(f'E{epoch + 1}; loss: {loss_value:.5f}; avl: {avg_loss:.5f}; avl_hard: {avg_hard_loss:.5f}; avl_soft: {avg_soft_loss:.5f}; av_acc: {avg_metric:.5f}')
        return train_losses
