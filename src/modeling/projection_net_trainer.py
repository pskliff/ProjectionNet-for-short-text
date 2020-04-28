import os
import sys
import numpy as np
import tensorflow as tf
from os.path import dirname, abspath
from typing import List, Dict, Tuple, Optional, Union
from .projection_net import RandomProjectionNet
# from tqdm.notebook import tqdm
from tqdm import tqdm
from sklearn.metrics import accuracy_score

SRC_PATH = dirname(dirname(abspath(__file__)))
sys.path.append(SRC_PATH)

from data_processors.slu_data_to_bert_converter import slu_processors
from settings import TrainBertConfig, TrainAdamOptimizerConfig, TrainProjectionNetConfig


class ProjectionNetModelBase(object):
    def __init__(
        self,
        task: str,
        train_config: TrainProjectionNetConfig,
        optimizer_config: TrainAdamOptimizerConfig
        ):

        self.train_config = train_config
        self.optimizer_config = optimizer_config
        self._model = self._create_model(task=task)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=optimizer_config.learning_rate,
            epsilon=optimizer_config.epsilon
            )
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
        self.val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")


    def _create_model(self, task: str) -> tf.keras.Model:
        num_labels = len(slu_processors[task]().get_labels())
        model = RandomProjectionNet(output_dim=num_labels,
                            T=self.train_config.T,
                            D=self.train_config.D,
                            std=self.train_config.std,
                            hidden_dims=self.train_config.hidden_dims
                            )
        return model

    
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

    
    def predict_proba(self, test_dataset: Union[tf.data.Dataset, np.array]):
        return self._model.predict(test_dataset, verbose=True)
    

    def predict(self, test_dataset: Union[tf.data.Dataset, np.array]):
        y_pred_proba = self.predict_proba(test_dataset=test_dataset)
        y_pred = y_pred_proba.argmax(-1)
        return y_pred


    def evaluate(
        self,
        test_dataset: tf.data.Dataset,
        test_labels: np.array,
        ):
        X_test = np.array(list(test_dataset.as_numpy_iterator()))
        y_pred = self.predict(test_dataset=X_test)
        acc = accuracy_score(test_labels, y_pred)
        return acc
    

    def evaluate_exp(
        self,
        test_dataset: tf.data.Dataset,
        test_labels: np.array,
        experiment
        ):
        with experiment.test():
            acc = self.evaluate(test_dataset, test_labels)
            experiment.log_metric(name='Accuracy', value=acc)
        
        return acc


    def save(self, filepath: str):
        self._model.save(filepath)



class ProjectionNetModel(ProjectionNetModelBase):
    def __init__(
        self,
        task: str,
        train_config: TrainProjectionNetConfig,
        optimizer_config: TrainAdamOptimizerConfig
        ):
        super().__init__(task, train_config, optimizer_config)
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
            if experiment is not None:
                experiment.set_step(step * (1 + epoch))

            x_batch_train = tf.cast(
                                    x_batch_train, tf.float32
                                )
                
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




class ProjectionNetModelDistill(ProjectionNetModelBase):
    def __init__(
        self,
        task: str,
        train_config: TrainProjectionNetConfig,
        optimizer_config: TrainAdamOptimizerConfig,
        alpha: float
        ):
        super().__init__(task, train_config, optimizer_config)
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
            if experiment is not None:
                experiment.set_step(step * (1 + epoch))

            x_batch_train = tf.cast(
                                    x_batch_train, tf.float32
                                )
                
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
