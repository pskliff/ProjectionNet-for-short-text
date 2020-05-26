import os
import sys
from os.path import dirname, abspath
import datetime
from typing import Dict, Union, Optional, Any, Tuple
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.text import Tokenizer


from data_processors.slu_data_to_bert_converter import slu_processors
from data_processors.tokenizers import SGNNTokenizer
from data_pipelines.create_tf_datasets_for_projection_base import ProjectionDatasetReaderBase
from modeling.projection_net_models import (
    RandomProjectionNet,
    SGNNNet
)
from modeling.projection_net_trainer_base import ProjectionNetTrainerBase
from modeling.projection_net_trainers import (
    ProjectionNetTrainer,
    ProjectionNetDistillTrainer
)
from data_pipelines.create_tf_datasets_for_projection import (
    ProjectionDatasetReader,
    ProjectionDatasetReaderDistill
)
from settings import (
    TrainProjectionNetConfig,
    TrainAdamOptimizerConfig,
    ProjectionNetConfig,
    SGGNNetConfig 
)



class PipelineBase(object):
    def __init__(
        self,
        basic_params,
        model_params,
        train_params,
        optimizer_params,
        is_distill: Optional[bool] = False,
        experiment: Optional = None):
        self.tokenizer = self._get_tokenizer(model_params=model_params)
        self.data_reader = self._get_data_reader(
            basic_params=basic_params,
            model_params=model_params,
            is_distill=is_distill
            )
        self.train_config = self._get_train_config(train_params=train_params)
        self.optimizer_config = self._get_optimizer_config(optimizer_params=optimizer_params)
        self.model_config = self._get_model_config(model_params=model_params)
        self.task = basic_params['task']
        self.is_distill = is_distill
        self.loss_alpha = model_params['loss_alpha'] if is_distill else 0
        self.save_path = basic_params['models_path']
        self.teacher_name = basic_params['teacher_name'] if is_distill else 'No_Teacher'
        self.experiment = experiment
        super().__init__()


    def _get_model_config(
        self,
        model_params: Dict[str, Union[str, int, float]]
        ):
        raise NotImplementedError("Depends on model, Must be implemented in heir class")


    def _get_train_config(
        self,
        train_params: Dict[str, Union[str, int, float]]
        ) -> TrainProjectionNetConfig:
        train_config = TrainProjectionNetConfig(
            batch_size=train_params['batch_size'],
            eval_batch_size=train_params['eval_batch_size'],
            epochs=train_params['epochs']
        )
        return train_config


    def _get_optimizer_config(self, optimizer_params: Dict[str, float]):
        optimizer_config = TrainAdamOptimizerConfig(
            learning_rate=optimizer_params['learning_rate'],
            epsilon=optimizer_params['adam_epsilon']
        )
        return optimizer_config


    def _get_tokenizer(
        self,
        model_params: Dict[str, Union[str, int, float]]
        ) -> Any:
        raise NotImplementedError("Tokenizer must be implemented based on model")


    def _get_data_reader(
        self,
        basic_params: Dict[str, Union[str, int, float]],
        model_params: Dict[str, Union[str, int, float]],
        is_distill: Optional[bool] = False
        ) -> ProjectionDatasetReaderBase:
        if is_distill:
            data_reader = ProjectionDatasetReaderDistill(
                data_path=basic_params['data_path'],
                task=basic_params['task'],
                max_features=model_params['max_features'],
                tokenizer=self.tokenizer,
                logits_filepath=basic_params['logits_path'],
                tokenizer_mode=model_params['tokenizer_mode'],
                is_pretrained_tokenizer=False
                )
        else:
            data_reader = ProjectionDatasetReader(
                data_path=basic_params['data_path'],
                task=basic_params['task'],
                max_features=model_params['max_features'],
                tokenizer=self.tokenizer,
                tokenizer_mode=model_params['tokenizer_mode'],
                is_pretrained_tokenizer=False
            )


        return data_reader

    
    def _get_datasets(
        self,
        data_reader: ProjectionDatasetReaderBase,
        train_config: TrainProjectionNetConfig
        ) -> Tuple[int, tf.data.Dataset, int, tf.data.Dataset, int, tf.data.Dataset, np.array]:
        train_examples, train_dataset = data_reader.get_train_dataset(
            folder_name='train',
            batch_size=train_config.batch_size,
            shuffle=True
            )
        
        valid_examples, valid_dataset = data_reader.get_valid_dataset(
            folder_name='valid',
            batch_size=train_config.batch_size
            )

        test_examples, test_dataset, test_labels = data_reader.get_test_dataset(
            folder_name='test',
            batch_size=train_config.batch_size
            )

        return train_examples, train_dataset, \
                valid_examples, valid_dataset, \
                test_examples, test_dataset, test_labels


    def _create_model(self, task: str) -> tf.keras.Model:
        raise NotImplementedError("Depends on the model, must be implemented in heir")
    

    def _get_trainer(
        self,
        model: tf.keras.Model,
        is_distill: bool
        ) -> ProjectionNetTrainerBase:
        if is_distill:
            trainer = ProjectionNetDistillTrainer(
                task=self.task,
                model=model,
                train_config=self.train_config,
                optimizer_config=self.optimizer_config,
                alpha=self.loss_alpha
                )
        else:
            trainer = ProjectionNetTrainer(
                task=self.task,
                model=model,
                train_config=self.train_config,
                optimizer_config=self.optimizer_config
                )
        return trainer


    def _save_model(
        self,
        model: tf.keras.Model,
        save_path: str,
        model_type: str,
        experiment = None
        ) -> str:
        
        model_path = os.path.join(save_path, 'projections')

        if isinstance(self.model_config.hidden_dims, list):
            phd = '_'.join(list(map(str, self.model_config.hidden_dims)))
        else:
            phd = 'None'

        model_name = f"{model_type}\
_{self.task}\
_{self.teacher_name}\
__T_{self.model_config.T}_D_{self.model_config.D}_std_{self.model_config.std}\
_HD_{phd}\
__DT_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}"
    
        model_filepath = os.path.join(model_path,  model_name)
        model.save(model_filepath)
        if experiment is not None:
            experiment.log_model(model_name, model_filepath)
        return model_filepath


    def _save_tflite_model(self, model, save_filepath) -> Tuple[str, int]:
        tflite_model_filepath = save_filepath + '.tflite'
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                            tf.lite.OpsSet.SELECT_TF_OPS]
        converter.post_training_quantize = True
        tflite_buffer = converter.convert()
        with open(tflite_model_filepath, 'wb') as f:
            f.write(tflite_buffer)

        return tflite_model_filepath, os.path.getsize(tflite_model_filepath)//1024


    def _count_model_params(self, model):
        trainable_count = int(np.sum([K.count_params(p) for p in model.trainable_weights]))
        non_trainable_count = int(np.sum([K.count_params(p) for p in model.non_trainable_weights]))
        return trainable_count + non_trainable_count, trainable_count, non_trainable_count


    def _run(self):
        train_examples, train_dataset, \
        valid_examples, valid_dataset, \
        test_examples, test_dataset, test_labels = self._get_datasets(
            data_reader=self.data_reader,
            train_config=self.train_config
            )
        
        print("Number of examples: ", train_examples, valid_examples, test_examples)

        model = self._create_model(task=self.task)
        trainer = self._get_trainer(model=model, is_distill=self.is_distill)


        steps_per_epoch = train_examples // self.train_config.batch_size
        trainer.fit_exp(
            train_dataset=train_dataset,
            validation_data=valid_dataset,
            steps_per_epoch=steps_per_epoch,
            experiment=self.experiment
            )

        test_metric = trainer.evaluate_exp(
            test_dataset=test_dataset,
            test_labels=test_labels,
            experiment=self.experiment
            )
        
        print("\n\n")
        print(trainer._model.summary())

        return trainer, test_metric
    

    def run(self, num_teacher_params: int = 0, teacher_model_size_kb: int = 0, teacher_accuracy: float = 0.98):
        raise NotImplementedError()




class ProjectionPipeline(PipelineBase):
    def __init__(
        self,
        basic_params,
        model_params,
        train_params,
        optimizer_params,
        is_distill: Optional[bool] = False,
        experiment: Optional = None
        ):
        super().__init__(
            basic_params,
            model_params,
            train_params,
            optimizer_params,
            is_distill=is_distill,
            experiment=experiment
            )
        self.model_type = "RandomProjectionNet"
    
    def _get_model_config(
        self,
        model_params: Dict[str, Union[str, int, float]]
        ):
        return ProjectionNetConfig(
            T=model_params['T'],
            D=model_params['D'],
            std=model_params['std'],
            trainable_projection=model_params['trainable_projection'],
            hidden_dims=model_params['hidden_dims']
            )

    
    def _get_tokenizer(
        self,
        model_params: Dict[str, Union[str, int, float]]
        ) -> Any:
        return Tokenizer(num_words=model_params['max_features'])

    
    def _create_model(self, task: str) -> tf.keras.Model:
        num_labels = len(slu_processors[task]().get_labels())
        model = RandomProjectionNet(
            output_dim=num_labels,
            T=self.model_config.T,
            D=self.model_config.D,
            std=self.model_config.std,
            trainable_projection=self.model_config.trainable_projection,
            hidden_dims=self.model_config.hidden_dims
        )
        return model


    def run(self, num_teacher_params: int = 0, teacher_model_size_kb: int = 0, teacher_accuracy: float = 0.98):
        trainer, test_metric = self._run()

        num_params, num_trainable_params, num_non_trainable_params = self._count_model_params(trainer._model)
        

        model_filepath = self._save_model(
            model=trainer._model,
            save_path=self.save_path,
            model_type=self.model_type,
            experiment=self.experiment
            )

        tflite_model_filepath, model_size = self._save_tflite_model(
            model=trainer._model,
            save_filepath=model_filepath
            )

        if self.experiment is not None:
                dMSp = 100 - 100 * (num_trainable_params / num_teacher_params)
                dMSs = 100 - 100 * (model_size / teacher_model_size_kb)
                dA = 100 * (teacher_accuracy - test_metric)
                metrics_dict = {'num_params': num_params,
                                'num_trainable_params': num_trainable_params,
                                'num_non_trainable_params': num_non_trainable_params,
                                'params_compression': num_teacher_params / num_params,
                                'trainable_params_compression': num_teacher_params / num_trainable_params,
                                'tflite_model_size_kilobytes': model_size,
                                'tflite_model_size_megabytes': model_size//1024,
                                'model_size_compression': teacher_model_size_kb / model_size,
                                'dMSp': dMSp,
                                'dMSs': dMSs,
                                'dA': dA,
                                'dMSp/dA': dMSp/dA,
                                'dMSs/dA': dMSs/dA,
                                }
                self.experiment.log_metrics(metrics_dict)
        


class SGNNPipeline(PipelineBase):
    def __init__(
        self,
        basic_params,
        model_params,
        train_params,
        optimizer_params,
        is_distill: Optional[bool] = False,
        experiment: Optional = None
        ):
        super().__init__(
            basic_params,
            model_params,
            train_params,
            optimizer_params,
            is_distill=is_distill,
            experiment=experiment
            )
        self.model_type = "SGNNNet"
    

    def _get_model_config(
        self,
        model_params: Dict[str, Union[str, int, float]]
        ):
        word_ngram_diff = model_params['word_ngram_range'][1] - model_params['word_ngram_range'][0] + 1
        char_ngram_diff = model_params['char_ngram_range'][1] - model_params['char_ngram_range'][0] + 1
        return SGGNNetConfig(
            T=model_params['T'],
            D=model_params['D'],
            std=model_params['std'],
            num_word_grams=word_ngram_diff,
            num_char_grams=char_ngram_diff,
            word_ngram_range=model_params['word_ngram_range'],
            char_ngram_range=model_params['char_ngram_range'],
            trainable_projection=model_params['trainable_projection'],
            hidden_dims=model_params['hidden_dims']
            )

    
    def _get_tokenizer(
        self,
        model_params: Dict[str, Union[str, int, float]]
        ) -> Any:
        return SGNNTokenizer(
                word_ngram_range=(
                    model_params['word_ngram_range'][0],
                    model_params['word_ngram_range'][1]
                    ),
                char_ngram_range=(
                    model_params['char_ngram_range'][0],
                    model_params['char_ngram_range'][1]
                    ),
                max_features=model_params['max_features']
                )

    
    def _create_model(self, task: str) -> tf.keras.Model:
        num_labels = len(slu_processors[task]().get_labels())
        model = SGNNNet(
            output_dim=num_labels,
            num_word_grams=self.model_config.num_word_grams,
            num_char_grams=self.model_config.num_char_grams,
            T=self.model_config.T,
            D=self.model_config.D,
            std=self.model_config.std,
            trainable_projection=self.model_config.trainable_projection,
            hidden_dims=self.model_config.hidden_dims
        )
        return model


    def run(self, num_teacher_params: int = 0, teacher_model_size_kb: int = 0, teacher_accuracy: float = 0.98):
        trainer, test_metric = self._run()

        num_params, num_trainable_params, num_non_trainable_params = self._count_model_params(trainer._model)
        

        model_filepath = self._save_model(
            model=trainer._model,
            save_path=self.save_path,
            model_type=self.model_type,
            experiment=self.experiment
            )

        tflite_model_filepath, model_size = self._save_tflite_model(
            model=trainer._model,
            save_filepath=model_filepath
            )

        if self.experiment is not None:
                dMSp = 100 - 100 * (num_trainable_params / num_teacher_params)
                dMSs = 100 - 100 * (model_size / teacher_model_size_kb)
                dA = 100 * (teacher_accuracy - test_metric)
                metrics_dict = {'num_params': num_params,
                                'num_trainable_params': num_trainable_params,
                                'num_non_trainable_params': num_non_trainable_params,
                                'params_compression': num_teacher_params / num_params,
                                'trainable_params_compression': num_teacher_params / num_trainable_params,
                                'tflite_model_size_kilobytes': model_size,
                                'tflite_model_size_megabytes': model_size//1024,
                                'model_size_compression': teacher_model_size_kb / model_size,
                                'dMSp': dMSp,
                                'dMSs': dMSs,
                                'dA': dA,
                                'dMSp/dA': dMSp/dA,
                                'dMSs/dA': dMSs/dA,
                                }
                self.experiment.log_metrics(metrics_dict)        
    
    
       