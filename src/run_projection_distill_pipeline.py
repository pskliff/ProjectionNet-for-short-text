# import comet_ml in the top of your file
from comet_ml import Experiment
    
# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="NS98Kn8X5OBW4OJgNA9rtsbld", 
                        project_name="test-project", workspace="kstoufel")

import os
import logging
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score
import argparse
import datetime
from typing import List, Dict, Tuple, Optional, Union
from utils import set_seed
from settings import TrainProjectionNetConfig, TrainAdamOptimizerConfig
from data_pipelines.create_tf_datasets_for_projection import ProjectionDatasetReaderDistill
from modeling.projection_net_trainer import ProjectionNetModelDistill


logging.basicConfig(
    format = u'%(filename)s[LINE:%(lineno)d] # %(levelname)-8s [%(asctime)s]  %(message)s',
    level = logging.INFO
    )



def parse_hidden_dims(hidden_dims: str) -> Union[List[int], None]:
    if hidden_dims == 'None' or hidden_dims == '0':
        return None
    else:
        return [int(dim.strip()) for dim in hidden_dims.split(',')]


def get_parser():
    parser = argparse.ArgumentParser(description='Runs ProjectionNetwork (w\o distillation)')

    # basic settings
    parser.add_argument('--proj_path', type=str, help='Project path')
    parser.add_argument('--data_path', type=str, help='Datasets path')
    parser.add_argument('--logits_path', type=str, help='Path to trainer logits')
    parser.add_argument('--random_seed', type=int, help='Random seed')
    parser.add_argument('--task', type=str, help='Project path')
    parser.add_argument('--teacher_name', type=str, help='Teacher model name')


    # Model settings
    parser.add_argument('--loss_alpha', type=float, help='Defines tradeoff between soft and hard losses')
    parser.add_argument('--T', type=int, help='Number of hash tables')
    parser.add_argument('--D', type=int, help='Projection dimension')
    parser.add_argument('--std', type=float, help='Standart deviation for random projection matrix values')
    parser.add_argument('--max_features', type=int, help='Vocab size')
    parser.add_argument('--hidden_dims', type=str, help='Output dims for optional fully connected layers')
    parser.add_argument('--tokenizer_mode', type=str, choices=['count', 'freq'], help='Which text vectorizer to use')

    # Train procedure settings
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--eval_batch_size', type=int, help='Batch size for evaluation')
    parser.add_argument('--epochs', type=int, help='Number of train epochs')

    # Optimizer settings
    parser.add_argument('--learning_rate', type=float, help='Optimizer learning rate')
    parser.add_argument('--adam_epsilon', type=float, help='Adam Optimizer epsilon')

    return parser


def args_to_dicts(args) -> Tuple[
        Dict[str, Union[str, int, float]],
        Dict[str, Union[str, int, float]],
        Dict[str, Union[str, int, float]],
        Dict[str, Union[str, int, float]]
        ]:
    basic_params = {
        'task': args.task,
        'proj_path': args.proj_path,
        'data_path': args.data_path,
        'logits_path': args.logits_path,
        'models_path': os.path.join(args.proj_path, 'models'),
        'random_seed': args.random_seed,
        'teacher_name': args.teacher_name
    }

    model_params = {
        'loss_alpha': args.loss_alpha,
        'T': args.T,
        'D': args.D,
        'std': args.std,
        'max_features': args.max_features,
        'hidden_dims': parse_hidden_dims(args.hidden_dims),
        'tokenizer_mode': args.tokenizer_mode
    }

    train_params = {
        'batch_size': args.batch_size,
        'eval_batch_size': args.eval_batch_size,
        'epochs': args.epochs
    }

    optimizer_params = {
        'learning_rate': args.learning_rate,
        'adam_epsilon': args.adam_epsilon
    }

    return basic_params, model_params, train_params, optimizer_params


def get_train_config(
    model_params: Dict[str, Union[str, int, float]],
    train_params: Dict[str, Union[str, int, float]]
    ) -> TrainProjectionNetConfig:
    train_config = TrainProjectionNetConfig(
        batch_size=train_params['batch_size'],
        eval_batch_size=train_params['eval_batch_size'],
        epochs=train_params['epochs'],
        T=model_params['T'],
        D=model_params['D'],
        std=model_params['std'],
        hidden_dims=model_params['hidden_dims']
    )
    return train_config


def get_optimizer_config(optimizer_params: Dict[str, float]):
    optimizer_config = TrainAdamOptimizerConfig(
        learning_rate=optimizer_params['learning_rate'],
        epsilon=optimizer_params['adam_epsilon']
    )
    return optimizer_config


def get_data_reader(
    basic_params: Dict[str, Union[str, int, float]],
    model_params: Dict[str, Union[str, int, float]],
    tokenizer: Tokenizer
    ) -> ProjectionDatasetReaderDistill:

    data_reader = ProjectionDatasetReaderDistill(
        data_path=basic_params['data_path'],
        task=basic_params['task'],
        max_features=model_params['max_features'],
        tokenizer=tokenizer,
        logits_filepath=basic_params['logits_path'],
        tokenizer_mode=model_params['tokenizer_mode'],
        is_pretrained_tokenizer=False
        )

    return data_reader


def get_datasets(
    data_reader: ProjectionDatasetReaderDistill,
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



def main():
    args = get_parser().parse_args()
    basic_params, model_params, train_params, optimizer_params = args_to_dicts(args=args)
    experiment.set_filename(__file__)
    experiment.log_parameters(basic_params)
    experiment.log_parameters(model_params)
    experiment.log_parameters(train_params)
    experiment.log_parameters(optimizer_params)
    experiment.add_tags([basic_params['task'], basic_params['teacher_name'], 'projectionnet'])

    set_seed(basic_params['random_seed'])

    train_config = get_train_config(model_params=model_params, train_params=train_params)
    optimizer_config = get_optimizer_config(optimizer_params=optimizer_params)

    tokenizer = Tokenizer(num_words=model_params['max_features'])
    data_reader = get_data_reader(basic_params=basic_params, model_params=model_params, tokenizer=tokenizer)

    train_examples, train_dataset, \
    valid_examples, valid_dataset, \
    test_examples, test_dataset, test_labels = get_datasets(data_reader=data_reader, train_config=train_config)

    print("Number of examples: ", train_examples, valid_examples, test_examples)

    model = ProjectionNetModelDistill(task=basic_params['task'],
                  train_config=train_config,
                  optimizer_config=optimizer_config,
                  alpha=model_params['loss_alpha']
                  )

    steps_per_epoch = train_examples // train_params['batch_size']
    model.fit_exp(
        train_dataset=train_dataset,
        validation_data=valid_dataset,
        steps_per_epoch=steps_per_epoch,
        experiment=experiment
        )

    model.evaluate_exp(
        test_dataset=test_dataset,
        test_labels=test_labels,
        experiment=experiment
        )
    
    print("\n\n")
    print(model._model.summary())

    
    model_path = os.path.join(basic_params['models_path'], 'projections_distill')

    if isinstance(model_params['hidden_dims'], list):
        phd = '_'.join(model_params['hidden_dims'])
    else:
        phd = 'None'

    model_name = f"projectionnet\
_{basic_params['task']}\
_{basic_params['teacher_name']}\
__T_{model_params['T']}_D_{model_params['D']}_std_{model_params['std']}\
_HD_{phd}\
__DT_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}"
    
    model_filepath = os.path.join(model_path,  model_name)
    model.save(model_filepath)
    experiment.log_model(model_name, model_filepath)



if __name__ == "__main__":
    main()
