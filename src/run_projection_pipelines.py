# import comet_ml in the top of your file
from comet_ml import Experiment
    

experiment = Experiment(api_key="KEY",
                        project_name="PROJECT_NAME", workspace="WORKSPACE")


import os
import logging
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
import argparse
import datetime
from typing import List, Dict, Tuple, Optional, Union
from utils import set_seed
from pipelines import ProjectionPipeline, SGNNPipeline



logging.basicConfig(
    format = u'%(filename)s[LINE:%(lineno)d] # %(levelname)-8s [%(asctime)s]  %(message)s',
    level = logging.INFO
    )



def parse_hidden_dims(hidden_dims: str) -> Union[List[int], None]:
    if hidden_dims == 'None' or hidden_dims == '0':
        return None
    else:
        return [int(dim.strip()) for dim in hidden_dims.split(',')]


def parse_ngram_range(line: str) -> Tuple[int, int]:
    ngram_range = tuple([int(dim.strip()) for dim in line.split(',')])
    assert len(ngram_range) == 2, "Range must contain only 2 values divided by `,`, example: `1,2`"
    return ngram_range


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser(description='Runs ProjectionNetwork')

    # basic settings
    parser.add_argument('--proj_path', type=str, help='Project path')
    parser.add_argument('--data_path', type=str, help='Datasets path')
    parser.add_argument('--logits_path', type=str, help='Path to trainer logits')
    parser.add_argument('--random_seed', type=int, help='Random seed')
    parser.add_argument('--task', type=str, help='Project path') 
    parser.add_argument('--experiment_description', type=str, help='Additional tags to describe experiment')
    parser.add_argument('--is_distill', type=str2bool, help='Use distillation or not') 
    parser.add_argument('--student_name', type=str, help='Student model name')
    parser.add_argument('--teacher_name', type=str, help='Teacher model name')


    # Model settings
    parser.add_argument('--trainable_projection', type=str2bool, help='To make projection tensors trainable or not')
    parser.add_argument('--word_ngram_range', type=str, help='Ngram range for word level features')
    parser.add_argument('--char_ngram_range', type=str, help='Ngram range for character level features')
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
        'experiment_description': args.experiment_description,
        'is_distill': args.is_distill,
        'proj_path': args.proj_path,
        'data_path': args.data_path,
        'logits_path': args.logits_path,
        'models_path': os.path.join(args.proj_path, 'models'),
        'random_seed': args.random_seed,
        'teacher_name': args.teacher_name,
        'student_name': args.student_name
    }

    model_params = {
        'trainable_projection': args.trainable_projection,
        'word_ngram_range': parse_ngram_range(args.word_ngram_range),
        'char_ngram_range': parse_ngram_range(args.char_ngram_range),
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



def main():
    args = get_parser().parse_args()
    basic_params, model_params, train_params, optimizer_params = args_to_dicts(args=args)

    print("IS TRAINABLE = ", model_params['trainable_projection'], type(model_params['trainable_projection']))
    print("IS DISTILL = ", basic_params['is_distill'], type(basic_params['is_distill']))

    assert basic_params['student_name'] in ["SGNN", "SGNN++"], "Wrong student name"
    assert basic_params['teacher_name'] in ['bert', 'albert', 'fasttext', 'No_Teacher'], "wrong teacher name"

    if not basic_params['is_distill']:
        basic_params['teacher_name'] = 'No_Teacher'
        model_params['loss_alpha'] = 1


    experiment.set_filename(__file__)
    experiment.log_parameters(basic_params)
    experiment.log_parameters(model_params)
    experiment.log_parameters(train_params)
    experiment.log_parameters(optimizer_params)
    experiment.add_tags([
        basic_params['task'],
        basic_params['teacher_name'],
        basic_params['student_name'],
        basic_params['experiment_description']
        ])

    set_seed(basic_params['random_seed'])

    if basic_params['student_name'] == "SGNN":
        pipeline = ProjectionPipeline(
            basic_params=basic_params,
            model_params=model_params,
            train_params=train_params,
            optimizer_params=optimizer_params,
            is_distill=basic_params['is_distill'],
            experiment=experiment
            )
    elif basic_params['student_name'] == "SGNN++":
        pipeline = SGNNPipeline(
            basic_params=basic_params,
            model_params=model_params,
            train_params=train_params,
            optimizer_params=optimizer_params,
            is_distill=basic_params['is_distill'],
            experiment=experiment
            )

    teacher2params = {
        'bert': 110_000_000,
        'albert': 12_000_000,
        'fasttext': 4_000_000,
        'No_Teacher': 650_000
    }
    teacher2size = {
        'bert': 406*1024,
        'albert': 46*1024,
        'fasttext': 4500*1024,
        'No_Teacher': 2.7 * 1024
    }
    teacher2metric = {
        'snips': {
            'bert': 0.98,
            'albert': 0.984,
            'fasttext': 0.9786,
            'No_Teacher': 0.9557
        },
        'atis': {
            'bert': 0.979,
            'albert': 0.974,
            'fasttext': 0.96,
            'No_Teacher': 0.854
        }
    }

    pipeline.run(
        num_teacher_params=teacher2params[basic_params['teacher_name']],
        teacher_model_size_kb=teacher2size[basic_params['teacher_name']],
        teacher_accuracy=teacher2metric[basic_params['task']][basic_params['teacher_name']]
        )



if __name__ == "__main__":
    main()
