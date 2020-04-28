

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




def main():
    args = get_parser().parse_args()
    basic_params, model_params, train_params, optimizer_params = args_to_dicts(args=args)


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
__DT_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.h5"

    model_filepath = os.path.join(model_path,  model_name)
    print("MODEL_FILEPATH:\n", model_filepath)

    print()
    print(basic_params)
    print(model_params)
    print(train_params)
    print(optimizer_params)



if __name__ == "__main__":
    main()



# /anaconda3/bin/python test.py --proj_path='./'\
#  --data_path='../data/'\
#  --random_seed=42\
#  --task='snips'\
#  --logits_path='./logits'\
#  --teacher_name='bert'\
#  --loss_alpha=0.5\
#  --T=10\
#  --D=12\
#  --std=1\
#  --max_features=1000\
#  --hidden_dims=0\
#  --tokenizer_mode='count'\
#  --batch_size=256\
#  --eval_batch_size=512\
#  --epochs=20\
#  --learning_rate=0.0001\
#  --adam_epsilon=0.00000001