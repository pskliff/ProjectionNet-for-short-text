import os
import sys
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from os.path import dirname, abspath
import tensorflow as tf
from transformers import PreTrainedTokenizer

SRC_PATH = dirname(dirname(abspath(__file__)))
sys.path.append(SRC_PATH)

from data_processors.base_data_processor import get_tensorflow_dataset
from settings import DataSettings, TaskSettings, BertTrainSettings
from data_processors.slu_data_to_bert_converter import slu_convert_examples_to_features
import logging


class BertDatasetReader(object):
    def __init__(self,
        data_path: str,
        task: str,
        seq_len: int,
        tokenizer: PreTrainedTokenizer):
        self._data = dict()
        self._data_path = data_path
        self._task = task
        self._seq_len = seq_len
        self._tokenizer = tokenizer


    def _get_data_path(self, folder_name: str) -> str:
        return os.path.join(self._data_path, self._task, folder_name)


    def _load_data(self, path: str) -> tf.data.Dataset:
        return get_tensorflow_dataset(path)


    def _get_dataset(self,
        folder_name: str,
        batch_size: int,
        is_train: bool,
        shuffle: bool
        ) -> Tuple[int, tf.data.Dataset, Union[np.array, None]]:
        dataset = self._load_data(self._get_data_path(folder_name))
        dataset, labels = slu_convert_examples_to_features(dataset,
                    self._tokenizer,
                    max_length=self._seq_len,
                    task=self._task,
                    is_train=is_train)
        
        num_examples = len(list(dataset.as_numpy_iterator()))

        if shuffle:
            dataset = dataset.shuffle(128)
        if is_train:
            return num_examples, dataset.batch(batch_size).repeat(-1), None
        else:
            return num_examples, dataset, np.array(list(labels.as_numpy_iterator()))


    def get_train_dataset(self,
        folder_name: str,
        batch_size: int,
        shuffle: Optional[bool] = False
        ) -> Tuple[int, tf.data.Dataset]:
        num_examples, dataset, _ =  self._get_dataset(folder_name, batch_size, is_train=True, shuffle=shuffle)
        return num_examples, dataset

    
    def get_valid_dataset(self,
        folder_name: str,
        batch_size: int,
        ) -> Tuple[int, tf.data.Dataset]:
        num_examples, dataset, _ =  self._get_dataset(folder_name, batch_size,is_train=True, shuffle=False)
        return num_examples, dataset


    def get_test_dataset(self,
        folder_name: str,
        batch_size: int) -> Tuple[int, tf.data.Dataset, tf.data.Dataset]:
        return self._get_dataset(folder_name, batch_size, is_train=False, shuffle=False)


