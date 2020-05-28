import os
import sys
import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple, Optional, Union, Any
from os.path import dirname, abspath

SRC_PATH = dirname(dirname(abspath(__file__)))
sys.path.append(SRC_PATH)

from data_processors.base_data_processor import get_data_from_files
from data_processors.slu_data_to_bert_converter import slu_processors



class ProjectionDatasetReaderBase(object):
    def __init__(
        self,
        data_path: str,
        task: str,
        max_features: int,
        tokenizer: Any,
        tokenizer_mode: Optional[str] = 'count',
        is_pretrained_tokenizer: Optional[bool] = False
        ):
        super().__init__()

        self._data = dict()
        self._data_path = data_path
        self._task = task
        self._max_feature = max_features
        self._tokenizer = tokenizer
        self._tokenizer_mode = tokenizer_mode
        label_list = slu_processors[task]().get_labels()
        self.label_map = {label: i for i, label in enumerate(label_list)}
        self._is_pretrained_tokenizer = is_pretrained_tokenizer

    
    def _get_data_path(
        self,
        folder_name: str
        ) -> str:
        return os.path.join(self._data_path, self._task, folder_name)


    def _load_data(
        self,
        path: str
        ) -> Tuple[List[str], List[str]]:
        return get_data_from_files(path)


    def _fit_tokenizer(self, texts: List[str]):
        self._tokenizer.fit_on_texts(texts)


    def _apply_tokenizer(self, texts: List[str]):
        return self._tokenizer.texts_to_matrix(texts, mode=self._tokenizer_mode)


    def _map_target(self, y: List[str]) -> np.array:
        label_map_func = lambda label: self.label_map[label]
        return np.array(list(map(label_map_func, y)))
    

    def _get_dataset(self,
        folder_name: str,
        batch_size: int,
        is_train: bool,
        shuffle: bool
        ) -> Tuple[int, tf.data.Dataset, Union[np.array, None]]:
        raise NotImplementedError("You have to implement `get_dataset` procedure")
            

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
        num_examples, dataset, _ =  self._get_dataset(folder_name, batch_size, is_train=True, shuffle=False)
        return num_examples, dataset

    
    def get_test_dataset(self,
        folder_name: str,
        batch_size: int
        ) -> Tuple[int, tf.data.Dataset, np.array]:
        return self._get_dataset(folder_name, batch_size, is_train=False, shuffle=False)
