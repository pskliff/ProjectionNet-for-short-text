import os
import sys
import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple, Optional, Union, Any
from os.path import dirname, abspath


SRC_PATH = dirname(dirname(abspath(__file__)))
sys.path.append(SRC_PATH)

from data_processors.slu_data_to_bert_converter import slu_processors
from data_pipelines.create_tf_datasets_for_projection_base import ProjectionDatasetReaderBase


class ProjectionDatasetReader(ProjectionDatasetReaderBase):
    def __init__(
        self,
        data_path: str,
        task: str,
        max_features: int,
        tokenizer: Any,
        tokenizer_mode: Optional[str] = 'count',
        is_pretrained_tokenizer: Optional[bool] = False
        ):
        super().__init__(
            data_path,
            task,
            max_features,
            tokenizer,
            tokenizer_mode=tokenizer_mode,
            is_pretrained_tokenizer=is_pretrained_tokenizer
            )

    
    def _get_dataset(self,
        folder_name: str,
        batch_size: int,
        is_train: bool,
        shuffle: bool
        ) -> Tuple[int, tf.data.Dataset, Union[np.array, None]]:
        X, y = self._load_data(self._get_data_path(folder_name))
        
        if is_train and not self._is_pretrained_tokenizer:
            self._fit_tokenizer(X)
            self._is_pretrained_tokenizer = True
        
        X = self._apply_tokenizer(X)
        y = self._map_target(y)
        
        if isinstance(X, tuple):
            num_examples = len(X[0])
        else:
            num_examples = len(X)

        if is_train:
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
            if shuffle:
                dataset = dataset.shuffle(128)
            return num_examples, dataset.batch(batch_size), None
        else:
            dataset = tf.convert_to_tensor(X)
            return num_examples, dataset, y




class ProjectionDatasetReaderDistill(ProjectionDatasetReaderBase):
    def __init__(
        self,
        data_path: str,
        task: str,
        max_features: int,
        tokenizer: Any,
        logits_filepath: str,
        tokenizer_mode: Optional[str] = 'count',
        is_pretrained_tokenizer: Optional[bool] = False
        ):
        super().__init__(
            data_path,
            task,
            max_features,
            tokenizer,
            tokenizer_mode=tokenizer_mode,
            is_pretrained_tokenizer=is_pretrained_tokenizer
            )
        self._logits_filepath = logits_filepath

    
    def _get_dataset(self,
        folder_name: str,
        batch_size: int,
        is_train: bool,
        shuffle: bool
        ) -> Tuple[int, tf.data.Dataset, Union[np.array, None]]:
        X, y = self._load_data(self._get_data_path(folder_name))
        y_soft = np.load(self._logits_filepath)
        if is_train and not self._is_pretrained_tokenizer:
            self._fit_tokenizer(X)
            
        X = self._apply_tokenizer(X)
        y = self._map_target(y)
        
        if isinstance(X, tuple):
            num_examples = len(X[0])
        else:
            num_examples = len(X)

        if is_train:
            if not self._is_pretrained_tokenizer:
                dataset = tf.data.Dataset.from_tensor_slices((X, y, y_soft))
                self._is_pretrained_tokenizer = True
            else:
                dataset = tf.data.Dataset.from_tensor_slices((X, y))
            if shuffle:
                dataset = dataset.shuffle(128)
            return num_examples, dataset.batch(batch_size), None
        else:
            dataset = tf.convert_to_tensor(X)
            return num_examples, dataset, y
