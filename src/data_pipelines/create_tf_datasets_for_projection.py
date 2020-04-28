import os
import sys
import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple, Optional, Union
from os.path import dirname, abspath

from tensorflow.keras.preprocessing.text import Tokenizer


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
        tokenizer: Tokenizer,
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



class ProjectionDatasetReader(ProjectionDatasetReaderBase):
    def __init__(
        self,
        data_path: str,
        task: str,
        max_features: int,
        tokenizer: Tokenizer,
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
        
        num_examples = len(X)

        if is_train:
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
            if shuffle:
                dataset = dataset.shuffle(128)
            return num_examples, dataset.batch(batch_size), None
        else:
            dataset = tf.data.Dataset.from_tensor_slices(X)
            return num_examples, dataset, y




class ProjectionDatasetReaderDistill(ProjectionDatasetReaderBase):
    def __init__(
        self,
        data_path: str,
        task: str,
        max_features: int,
        tokenizer: Tokenizer,
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
            dataset = tf.data.Dataset.from_tensor_slices(X)
            return num_examples, dataset, y





# class ProjectionDatasetReader(object):
#     def __init__(
#         self,
#         data_path: str,
#         task: str,
#         max_features: int,
#         tokenizer: Tokenizer,
#         tokenizer_mode: Optional[str] = 'count',
#         is_pretrained_tokenizer: Optional[bool] = False
#         ):
#         super().__init__()

#         self._data = dict()
#         self._data_path = data_path
#         self._task = task
#         self._max_feature = max_features
#         self._tokenizer = tokenizer
#         self._tokenizer_mode = tokenizer_mode
#         label_list = slu_processors[task]().get_labels()
#         self.label_map = {label: i for i, label in enumerate(label_list)}
#         self._is_pretrained_tokenizer = is_pretrained_tokenizer

    
#     def _get_data_path(
#         self,
#         folder_name: str
#         ) -> str:
#         return os.path.join(self._data_path, self._task, folder_name)


#     def _load_data(
#         self,
#         path: str
#         ) -> Tuple[List[str], List[str]]:
#         return get_data_from_files(path)


#     def _fit_tokenizer(self, texts: List[str]):
#         self._tokenizer.fit_on_texts(texts)


#     def _apply_tokenizer(self, texts: List[str]):
#         return self._tokenizer.texts_to_matrix(texts, mode=self._tokenizer_mode)


#     def _map_target(self, y: List[str]) -> np.array:
#         label_map_func = lambda label: self.label_map[label]
#         return np.array(list(map(label_map_func, y)))
    

#     def _get_dataset(self,
#         folder_name: str,
#         batch_size: int,
#         is_train: bool,
#         shuffle: bool
#         ) -> Tuple[int, tf.data.Dataset, Union[np.array, None]]:
#         X, y = self._load_data(self._get_data_path(folder_name))
        
#         if is_train and not self._is_pretrained_tokenizer:
#             self._fit_tokenizer(X)
#             self._is_pretrained_tokenizer = True
        
#         X = self._apply_tokenizer(X)
#         y = self._map_target(y)
        
#         num_examples = len(X)

#         if is_train:
#             dataset = tf.data.Dataset.from_tensor_slices((X, y))
#             if shuffle:
#                 dataset = dataset.shuffle(128)
#             return num_examples, dataset.batch(batch_size), None
#         else:
#             dataset = tf.data.Dataset.from_tensor_slices(X)
#             return num_examples, dataset, y
            

#     def get_train_dataset(self,
#         folder_name: str,
#         batch_size: int,
#         shuffle: Optional[bool] = False
#         ) -> Tuple[int, tf.data.Dataset]:
#         num_examples, dataset, _ =  self._get_dataset(folder_name, batch_size, is_train=True, shuffle=shuffle)
#         return num_examples, dataset


#     def get_valid_dataset(self,
#         folder_name: str,
#         batch_size: int,
#         ) -> Tuple[int, tf.data.Dataset]:
#         num_examples, dataset, _ =  self._get_dataset(folder_name, batch_size, is_train=True, shuffle=False)
#         return num_examples, dataset

    
#     def get_test_dataset(self,
#         folder_name: str,
#         batch_size: int
#         ) -> Tuple[int, tf.data.Dataset, np.array]:
#         return self._get_dataset(folder_name, batch_size, is_train=False, shuffle=False)
