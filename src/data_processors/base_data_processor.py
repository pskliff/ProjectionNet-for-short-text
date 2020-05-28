import os
import tensorflow as tf
import json
import copy
import csv
import dataclasses
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import logging
from tensorflow.keras.preprocessing.text import Tokenizer



def get_data_from_files(path: str) -> Tuple[List[str], List[str]]:
    with open(os.path.join(path, "seq.in")) as f:
        sentences = [line.strip() for line in f.readlines()]
    with open(os.path.join(path, "label")) as f:
        labels = [line.strip() for line in f.readlines()]
        
    return sentences, labels


def get_tensorflow_dataset(path: str, verbose: bool = False):
    def data_generator():
        with open(os.path.join(path, "seq.in")) as f:
            sentences = f.readlines()
        with open(os.path.join(path, "label")) as f:
            labels = f.readlines()
        
        for sentence, label in zip(sentences, labels):
            yield {"text": sentence.strip(), "label": label.strip()}

    dataset = tf.data.Dataset.from_generator( 
      data_generator, 
     {"text": tf.string, "label": tf.string}, 
     {"text": tf.TensorShape([]), "label": tf.TensorShape([])}) 
    
    if verbose:
       logging.info("Dataset Length = ", len(list(dataset.as_numpy_iterator())))
    return dataset



@dataclass(frozen=False)
class InputExample:
    """
    A single training/test example for simple sequence classification.
    Args:
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"



class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True) + "\n"



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def tfds_map(self, example):
        """Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are.
        This method converts examples to the correct format."""
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))