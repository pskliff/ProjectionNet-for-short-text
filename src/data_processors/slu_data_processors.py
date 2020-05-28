import os
from .base_data_processor import InputExample, DataProcessor
from typing import List, Dict, Tuple, Optional, Union



class SnipsProcessor(DataProcessor):
    """Processor for the SNIPS data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["text"].numpy().decode("utf-8"),
            None,
            tensor_dict["label"].numpy().decode("utf-8"),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic',
       'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(text_a=text_a, text_b=None, label=label))
        return examples



class AtisProcessor(DataProcessor):
    """Processor for the ATIS flight data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["text"].numpy().decode("utf-8"),
            None,
            tensor_dict["label"].numpy().decode("utf-8"),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ['atis_abbreviation', 'atis_aircraft',
       'atis_aircraft#atis_flight#atis_flight_no', 'atis_airfare',
       'atis_airline', 'atis_airline#atis_flight_no', 'atis_airport',
       'atis_capacity', 'atis_cheapest', 'atis_city', 'atis_distance',
       'atis_flight', 'atis_flight#atis_airfare', 'atis_flight_no',
       'atis_flight_time', 'atis_ground_fare', 'atis_ground_service',
       'atis_ground_service#atis_ground_fare', 'atis_meal',
       'atis_quantity', 'atis_restriction']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(text_a=text_a, text_b=None, label=label))
        return examples