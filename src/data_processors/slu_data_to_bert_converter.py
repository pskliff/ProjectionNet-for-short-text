import logging
import tensorflow as tf
from .slu_dataset_info_dicts import slu_output_modes, slu_processors, slu_tasks_num_labels
from .base_data_processor import InputExample, InputFeatures
from typing import List, Dict, Tuple, Optional, Union
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


def _slu_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    if task is not None:
        processor = slu_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = slu_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}
    def label_from_example(example: InputExample) -> Union[int, float]:
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        # logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


def _tf_slu_convert_examples_to_features(
        examples: tf.data.Dataset,
        tokenizer: PreTrainedTokenizer,
        task=str,
        max_length: Optional[int] = None,
        is_train: Optional[bool] = True
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Returns:
            A ``tf.data.Dataset`` containing the task-specific features.

        """
        processor = slu_processors[task]()
        examples = [processor.get_example_from_tensor_dict(example) for example in examples]
        features = _slu_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)

        if is_train:
            def gen():
                for ex in features:
                    yield (
                        {
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "token_type_ids": ex.token_type_ids,
                        },
                        ex.label,
                    )

            return (tf.data.Dataset.from_generator(
                        gen,
                        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
                        (
                            {
                                "input_ids": tf.TensorShape([None]),
                                "attention_mask": tf.TensorShape([None]),
                                "token_type_ids": tf.TensorShape([None]),
                            },
                            tf.TensorShape([]),
                        ),
                    ),
                    None
            )
        else:
            def gen_examples():
                for ex in features:
                    yield {
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "token_type_ids": ex.token_type_ids,
                        }

            def gen_labels():
                for ex in features:
                    yield ex.label

            return (tf.data.Dataset.from_generator(
                            gen_examples,
                            {"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32},
                                {
                                    "input_ids": tf.TensorShape([None]),
                                    "attention_mask": tf.TensorShape([None]),
                                    "token_type_ids": tf.TensorShape([None]),
                                },
                        ),
                        tf.data.Dataset.from_generator(
                            gen_labels,
                            tf.int64,
                            tf.TensorShape([])
                        )
                )

def slu_convert_examples_to_features(
    examples: Union[List[InputExample], "tf.data.Dataset"],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
    is_train=True
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    return _tf_slu_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task, is_train=is_train)
    