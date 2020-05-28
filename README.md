
# Comparative Analysis of Short Domain-specific Texts On-Device Classification Approaches Using Neural Projections

This repository provides code for experimental comparison of ProjectionNet modifications for short text classification 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Datasets
Comparison is provided using two datasets: [SNIPS](https://github.com/snipsco/snips-nlu) and [ATIS](https://github.com/howl-anderson/ATIS_dataset)
Train/Validation/Test sets are the same as in [Capsule-NLU](https://github.com/czhang99/Capsule-NLU)

## Training and Evaluation

Training and evaluation are represented as experiments, which are stored in [comet.ml](comet.ml)

To run experiment run this command:

```train and evaluation
python3 run_projection_pipelines.py --proj_path=<path_to_project>\
 --data_path=<path_to_data>\
 --random_seed=42\
 --task=<snips_or_atis>\
 --experiment_description=<experiment_special_tag>\
 --is_distill=<train_using_distillation_or_not_(true/false)>\
 --logits_path=<path_to_teacher_model_logits>\
 --student_name=<SGNN/SGNN++>\
 --teacher_name=<bert/albert/fasttext>\
 --trainable_projection=False\
 --word_ngram_range=1,2\
 --char_ngram_range=2,3\
 --loss_alpha=<distillation_tradeoff(float)>\
 --T=<number_of_projection_functions>\
 --D=<projection_space_dimension>\
 --std=1\
 --max_features=<input_dimension>\
 --hidden_dims=<number_of_units>\
 --tokenizer_mode='count'\
 --batch_size=100\
 --eval_batch_size=200\
 --epochs=120\
 --learning_rate=3e-5\
 --adam_epsilon=1e-08
```

## Pre-trained Models

You can download pretrained models here:

- [All models from experiments](https://drive.google.com/drive/folders/1oajiFdphlShF6O2Kpu_a7BCrlimC52Mh?usp=sharing)
