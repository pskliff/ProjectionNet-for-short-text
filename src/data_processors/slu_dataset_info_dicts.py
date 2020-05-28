from .slu_data_processors import SnipsProcessor, AtisProcessor


slu_tasks_num_labels = {
    "snips": 7,
    "atis": 21
}

slu_processors = {
    "snips": SnipsProcessor,
    "atis": AtisProcessor
}

slu_output_modes = {
    "snips": "classification",
    "atis": "classification"
}