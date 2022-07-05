
from simpletransformers.language_modeling import LanguageModelingModel
import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "train_batch_size": 64,
    "num_train_epochs": 3,
    "mlm": False,
}

model = LanguageModelingModel(model_type = 'gpt2', 
                            model_name = 'sberbank-ai/mGPT', 
                            args=train_args)

model.train_model(train_file = "data/epzh-for-gpt.txt",
                output_dir = "rewriting/gpt2-ep")
