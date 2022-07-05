
from simpletransformers.language_modeling import LanguageModelingModel
import logging
import torch

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "train_batch_size": 16,
    "num_train_epochs": 3,
    "mlm": False,
}

# Use GPU
GPU = True
if GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(f"Using {device}")

cuda_available = torch.cuda.is_available()

model = LanguageModelingModel(model_type = 'gpt2', 
                            model_name = 'sberbank-ai/mGPT', 
                            args=train_args,
                            use_cuda=cuda_available)

model.train_model(train_file = "data/epzh-for-gpt.txt",
                output_dir = "rewriting/gpt2-ep")

# LOGS
# 55754: train mGPT w/ right input data format + reduced batch size = 16 (cuda out of memory)
