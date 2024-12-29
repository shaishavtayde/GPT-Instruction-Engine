from typing import List, Dict
import logging
import os
import math
import torch.backends
import torch.backends.cudnn
import tqdm

import numpy as np
import evaluate
import torch
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from sklearn.utils.class_weight import compute_class_weight
from transformers.modeling_utils import unwrap_model

logger = logging.getLogger(__name__)

import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed) # Pyhon random module seed
    np.random.seed(seed) # Numpy seed for generating random numbers
    torch.manual_seed(seed) # PyTorch seed for CPU
    torch.cuda.manual_seed(seed) # PyTorch seed for CUDA
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior (note: can slow down performance)
    torch.backends.cudnn.benchmark = False  # Disable the benchmark for deterministic results

def get_compute_classifier_metrics(num_labels):
    logger.info("Metrics been loaded")
    if num_labels == 2:
        average = 'binary'
    else:
        average = 'macro'

    load_accuracy = evaluate.load('accuracy', average=average)
    load_f1 = evaluate.load('f1', average=average)

    def this_compute_classifier_metrics(eval_pred):
        logits, labels = eval_pred 
        predictions = np.argmax(logits, axis=-1)
        accuracy = load_accuracy.compute(predictions, references=labels)["accuracy"] 
        f1 = load_f1.compute(predictions, references=labels)["f1"]
        return {"accuracy": accuracy, "f1": f1}
    logger.info("Metrics are loaded")
    return this_compute_classifier_metrics



