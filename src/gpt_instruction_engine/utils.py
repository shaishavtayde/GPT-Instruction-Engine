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
    """
    Returns a function to compute classification metrics (accuracy and F1-score).

    Args:
        num_labels (int): The number of labels in the classification task. Determines the type of averaging.

    Returns:
        function: A function to compute the metrics given logits and labels.
    """
    # Log the start of the metric loading process
    logger.info("Metrics been loaded")

    # Determine the type of averaging to use for metrics:
    # - 'binary' for binary classification tasks (num_labels == 2).
    # - 'macro' for multi-class classification tasks (num_labels > 2).
    if num_labels == 2:
        average = 'binary'
    else:
        average = 'macro'

    # Load the accuracy metric with the specified averaging method
    load_accuracy = evaluate.load('accuracy', average=average)
    # Load the F1 metric with the specified averaging method
    load_f1 = evaluate.load('f1', average=average)

    # Define the function to compute metrics using predictions and labels
    def this_compute_classifier_metrics(eval_pred):
        """
        Computes accuracy and F1-score for the given logits and labels.

        Args:
            eval_pred (tuple): A tuple containing:
                - logits (np.ndarray): The raw output predictions from the model.
                - labels (np.ndarray): The true labels for the dataset.

        Returns:
            dict: A dictionary containing the computed accuracy and F1-score.
        """
        logits, labels = eval_pred  # Unpack logits and labels
        # Get predicted labels by selecting the class with the highest logit value
        predictions = np.argmax(logits, axis=-1)
        # Compute accuracy metric
        accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        # Compute F1-score metric
        f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
        # Return the metrics as a dictionary
        return {"accuracy": accuracy, "f1": f1}

    # Log that the metrics function has been successfully loaded
    logger.info("Metrics are loaded")

    # Return the function that computes the metrics
    return this_compute_classifier_metrics


def compute_model_metrics(model, dataset, data_collator, batch_size, prefix="", compute_classifier_metrics_func=None):
    """
    Computes evaluation metrics (e.g., accuracy, F1-score, loss) for a model on a given dataset.

    Args:
        model: The PyTorch model to be evaluated.
        dataset: The dataset to evaluate the model on (e.g., validation or test dataset).
        data_collator: A function to collate and prepare batches of data.
        batch_size (int): The number of samples per batch during evaluation.
        prefix (str): A string to prefix metric names (e.g., "val_" or "test_").
        compute_classifier_metrics_func (function, optional): A function to compute classification metrics 
            (e.g., accuracy, F1-score). Defaults to a function based on the number of labels.

    Returns:
        dict: A dictionary containing the computed metrics with prefixed keys.
    """
    # Initialize containers for storing losses, logits (predictions), and labels
    all_losses = []
    all_logits, all_labels = [], []

    # Create a DataLoader to process the dataset in batches
    for batch in tqdm.tqdm(torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)):
        with torch.no_grad():  # Disable gradient computation for inference to save memory and speed up processing
            # Handle multi-GPU models (DataParallel) vs single-GPU models
            if type(model) is torch.nn.DataParallel:
                out = model(**{k: v.to(f'cuda:{model.device_ids[0]}') for k, v in batch.items()})
            else:
                out = model(**{k: v.to(model.device) for k, v in batch.items()})

        # Extract labels and remove them from the batch
        labels = batch.pop("labels")
        all_labels.append(labels)  # Collect true labels
        all_logits.append(out.logits)  # Collect logits (model predictions)
        # Store loss for each input in the batch, converted to CPU
        all_losses += [out.loss.cpu()] * len(batch['input_ids'])

    # If no custom metrics function is provided, use a default function based on the number of labels
    if compute_classifier_metrics_func is None:
        compute_classifier_metrics_func = get_compute_classifier_metrics(model.num_labels)

    # Compute metrics using the logits and labels
    metrics = compute_classifier_metrics_func(
        (torch.vstack(all_logits).cpu(), torch.hstack(all_labels).cpu())
    )
    # Compute and add the average loss to the metrics
    metrics['loss'] = float(torch.mean(torch.hstack(all_losses)))

    # Add prefix to all metric keys and return the metrics dictionary
    return {f'{prefix}{k}': v for (k, v) in metrics.items()}


def log_round_metrics(model, val_dataset, data_collator, inference_batch, round: int, selected_exes: List[int],
                      test_data=None, compute_classifier_metrics_func=None) -> Dict:
    """
    Logs evaluation metrics for a specific training round, including validation and optional test metrics.

    Args:
        model: The PyTorch model to evaluate.
        val_dataset: The validation dataset.
        data_collator: A function to collate data into batches for the model.
        inference_batch (int): Batch size for inference.
        round (int): The current training round (e.g., epoch or iteration number).
        selected_exes (List[int]): A list of selected examples for the current round.
        test_data (optional): The test dataset to evaluate on (default: None).
        compute_classifier_metrics_func (optional): A function to compute metrics like accuracy or F1-score.

    Returns:
        Dict: A dictionary containing computed metrics for the validation and (optionally) test datasets,
              along with the training round and selected examples.
    """
    # Set the model to evaluation mode
    model.eval()

    # Move the model to GPU if available
    if torch.cuda.is_available():
        if type(model) is not torch.nn.DataParallel and torch.cuda.device_count() > 1:
            # Use DataParallel if multiple GPUs are available
            model = torch.nn.DataParallel(model.to('cuda'))
        elif torch.cuda.device_count() == 1:
            # Move model to the single GPU
            model.to('cuda')

    # Log the start of validation metrics computation
    logger.info("Computing val metrics")
    # Compute validation metrics using the `compute_model_metrics` function
    metrics = compute_model_metrics(model, val_dataset, data_collator,
                                    inference_batch, prefix='val_', compute_classifier_metrics_func=compute_classifier_metrics_func)

    # Add the current training round to the metrics
    metrics['round'] = round

    # If test data is provided, compute and log test metrics
    if test_data is not None:
        logger.info("Computing test metrics")
        metrics.update(compute_model_metrics(model, test_data, data_collator,
                                             inference_batch, prefix='test_', compute_classifier_metrics_func=compute_classifier_metrics_func))

    # Log the selected examples for the round
    metrics['selected_ex'] = selected_exes

    # Move the model back to the CPU to free GPU memory
    model.to('cpu')

    # Return the computed metrics
    return metrics

    


