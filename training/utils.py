from typing import Any, Dict

import torch


def print_gpu_utilization():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.1f}GB")
    else:
        print("CUDA is not available")


def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
