from typing import Optional

import torch
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from models.lora import LoRAConfig


class CodeModel:
    """Handles loading and preparing the language model for training"""

    def __init__(self, config: DictConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.lora_config = None

    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer with quantization"""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.model.load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_name, use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name,
            quantization_config=bnb_config if self.config.model.load_in_4bit else None,
            device_map=self.config.model.device_map,
        )

        if self.config.model.use_peft:
            self._prepare_peft_model()

        return self.model, self.tokenizer

    def _prepare_peft_model(self):
        """Prepare the model for PEFT/LoRA training"""
        self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoRAConfig(**self.config.training.lora)
        peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            target_modules=lora_config.target_modules,
            bias=lora_config.bias,
            task_type=lora_config.task_type,
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def save_model(self, output_dir: str):
        """Save the trained model"""
        if self.model is not None:
            self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
