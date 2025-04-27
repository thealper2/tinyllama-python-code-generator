import os
from typing import Tuple

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from transformers import (
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from trl import SFTTrainer

from data.dataset import CodeDatasetProcessor
from models.lora import TrainingConfig
from models.model import CodeModel


class CodeModelTrainer:
    """Handles the training process for the code generation model"""

    def __init__(self, config: DictConfig):
        self.config = config
        self.model_handler = CodeModel(config)
        self.dataset_processor = CodeDatasetProcessor(instantiate(config.dataset))
        self.trainer = None

    def setup_training(self) -> Tuple:
        """Setup model, tokenizer, and datasets for training"""
        model, tokenizer = self.model_handler.load_model_and_tokenizer()
        tokenized_datasets = self.dataset_processor.preprocess_dataset(tokenizer)

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # Convert Hydra config to TrainingArguments
        training_config = TrainingConfig(
            output_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            num_train_epochs=self.config.training.num_train_epochs,
            learning_rate=self.config.training.learning_rate,
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            eval_steps=self.config.training.eval_steps,
            max_seq_length=self.config.dataset.max_seq_length,  # Changed from training to dataset
            optim=self.config.training.optim,
            save_total_limit=self.config.training.save_total_limit,
            fp16=self.config.training.fp16,
            report_to=self.config.training.report_to,
        )
        training_args = training_config.to_training_arguments()

        # Initialize SFTTrainer
        self.trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            max_seq_length=self.config.dataset.max_seq_length,
            dataset_text_field="text",  # This will be added by our tokenize_function
        )

        return model, tokenizer, tokenized_datasets

    def train(self):
        """Run the training process"""
        if self.trainer is None:
            self.setup_training()

        # Start training
        self.trainer.train()

        # Save the best model
        output_dir = "final_model"
        self.model_handler.save_model(output_dir)

        return output_dir

    def evaluate(self):
        """Evaluate the model on test set"""
        if self.trainer is None:
            self.setup_training()

        eval_results = self.trainer.evaluate()
        return eval_results
