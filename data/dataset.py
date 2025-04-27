from typing import List, Tuple

import datasets
from omegaconf import DictConfig
from transformers import AutoTokenizer

from data.schemas import CodeInstructionExample, DatasetConfig, SplitType


class CodeDatasetProcessor:
    """Handles loading and processing of the Python code instructions dataset"""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.dataset = None

    def load_dataset(self) -> datasets.DatasetDict:
        """Load and split the dataset according to configuration"""
        dataset = datasets.load_dataset(self.config.dataset_name)

        if self.config.max_samples is not None:
            # Apply selection to each split in the DatasetDict
            for split in dataset.keys():
                dataset[split] = dataset[split].select(
                    range(min(self.config.max_samples, len(dataset[split])))
                )

        # Split dataset into train and test
        split_dataset = dataset["train"].train_test_split(
            test_size=self.config.test_size, seed=self.config.random_seed
        )

        self.dataset = split_dataset
        return split_dataset

    def preprocess_dataset(self, tokenizer: AutoTokenizer) -> datasets.DatasetDict:
        """Tokenize and prepare the dataset for training"""
        if self.dataset is None:
            self.load_dataset()

        def tokenize_function(examples):
            """Tokenize the instruction-output pairs"""
            prompts = []
            for i in range(len(examples["instruction"])):
                example = CodeInstructionExample(
                    instruction=examples["instruction"][i],
                    input=examples["input"][i],
                    output=examples["output"][i],
                )
                prompts.append(example.to_prompt())

            return tokenizer(
                prompts,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )

        tokenized_datasets = self.dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.dataset["train"].column_names,
        )

        return tokenized_datasets

    def get_train_test_datasets(self) -> Tuple[datasets.Dataset, datasets.Dataset]:
        """Get the processed train and test datasets"""
        if self.dataset is None:
            self.load_dataset()
        return self.dataset["train"], self.dataset["test"]
