from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel, validator
from transformers import TrainingArguments


class LoRAConfig(BaseModel):
    """Configuration for LoRA (Low-Rank Adaptation)"""

    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    @validator("bias")
    def validate_bias(cls, v):
        if v not in ["none", "all", "lora_only"]:
            raise ValueError("bias must be one of 'none', 'all', or 'lora_only'")
        return v


class TrainingConfig(BaseModel):
    """Configuration for model training"""

    output_dir: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    learning_rate: float
    logging_steps: int
    save_steps: int
    eval_steps: int
    max_seq_length: int
    optim: str = "paged_adamw_32bit"
    save_total_limit: int = 3
    fp16: bool = True
    report_to: str = "none"

    def to_training_arguments(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments"""
        return TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.num_train_epochs,
            learning_rate=self.learning_rate,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            max_steps=-1,
            optim=self.optim,
            save_total_limit=self.save_total_limit,
            fp16=self.fp16,
            report_to=self.report_to,
        )
