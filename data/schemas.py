from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pydantic import BaseModel, validator


class SplitType(str, Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


class DatasetConfig(BaseModel):
    """Configuration for dataset loading and processing"""

    dataset_name: str
    test_size: float
    random_seed: int
    max_samples: Optional[int]

    @validator("test_size")
    def validate_test_size(cls, v):
        if not 0 < v < 1:
            raise ValueError("test_size must be between 0 and 1")
        return v


@dataclass
class CodeInstructionExample:
    """Single example from the Python code instructions dataset"""

    instruction: str
    input: str
    output: str

    def to_prompt(self) -> str:
        """Convert example to prompt format for model training"""
        if self.input:
            return f"### Instruction:\n{self.instruction}\n\n### Input:\n{self.input}\n\n### Output:\n{self.output}"
        return f"### Instruction:\n{self.instruction}\n\n### Output:\n{self.output}"
