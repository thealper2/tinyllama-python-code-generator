from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class GenerationParameters(BaseModel):
    """Parameters for controlling text generation"""

    temperature: float = Field(0.7, ge=0.1, le=1.5)
    max_length: int = Field(512, ge=64, le=1024)
    top_p: float = Field(0.9, ge=0.1, le=1.0)
    top_k: int = Field(50, ge=1, le=100)
    num_return_sequences: int = Field(1, ge=1, le=5)
    do_sample: bool = True


class CodeResponse(BaseModel):
    """Response containing generated Python code"""

    code: str
    processing_time: float
    status: str
    details: Optional[Dict[str, Any]] = None


class HealthCheck(BaseModel):
    """Health check response"""

    status: str
    model: str
    device: str
