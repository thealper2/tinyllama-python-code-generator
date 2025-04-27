import time
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

from api.schemas import GenerationParameters
from data.schemas import CodeInstructionExample


class CodeGenerationService:
    """Service for generating Python code from natural language instructions"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.generator = None

    def load_model(self):
        """Load the trained model and tokenizer"""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, device_map="auto", torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize pipeline without device parameter
        self.generator = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer
        )

    def generate_code(
        self,
        instruction: str,
        input_text: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate Python code from instruction with customizable parameters

        Args:
            instruction: Natural language instruction
            input_text: Optional additional context
            parameters: Dictionary of generation parameters including:
                - temperature
                - max_length
                - top_p
                - top_k
                - num_return_sequences
                - do_sample

        Returns:
            Generated Python code
        """
        if self.generator is None:
            self.load_model()

        # Set default parameters if not provided
        if parameters is None:
            parameters = {
                "temperature": 0.7,
                "max_length": 512,
                "top_p": 0.9,
                "top_k": 50,
                "num_return_sequences": 1,
                "do_sample": True,
            }

        # Format the prompt
        example = CodeInstructionExample(
            instruction=instruction, input=input_text or "", output=""
        )
        prompt = example.to_prompt()

        # Create generation config
        generation_config = GenerationConfig(
            temperature=parameters.get("temperature", 0.7),
            max_new_tokens=parameters.get("max_length", 512),
            top_p=parameters.get("top_p", 0.9),
            top_k=parameters.get("top_k", 50),
            num_return_sequences=parameters.get("num_return_sequences", 1),
            do_sample=parameters.get("do_sample", True),
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Generate code
        generated = self.generator(prompt, generation_config=generation_config)

        # Extract the generated code
        generated_text = generated[0]["generated_text"]
        output_prefix = (
            "### Output:" if "### Output:" in generated_text else "### Output:\n"
        )
        generated_code = generated_text.split(output_prefix)[-1].strip()

        return generated_code

    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        device = next(self.model.parameters()).device if self.model else "not loaded"
        return {
            "model": self.model_path,
            "device": str(device),
            "status": "loaded" if self.model is not None else "not loaded",
        }
