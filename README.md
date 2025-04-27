# Python Code Generator - LLM Fine-tuning Project

This project fine-tunes a small language model (LLM) to generate Python code from natural language instructions. It uses the PEFT library with LoRA to efficiently fine-tune the model on an 8GB GPU.

## Features

- Fine-tunes a small LLM model (TinyLlama-1.1B) using the PEFT library with LoRA
- Training with the "iamtarun/python_code_instructions_18k_alpaca" dataset
- Optimized for an 8GB GPU with 4-bit quantization
- Configurable training parameters using Hydra and OmegaConf
- FastAPI backend for model serving
- Web UI for interacting with the model
- Comprehensive evaluation metrics

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU with at least 8GB VRAM
- Dependencies listed in requirements.txt

## Installation

```bash
# Clone the repository
git clone https://github.com/thealper2/tinyllama-python-code-generation.git
cd tinyllama-python-code-generation

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python train.py
```

### Starting the Web UI

```bash
python serve.py
```

### API Usage

You can also use the API directly:

```python
import requests

# Generate Python code
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "Write a function to calculate the Fibonacci sequence",
        "temperature": 0.7,
        "max_new_tokens": 512
    }
)

# Print the generated code
print(response.json()["code"])
```

## Project Structure

- **api/**: FastAPI implementation
- **config/**: Configuration files
- **data/**: Data processing utilities
- **models/**: Model and fine-tuning implementations
- **training/**: Training files
- **train.py**: Main training script
- **serve.py**: Web UI application

## Configuration

The project uses Hydra for configuration management. The main configuration files are:

- **config.yaml**: Main configuration file

You can override any configuration parameter using command line arguments or creating your own configuration files.

## Model

The default model is `TinyLlama-1.1B-Chat-v1.0`, a small but capable language model that can run on an 8GB GPU. The model is fine-tuned using `LoRA (Low-Rank Adaptation)` to efficiently adapt it to the Python code generation task.

## Dataset

The project uses the `"iamtarun/python_code_instructions_18k_alpaca"` dataset, which contains pairs of natural language instructions and corresponding Python code. The dataset is split into 90% training and 10% testing.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- HuggingFace for the transformers library
- PEFT library for efficient fine-tuning
- TinyLlama for the base model
- iamtarun for the Python code dataset