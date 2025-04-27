import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.schemas import CodeResponse, HealthCheck
from api.services import CodeGenerationService

app = FastAPI(
    title="Python Code Generator API",
    description="API for generating Python code from natural language instructions",
    version="0.1.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the service
MODEL_PATH = "final_model"
service = CodeGenerationService(MODEL_PATH)


class GenerationRequest(BaseModel):
    """Request model for code generation"""

    prompt: str
    temperature: float = 0.7
    max_new_tokens: int = 512
    top_p: float = 0.9
    top_k: int = 50
    num_return_sequences: int = 1
    num_beams: int = 1
    do_sample: bool = True


@app.on_event("startup")
async def startup_event():
    """Load the model when the application starts"""
    try:
        service.load_model()
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    model_info = service.get_model_info()
    return {
        "status": "healthy",
        "model": model_info["model"],
        "device": model_info["device"],
    }


@app.post("/generate", response_model=CodeResponse)
async def generate_code(request: GenerationRequest):
    """Generate Python code from natural language instruction"""
    try:
        start_time = time.time()

        generated_code = service.generate_code(
            instruction=request.prompt,
            parameters={
                "temperature": request.temperature,
                "max_length": request.max_new_tokens,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "num_return_sequences": request.num_return_sequences,
                "do_sample": request.do_sample,
            },
        )

        processing_time = time.time() - start_time

        return {
            "code": generated_code,
            "processing_time": processing_time,
            "status": "success",
            "details": {
                "model": service.get_model_info(),
                "parameters": request.dict(),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")
