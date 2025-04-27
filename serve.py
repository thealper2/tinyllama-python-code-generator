#!/usr/bin/env python3
import argparse

import uvicorn

from api.main import app


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Code Generator API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument(
        "--model-path", type=str, default="final_model", help="Path to trained model"
    )

    args = parser.parse_args()

    # Set environment variable for model path
    import os

    os.environ["MODEL_PATH"] = args.model_path

    run_server(host=args.host, port=args.port)
