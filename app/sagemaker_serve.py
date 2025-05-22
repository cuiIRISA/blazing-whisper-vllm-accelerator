from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import os
import logging
import boto3
from vllm import LLM
import json
from fastapi import Request

# Import shared transcription function
from app.vllm_transcribe import transcribe_audio_bytes

app = FastAPI()
logger = logging.getLogger(__name__)

# Global variables (will be initialized per-worker)
llm = None
s3_client = None

@app.on_event("startup")
async def load_model():
    """Load the Whisper model once per Gunicorn worker"""
    global llm, s3_client
    
    try:
        # Initialize LLM per worker
        llm = LLM(
            model="openai/whisper-large-v3-turbo",
            max_model_len=448,
            max_num_seqs=200,  # Reduce if OOM
            limit_mm_per_prompt={"audio": 1},
            kv_cache_dtype="fp8",  # Use "auto" if fp8 not supported
            tensor_parallel_size=1  # Must match GPU count per worker
        )
        
        # Initialize S3 client per worker
        s3_client = boto3.client('s3')
        logger.info("Model and S3 client loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError("Model initialization failed")

@app.get("/ping")
async def ping():
    """Health check endpoint required by SageMaker"""
    return {"status": "healthy"}

@app.post("/invocations")
async def invocations(request: Request):
    global llm, s3_client
    
    try:
        audio_bytes = await request.body()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty request body")

        # Process the audio file from S3
        result = await transcribe_audio_bytes(llm, audio_bytes)

        return result
    
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))