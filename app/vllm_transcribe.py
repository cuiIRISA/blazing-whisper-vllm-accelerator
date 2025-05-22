import time
import numpy as np
import librosa
import json
import tempfile
import os
from vllm.sampling_params import SamplingParams
import io
from vllm import LLM


async def _process_audio_data(llm: LLM, audio_data: np.ndarray, sample_rate: int) -> dict:
    """
    Internal function to process audio data into chunks and transcribe.
    """
    try:
        CHUNK_DURATION = 30
        samples_per_chunk = CHUNK_DURATION * sample_rate
        num_samples = len(audio_data)
        num_chunks = (num_samples + samples_per_chunk - 1) // samples_per_chunk

        audio_chunks = []
        for i in range(num_chunks):
            start = i * samples_per_chunk
            end = min((i + 1) * samples_per_chunk, num_samples)
            audio_chunks.append((audio_data[start:end], sample_rate))

        # Create prompts
        prompts = [{
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {"audio": chunk}
            },
            "decoder_prompt": "<|startoftranscript|>",
        } for chunk in audio_chunks]

        # Sampling params
        sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens=4096
        )

        # Generate transcription
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        duration = time.time() - start_time

        # Process outputs
        full_transcript = []
        for i, output in enumerate(outputs):
            if output.outputs and len(output.outputs) > 0:
                text = output.outputs[0].text.strip()
                full_transcript.append({
                    "chunk_index": i + 1,
                    "text": text
                })
            else:
                full_transcript.append({
                    "chunk_index": i + 1,
                    "text": "[ERROR: No transcription generated]"
                })

        return {
            "transcript": full_transcript,
            "processing_time_seconds": round(duration, 2),
            "chunks_processed": num_chunks,
            "transcription_speed_chunks_per_second": round(num_chunks / duration, 2) if duration > 0 else 0
        }

    except Exception as e:
        raise RuntimeError(f"Audio processing failed: {str(e)}")


async def transcribe_audio_bytes(llm, audio_bytes: bytes):
    """
    Transcribes raw audio bytes using _process_audio_data.
    """
    try:
        audio_file = io.BytesIO(audio_bytes)
        audio_data, sample_rate = librosa.load(audio_file, sr=None)

        result = await _process_audio_data(llm, audio_data, sample_rate)
        return result

    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}")