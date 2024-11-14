#!/usr/bin/env python3
import argparse
import io
import json
import logging
import wave
from pathlib import Path
from typing import Any, Dict
import subprocess
import uuid
import os

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import FileResponse
import uvicorn
from starlette.background import BackgroundTask

from . import PiperVoice
from .download import ensure_voice_exists, find_voice, get_voices

_LOGGER = logging.getLogger()

# Global dictionary to store loaded models
loaded_models = {}

app = FastAPI(title="Piper Voice API", description="API for synthesizing speech with Piper Voice models.", version="1.0.0")
security = HTTPBasic()

# Authentication credentials
USERS = {
    "piper": "digital-human"
}

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_password = USERS.get(credentials.username)
    if not correct_password or correct_password != credentials.password:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    return credentials.username

@app.get("/docs", dependencies=[Depends(authenticate)])
async def get_docs():
    """Protected API documentation"""
    pass

class SynthesizeModel(BaseModel):
    text: str
    model_name: str
    speaker: int = 0
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w: float = 0.8
    sentence_silence: float = 0.0

@app.get("/models")
async def list_models():
    """List available models"""
    models_path = Path("/opt/digital_human/piper/app/models")
    model_files = [f.stem for f in models_path.glob("*.onnx")]
    return {"available_models": model_files}


import subprocess
import uuid
import os
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

@app.post("/synthesize")
async def synthesize(data: SynthesizeModel):
    """Synthesize speech"""
    text = data.text.strip()
    model_name = data.model_name.strip() + ".onnx"

    if not text:
        _LOGGER.error("No text provided for synthesis.")
        raise HTTPException(status_code=400, detail="No text provided")
    if not model_name:
        _LOGGER.error("No model name provided for synthesis.")
        raise HTTPException(status_code=400, detail="No model name provided")

    _LOGGER.info("Starting synthesis for text: '%s' with model: '%s'", text, model_name)

    # Load model if not already loaded
    if model_name not in loaded_models:
        _LOGGER.info("Loading model: %s", model_name)
        model_path = Path(f"/opt/digital_human/piper/app/models/{model_name}")
        if not model_path.exists():
            _LOGGER.info("Model path does not exist. Attempting to download model: %s", model_name)
            voices_info = get_voices(args.download_dir, update_voices=args.update_voices)

            aliases_info: Dict[str, Any] = {}
            for voice_info in voices_info.values():
                for voice_alias in voice_info.get("aliases", []):
                    aliases_info[voice_alias] = {"_is_alias": True, **voice_info}

            voices_info.update(aliases_info)
            ensure_voice_exists(model_name, args.data_dir, args.download_dir, voices_info)
            model_path, config_path = find_voice(model_name, args.data_dir)
        else:
            config_path = None

        _LOGGER.info("Successfully loaded model: %s", model_name)
        loaded_models[model_name] = PiperVoice.load(model_path, config_path=config_path, use_cuda=args.cuda)

    voice = loaded_models[model_name]

    synthesize_args = {
        "speaker_id": data.speaker,
        "length_scale": data.length_scale,
        "noise_scale": data.noise_scale,
        "noise_w": data.noise_w,
        "sentence_silence": data.sentence_silence,
    }

    unique_id = uuid.uuid4().hex
    output_path = Path(f"/tmp/synthesized_{unique_id}.wav")
    output_8khz_path = Path(f"/tmp/synthesized_8khz_{unique_id}.wav")
    try:
        import time
        start_time = time.time()
        _LOGGER.info("Synthesizing speech to temporary file: %s", output_path)
        with wave.open(str(output_path), "wb") as wav_file:
            voice.synthesize(text, wav_file, **synthesize_args)
        _LOGGER.info("Successfully synthesized speech to: %s", output_path)
        
        # Convert the synthesized wav file to 8kHz
        _LOGGER.info("Converting synthesized file to 8kHz: %s", output_8khz_path)
        subprocess.run([
            "ffmpeg", "-i", str(output_path), "-ar", "8000", "-y", str(output_8khz_path)
        ], check=True)
        _LOGGER.info("Successfully converted file to 8kHz: %s", output_8khz_path)

        end_time = time.time()
        _LOGGER.info("Synthesized text: '%s' in %.2f seconds", text, end_time - start_time)

        # Serve the file response before deleting temporary files
        response = FileResponse(
            output_8khz_path,
            media_type="audio/wav",
            filename="synthesized_8khz.wav",
            background=BackgroundTask(lambda: cleanup_temp_files([output_path, output_8khz_path]))
        )
        _LOGGER.info("Serving synthesized file: %s", output_8khz_path)
        return response
    except subprocess.CalledProcessError as e:
        _LOGGER.error("Error during conversion with ffmpeg: %s", e)
        raise HTTPException(status_code=500, detail="Error during conversion to 8kHz")
    except Exception as e:
        _LOGGER.error("Error during synthesis: %s", e)
        raise HTTPException(status_code=500, detail="Error during synthesis")

def cleanup_temp_files(file_paths):
    for file_path in file_paths:
        if file_path.exists():
            _LOGGER.info("Removing temporary file: %s", file_path)
            os.remove(file_path)




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="HTTP server host")
    parser.add_argument("--port", type=int, default=5000, help="HTTP server port")
    parser.add_argument("--cuda", action="store_true", help="Use GPU")
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        action="append",
        default=["/opt/digital_human/piper/app/models"], help="Data directory to check for downloaded models (default: /opt/digital_human/piper/app/models)",
    )
    parser.add_argument(
        "--download-dir",
        "--download_dir",
        help="Directory to download voices into (default: first data dir)",
    )
    parser.add_argument(
        "--update-voices",
        action="store_true",
        help="Download latest voices.json during startup",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    global args
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    if not args.download_dir:
        args.download_dir = args.data_dir[0]

    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()