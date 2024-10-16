#!/usr/bin/env python3
import argparse
import io
import json
import logging
import wave
from pathlib import Path
from typing import Any, Dict

from flask import Flask, request, jsonify

from . import PiperVoice
from .download import ensure_voice_exists, find_voice, get_voices

_LOGGER = logging.getLogger()

# Global dictionary to store loaded models
loaded_models = {}

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
    parser.add_argument(
        "--length-scale", "--length_scale", type=float, default=1.0, help="Phoneme length"
    )
    parser.add_argument(
        "--noise-scale", "--noise_scale", type=float, default=0.667, help="Generator noise"
    )
    parser.add_argument(
        "--noise-w", "--noise_w", type=float, default=0.8, help="Phoneme width noise"
    )
    parser.add_argument(
        "--sentence-silence",
        "--sentence_silence",
        type=float,
        default=0.0,
        help="Seconds of silence after each sentence",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    if not args.download_dir:
        args.download_dir = args.data_dir[0]

    app = Flask(__name__)

    @app.route("/synthesize", methods=["POST"])
    def app_synthesize() -> Any:
        if request.is_json:
            data = request.get_json()
            text = data.get("text", "").strip()
            model_name = data.get("model_name", "").strip() + ".onnx"
        else:
            return jsonify({"error": "Request must be JSON"}), 400

        if not text:
            return jsonify({"error": "No text provided"}), 400
        if not model_name:
            return jsonify({"error": "No model name provided"}), 400

        _LOGGER.debug("Synthesizing text: %s with model: %s", text, model_name)

        # Load model if not already loaded
        if model_name not in loaded_models:
            _LOGGER.info("Loading model: %s", model_name)
            model_path = Path(f"/opt/digital_human/piper/app/models/{model_name}")
            if not model_path.exists():
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

            loaded_models[model_name] = PiperVoice.load(model_path, config_path=config_path, use_cuda=args.cuda)

        voice = loaded_models[model_name]

        synthesize_args = {
            "speaker_id": data.get("speaker", 0),
            "length_scale": data.get("length_scale", args.length_scale),
            "noise_scale": data.get("noise_scale", args.noise_scale),
            "noise_w": data.get("noise_w", args.noise_w),
            "sentence_silence": data.get("sentence_silence", args.sentence_silence),
        }

        with io.BytesIO() as wav_io:
            import time
            start_time = time.time()
            with wave.open(wav_io, "wb") as wav_file:
                voice.synthesize(text, wav_file, **synthesize_args)
            end_time = time.time()
            _LOGGER.info("Synthesized text: '%s' in %.2f seconds", text, end_time - start_time)

            return wav_io.getvalue(), 200, {'Content-Type': 'audio/wav'}

    @app.route("/", methods=["GET"])
    def app_root() -> str:
        return "Piper Voice Server. Use POST /synthesize with JSON body {'text': 'your text here', 'model_name': 'your_model_here'} to synthesize speech."

    from gunicorn.app.base import BaseApplication

    class StandaloneApplication(BaseApplication):
        def __init__(self, app, options=None):
            self.application = app
            self.options = options or {}
            super().__init__()

        def load_config(self):
            config = {key: value for key, value in self.options.items() if key in self.cfg.settings and value is not None}
            for key, value in config.items():
                self.cfg.set(key.lower(), value)

        def load(self):
            return self.application

    options = {
        "bind": f"{args.host}:{args.port}",
        "workers": 1,
    }
    StandaloneApplication(app, options).run()

if __name__ == "__main__":
    main()