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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="HTTP server host")
    parser.add_argument("--port", type=int, default=5000, help="HTTP server port")
    parser.add_argument("-m", "--model", required=True, help="Path to Onnx model file")
    parser.add_argument("-c", "--config", help="Path to model config file")
    parser.add_argument("-s", "--speaker", type=int, help="Id of speaker (default: 0)")
    parser.add_argument(
        "--length-scale", "--length_scale", type=float, help="Phoneme length"
    )
    parser.add_argument(
        "--noise-scale", "--noise_scale", type=float, help="Generator noise"
    )
    parser.add_argument(
        "--noise-w", "--noise_w", type=float, help="Phoneme width noise"
    )
    parser.add_argument("--cuda", action="store_true", help="Use GPU")
    parser.add_argument(
        "--sentence-silence",
        "--sentence_silence",
        type=float,
        default=0.0,
        help="Seconds of silence after each sentence",
    )
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        action="append",
        default=[str(Path.cwd())],
        help="Data directory to check for downloaded models (default: current directory)",
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    if not args.download_dir:
        args.download_dir = args.data_dir[0]

    model_path = Path(args.model)
    if not model_path.exists():
        voices_info = get_voices(args.download_dir, update_voices=args.update_voices)

        aliases_info: Dict[str, Any] = {}
        for voice_info in voices_info.values():
            for voice_alias in voice_info.get("aliases", []):
                aliases_info[voice_alias] = {"_is_alias": True, **voice_info}

        voices_info.update(aliases_info)
        ensure_voice_exists(args.model, args.data_dir, args.download_dir, voices_info)
        args.model, args.config = find_voice(args.model, args.data_dir)

    voice = PiperVoice.load(args.model, config_path=args.config, use_cuda=args.cuda)
    synthesize_args = {
        "speaker_id": args.speaker,
        "length_scale": args.length_scale,
        "noise_scale": args.noise_scale,
        "noise_w": args.noise_w,
        "sentence_silence": args.sentence_silence,
    }

    app = Flask(__name__)

    @app.route("/synthesize", methods=["POST"])
    def app_synthesize() -> Any:
        if request.is_json:
            data = request.get_json()
            text = data.get("text", "").strip()
        else:
            return jsonify({"error": "Request must be JSON"}), 400

        if not text:
            return jsonify({"error": "No text provided"}), 400

        _LOGGER.debug("Synthesizing text: %s", text)
        with io.BytesIO() as wav_io:
            with wave.open(wav_io, "wb") as wav_file:
                voice.synthesize(text, wav_file, **synthesize_args)

            return wav_io.getvalue(), 200, {'Content-Type': 'audio/wav'}

    @app.route("/", methods=["GET"])
    def app_root() -> str:
        return "Piper Voice Server. Use POST /synthesize with JSON body {'text': 'your text here'} to synthesize speech."

    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
