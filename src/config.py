import argparse
from argparse import ArgumentParser
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from utils import setup_logging

class Config:
    DEFAULT_CONFIG = {
        "model_name": "small",
        "chunk_length": 120,
        "remove_timestamps": True,
        "cleanup": True,
        "use_cuda": True,
        "language": "ru",
        "beam_size": 5,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 500},
        "batch_size": 16,
        "compute_type": "float16",
        "output_format": "txt",
        "word_timestamps": False,
        "verbose": True,
        "temp_dir": None,
        "split_audio": True,
        "output_dir": "$HOME/Music",
        "output_file": "transcript.txt",
    }

    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        config = cls.DEFAULT_CONFIG.copy()
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    config.update(yaml_config)
            except (yaml.YAMLError, FileNotFoundError) as e:
                raise RuntimeError(f"Error loading config file: {e}")
        
        return config

    @classmethod
    def parse_args(cls) -> Dict[str, Any]:
        parser = argparse.ArgumentParser(
            description="Transcribe audio using faster-whisper with YAML config.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  Basic usage: python main.py --input input.wav
  With config: python main.py --config config.yaml --input input.mp4
  Batch processing: python main.py --input input.mp4 --output-dir /transcripts --output-format srt
"""
        )
        
        parser.add_argument('--config', type=str, 
                          help='Path to YAML config file. Settings here are overridden by CLI arguments.')
        parser.add_argument('--input', type=str, required=True,
                          help='Path to input audio or video file (WAV, MP3, MP4, AVI, MKV, MOV supported).')
        parser.add_argument('--model-name', type=str, default=cls.DEFAULT_CONFIG["model_name"],
                          help=f'Whisper model name. Available models: tiny, base, small, medium, large, large-v1, large-v2. Default: {cls.DEFAULT_CONFIG["model_name"]}')
        parser.add_argument('--chunk-length', type=int, default=cls.DEFAULT_CONFIG["chunk_length"],
                          help=f'Length of audio chunks in seconds when splitting. Default: {cls.DEFAULT_CONFIG["chunk_length"]}s')
        parser.add_argument('--output', type=str, default=cls.DEFAULT_CONFIG["output_file"],
                          help=f'Path to output text file. Default: {cls.DEFAULT_CONFIG["output_file"]}')
        parser.add_argument('--remove-timestamps', action='store_true',
                          help='Remove timestamps from output lines. Default: %(default)s')
        parser.add_argument('--cleanup', action='store_true',
                          help='Remove temporary files after processing. Default: %(default)s')
        parser.add_argument('--use-cuda', action='store_true',
                          help='Use CUDA backend for Whisper. Default: %(default)s')
        parser.add_argument('--language', type=str, default=cls.DEFAULT_CONFIG["language"],
                          help='Language code for transcription (e.g., en, fr, es). Default: auto-detect')
        parser.add_argument('--beam-size', type=int, default=cls.DEFAULT_CONFIG["beam_size"],
                          help=f'Beam size for decoding. Larger values use more memory but may improve accuracy. Default: {cls.DEFAULT_CONFIG["beam_size"]}')
        parser.add_argument('--vad-filter', action='store_true',
                          help='Enable voice activity detection filter. Default: %(default)s')
        parser.add_argument('--vad-min-silence', type=int, default=cls.DEFAULT_CONFIG["vad_parameters"]["min_silence_duration_ms"],
                          help=f'Minimum silence duration (ms) for VAD. Default: {cls.DEFAULT_CONFIG["vad_parameters"]["min_silence_duration_ms"]}ms')
        parser.add_argument('--batch-size', type=int, default=cls.DEFAULT_CONFIG["batch_size"],
                          help=f'Batch size for batched inference. Default: {cls.DEFAULT_CONFIG["batch_size"]}')
        parser.add_argument('--compute-type', choices=["default", "float16", "int8", "int8_float16"], 
                          default=cls.DEFAULT_CONFIG["compute_type"],
                          help=f'Model precision. Options: default (automatic), float16 (FP16), int8 (INT8), int8_float16 (hybrid). Default: {cls.DEFAULT_CONFIG["compute_type"]}')
        parser.add_argument('--format', choices=['txt', 'srt', 'vtt', 'json'], default='txt',
                          help='Output format. txt=text, srt=SubRip, vtt=WebVTT, json=JSON. Default: txt')
        parser.add_argument('--word-timestamps', action='store_true',
                          help='Include word-level timestamps. Default: %(default)s')
        parser.add_argument('--verbose', action='store_true',
                          help='Enable verbose logging. Default: %(default)s')
        parser.add_argument('--temp-dir', type=str, default=cls.DEFAULT_CONFIG["temp_dir"],
                          help='Custom temporary directory. Default: system temporary directory')
        parser.add_argument('--split-audio', action='store_true',
                          help='Split audio into chunks for better memory management. Default: %(default)s')

        
        args = parser.parse_args()

        config = cls.DEFAULT_CONFIG.copy()
        if args.config:
            try:
                with open(args.config, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    config.update({k: v for k, v in yaml_config.items() if k in config})
            except Exception as e:
                logging.error(f"Error loading config file: {e}")
                raise

        cli_args = {k: v for k, v in vars(args).items() if v is not None}
        config.update(cli_args)

        return argparse.Namespace(**config)

