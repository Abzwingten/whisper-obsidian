#!/usr/bin/env python3
"""Transcribe audio/video files using faster-whisper with GPU support and batch processing"""
import os
import re
import json
import yaml
import subprocess
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Tuple, Any
from contextlib import contextmanager
import torch
from faster_whisper import WhisperModel
from dataclasses import replace


### CONFIGURATION HANDLING ###
def str2bool(v: str) -> bool:
    """Convert string to boolean"""
    return v.lower() in ('yes', 'true', 't', '1')

def parse_args_with_config() -> Dict[str, Any]:
    """Parse command line arguments and merge with config files and environment variables"""
    parser = argparse.ArgumentParser(description="Transcribe media files using faster-whisper")
    
    # Core Parameters
    parser.add_argument("--input", required=True, help="Input file or directory path")
    parser.add_argument("--model-name", default="large-v2", help="Model size (tiny/base/small/medium/large/large-v2)")
    parser.add_argument("--compute-type", default="int8_float16", help="Compute type (float16/int8/int8_float16/default)")
    parser.add_argument("--use-cuda", type=str2bool, default=True, help="Use GPU acceleration")
    parser.add_argument("--format", default="txt", choices=["txt", "srt", "vtt", "json"], help="Output format")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--config", help="Path to config file (yaml/json)")
    parser.add_argument("--output-dir", default="./transcripts", help="Output directory")
    
    # Audio Processing Parameters
    parser.add_argument("--split-audio", type=str2bool, default=True, help="Split large files into chunks")
    parser.add_argument("--chunk-length", type=int, default=30, help="Seconds per audio chunk")
    parser.add_argument("--vad-filter", type=str2bool, default=True, help="Apply voice activity detection")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding (1-10)")
    parser.add_argument("--remove-timestamps", type=str2bool, default=False, help="Clean output without timestamps")
    parser.add_argument("--word-timestamps", type=str2bool, default=False, help="Generate word-level timestamps")
    parser.add_argument("--language", help="Specify source language")

    # Parse initial arguments to get config file path
    args, _ = parser.parse_known_args()
    
    # Default configuration
    config = {
        "model": {
            "name": "large-v2",
            "compute_type": "int8_float16",
            "use_cuda": True
        },
        "audio": {
            "split_audio": True,
            "chunk_length": 30,
            "vad_filter": True
        },
        "transcription": {
            "beam_size": 5,
            "word_timestamps": False,
            "remove_timestamps": False,
            "batch_size": 32,
            "format": "txt"
        },
        "output": {
            "directory": "./transcripts"
        },
        "language": None
    }

    # Merge with config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            file_config = yaml.safe_load(f) if args.config.endswith('.yaml') else json.load(f)
            deep_update(config, file_config)

    # Merge with environment variables
    env_prefix = "WHISPER_"
    for key in flatten_dict(config):
        env_key = f"{env_prefix}{key.upper()}"
        if env_key in os.environ:
            value = os.environ[env_key]
            current_value = get_nested_value(config, key.split('.'))
            # Type conversion
            if isinstance(current_value, bool):
                value = str2bool(value)
            elif isinstance(current_value, int):
                value = int(value)
            elif isinstance(current_value, float):
                value = float(value)
            set_nested_value(config, key.split('.'), value)

    # Parse final CLI arguments
    cli_args = parser.parse_args()
    config["input"] = cli_args.input
    config["model"]["name"] = cli_args.model_name
    config["model"]["compute_type"] = cli_args.compute_type
    config["model"]["use_cuda"] = cli_args.use_cuda
    config["transcription"]["format"] = cli_args.format
    config["transcription"]["batch_size"] = cli_args.batch_size
    config["audio"]["split_audio"] = cli_args.split_audio
    config["audio"]["chunk_length"] = cli_args.chunk_length
    config["audio"]["vad_filter"] = cli_args.vad_filter
    config["transcription"]["beam_size"] = cli_args.beam_size
    config["transcription"]["word_timestamps"] = cli_args.word_timestamps
    config["transcription"]["remove_timestamps"] = cli_args.remove_timestamps
    config["output"]["directory"] = cli_args.output_dir
    config["language"] = cli_args.language if cli_args.language else config["language"]

    return config

def deep_update(d: Dict, u: Dict) -> Dict:
    """Recursively update dictionary d with values from u"""
    for k, v in u.items():
        if isinstance(v, dict) and d.get(k) and isinstance(d[k], dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d

def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_nested_value(d: Dict, keys: List) -> Any:
    """Get value from nested dictionary"""
    for key in keys:
        d = d[key]
    return d

def set_nested_value(d: Dict, keys: List, value: Any) -> None:
    """Set value in nested dictionary using list of keys"""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

### AUDIO HANDLING ###
def is_valid_audio(file_path: str) -> bool:
    """Verify file exists and has minimum duration (0.1s)"""
    if not os.path.exists(file_path):
        print(f"Audio file not found: {file_path}")
        return False
    
    cmd = [
        "ffprobe", "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "default=nw=1", file_path
    ]
    
    cmd2 = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'stream=duration',
        '-of', 'default=nw=1'
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        if result.returncode != 0:
            result = subprocess.run(
            cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        duration_match = re.search(r'(\d+\.\d+|\d+)', result.stdout.strip())
        duration = float(duration_match.group(1))

        return duration >= 0.5
    except (subprocess.CalledProcessError, ValueError):
        return False

def is_video_file(file_path: str) -> bool:
    """Check if file is a video file"""
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov'}
    return os.path.splitext(file_path)[1].lower() in video_extensions

def extract_audio(input_path: str, output_dir: str) -> str:
    """Extract audio from video file"""
    audio_path = os.path.join(output_dir, f"{Path(input_path).stem}.wav")
    
    cmd = [
        "ffmpeg", "-i", input_path, "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", "-y", audio_path
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return audio_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Audio extraction failed: {e}")

def split_audio(audio_path: str, chunk_length: int, output_dir: str) -> List[str]:
    """Split audio file into chunks"""
    chunks = []
    output_template = os.path.join(output_dir, "chunk_%03d.wav")
    
    cmd = [
        "ffmpeg", "-i", audio_path, "-f", "segment", "-segment_time", str(chunk_length),
        "-c", "copy", "-y", output_template
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # Collect generated chunks
        i = 0
        while True:
            chunk_path = output_template % i
            if os.path.exists(chunk_path):
                chunks.append(chunk_path)
                i += 1
            else:
                break
        return chunks
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Audio splitting failed: {e}")

### MODEL & TRANSCRIPTION ###
def load_model(config: Dict) -> WhisperModel:
    """Load Whisper model with appropriate settings"""
    model_name = config["model"]["name"]
    compute_type = config["model"]["compute_type"]
    use_cuda = config["model"]["use_cuda"] and torch.cuda.is_available()
    
    if use_cuda:
        device = "cuda"
    else:
        device = "cpu"
        # Fallback compute type if CUDA not available
        if compute_type in ["float16", "int8_float16"]:
            compute_type = "default"
    
    try:
        print(f"Loading model {model_name} on {device} with compute type {compute_type}")
        return WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")

def transcribe_audio(model: WhisperModel, audio_path: str, config: Dict) -> Tuple[List, Any]:
    """Transcribe audio file with specified settings"""
    try:
        return model.transcribe(
            audio_path,
            beam_size=config["transcription"]["beam_size"],
            word_timestamps=config["transcription"]["word_timestamps"],
            vad_filter=config["audio"]["vad_filter"],
            language=config["language"]
        )
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")

### OUTPUT FORMATTING ###
def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT/VTT timestamp format"""
    m, s = divmod(int(seconds * 1000), 60000)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s//1000:02d},{s%1000:03d}"

def remove_timestamps(lines: List[str]) -> List[str]:
    """Remove timestamp-like patterns from list of lines"""
    pattern = re.compile(r'^$$.*$$]')
    return [pattern.sub('', line).strip() for line in lines]

def format_output(segments: List, config: Dict) -> str:
    """Format transcription segments to requested output format"""
    format_type = config["transcription"]["format"]
    remove_ts = config["transcription"]["remove_timestamps"]
    
    if format_type == "txt":
        lines = []
        for seg in segments:
            text = seg.text
            if remove_ts:
                text = " ".join(remove_timestamps([text]))
            lines.append(text)
        return "\n".join(lines)
    
    elif format_type == "srt":
        content = []
        for i, seg in enumerate(segments):
            text = seg.text
            if remove_ts:
                text = " ".join(remove_timestamps([text]))
            content.append(
                f"{i+1}\n"
                f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n"
                f"{text}\n"
            )
        return "\n".join(content)
    
    elif format_type == "vtt":
        content = ["WEBVTT\n"]
        for seg in segments:
            text = seg.text
            if remove_ts:
                text = " ".join(remove_timestamps([text]))
            content.append(
                f"{format_timestamp(seg.start).replace(',', '.')} --> "
                f"{format_timestamp(seg.end).replace(',', '.')}\n"
                f"{text}\n"
            )
        return "\n".join(content)
    
    elif format_type == "json":
        result = []
        for seg in segments:
            text = seg.text
            if remove_ts:
                text = " ".join(remove_timestamps([text]))
            seg_dict = {
                "start": seg.start,
                "end": seg.end,
                "text": text,
                "words": []
            }
            if hasattr(seg, "words"):
                seg_dict["words"] = [
                    {"start": w.start, "end": w.end, "word": w.word} for w in seg.words
                ]
            result.append(seg_dict)
        return json.dumps(result, indent=2)
    
    raise ValueError(f"Unsupported format: {format_type}")

### UTILS ###
@contextmanager
def temp_dir_manager(cleanup: bool = True) -> str:
    """Context manager for temporary directory handling"""
    with TemporaryDirectory() as temp_dir:
        try:
            yield temp_dir
        finally:
            if cleanup:
                pass  # Automatically cleaned up by TemporaryDirectory

### MAIN PROCESSING ###
def process_file(input_path: str, config: Dict) -> None:
    """Process a single media file"""
    output_dir = config["output"]["directory"]
    os.makedirs(output_dir, exist_ok=True)
    
    input_basename = os.path.splitext(os.path.basename(input_path))[0]
    output_ext = config["transcription"]["format"]
    output_path = os.path.join(output_dir, f"{input_basename}_transcript.{output_ext}")
    
    if os.path.exists(output_path):
        print(f"Skipping existing transcript: {output_path}")
        return

    with temp_dir_manager() as temp_dir:
        # Step 1: Audio extraction or conversion
        if is_video_file(input_path):
            audio_path = extract_audio(input_path, temp_dir)
        else:
            audio_path = input_path
        
        # Step 2: Audio chunking
        if config["audio"]["split_audio"]:
            chunks = split_audio(audio_path, config["audio"]["chunk_length"], temp_dir)
        else:
            chunks = [audio_path]
        
        # Step 3: Model loading
        model = load_model(config)
        
        # Step 4: Process all chunks
        all_segments = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            segments, _ = transcribe_audio(model, chunk, config)
            
            # Adjust timestamps for chunked files
            offset = i * config["audio"]["chunk_length"]
            adjusted_segments = [
                replace(
                    seg,
                    start=seg.start + offset,
                    end=seg.end + offset
                ) for seg in segments
            ]
            all_segments.extend(adjusted_segments)
        
        # Step 5: Format and save output
        output_content = format_output(all_segments, config)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_content)
        
        print(f"Transcription saved to {output_path}")

def main():
    """Main entry point"""
    try:
        config = parse_args_with_config()
        input_path = config["input"]
        
        if os.path.isdir(input_path):
            print(f"Processing directory: {input_path}")
            for filename in os.listdir(input_path):
                file_path = os.path.join(input_path, filename)
                if os.path.isfile(file_path) and is_valid_audio(file_path):
                    print(f"Processing file: {file_path}")
                    process_file(file_path, config)
        else:
            if is_valid_audio(input_path):
                process_file(input_path, config)
            else:
                print(f"Invalid input file: {input_path}")
    
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()

