import subprocess
import os
import glob
from pathlib import Path
from typing import Optional, List, Dict, Any
from utils import setup_logging
import logging 
import re

logger = logging.getLogger(__name__)

def extract_audio(input_path: str, temp_dir: str) -> str:
    """Extract audio from video file."""
    try:
        audio_path = os.path.join(temp_dir, "temp_extracted_audio.wav")
        logger.info(f"Extracting audio from {input_path}...")
        
        subprocess.run([
            'ffmpeg',
            '-i', input_path,
            '-ar', '16000',
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            '-vn',
            '-y',
            audio_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return audio_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract audio: {e}")
        raise

def split_audio(input_file: str, chunk_length: int, temp_dir: str) -> List[str]:
    """Split audio into chunks."""
    try:
        logger.info("Splitting audio into chunks...")
        chunk_pattern = os.path.join(temp_dir, 'chunk_%03d.wav')
        
        subprocess.run([
            'ffmpeg',
            '-i', input_file,
            '-ar', '16000',
            '-ac', '1',
            '-f', 'segment',
            '-segment_time', str(chunk_length),
            '-reset_timestamps', '1',
            '-c:a', 'pcm_s16le',
            chunk_pattern
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        chunk_files = sorted(glob.glob(os.path.join(temp_dir, 'chunk_*.wav')))
        return chunk_files
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed to split audio: {e}")
        raise

# File: audio_utils.py
def validate_audio_file(file_path: str, min_duration: float = 0.1) -> bool:
    """Check if audio file exists and has minimum duration."""
    try:
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return False

        # Use ffprobe to get duration
        result = subprocess.run([
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=nw=1',
            file_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)

        if result.returncode != 0:
            # Try alternative format
            result = subprocess.run([
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'stream=duration',
                '-of', 'default=nw=1',
                file_path
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)

        if result.returncode != 0:
            logger.error(f"ffprobe failed on {file_path}: {result.stderr}")
            return False

        duration_match = re.search(r'(\d+\.\d+|\d+)', result.stdout.strip())
        if not duration_match:
            logger.error(f"Could not parse duration from {file_path}")
            return False
            
        duration = float(duration_match.group(1))
        return duration >= min_duration

    except Exception as e:
        logger.error(f"Audio validation error: {e}")
        return False

def convert_to_wav(input_path: str, temp_dir: str) -> str:
    """Convert input file to 16kHz mono WAV."""
    try:
        converted_path = os.path.join(temp_dir, "converted_audio.wav")
        
        if Path(input_path).suffix.lower() == '.wav':
            result = subprocess.run([
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=sample_rate,channels',
                '-of', 'default=nw=1',
                input_path
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if 'sample_rate=16000' in result.stdout and 'channels=1' in result.stdout:
                return input_path

        
        subprocess.run([
            'ffmpeg',
            '-i', input_path,
            '-ar', '16000',
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            '-vn',
            '-y',
            '-hide_banner',
            '-loglevel', 'warning',
            converted_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if not os.path.exists(converted_path):
            raise RuntimeError("Conversion failed: output file not created")
            
        if not validate_audio_file(converted_path, 0.5):
            raise RuntimeError("Conversion failed: output file invalid")
            
        return converted_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion error: {e}")
        raise
