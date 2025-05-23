import os
import glob
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)
from tempfile import TemporaryDirectory
from config import Config
from audio_utils import validate_audio_file, extract_audio, convert_to_wav, split_audio
from transcriber import Transcriber
from formatter import format_output, remove_timestamps
from utils import setup_logging, format_timestamp

def generate_output_path(input_path: str, output_dir: Optional[str] = None, 
                        output_format: str = "txt") -> str:
    """Generate output path based on input filename without spaces"""
    input_stem = Path(input_path).stem.replace(" ", "_")
    filename = f"{input_stem}_transcript.{output_format}"
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)
    
    return os.path.join(os.path.dirname(input_path), filename)

def process_file(input_path: str, output_path: str, config: Dict[str, Any]) -> None:
    """Process audio file through full transcription pipeline."""
    temp_dir = None
    audio_file = input_path
    
    try:
        if config['temp_dir']:
            os.makedirs(config['temp_dir'], exist_ok=True)
            temp_dir = config['temp_dir']
        else:
            temp_dir_obj = TemporaryDirectory()
            temp_dir = temp_dir_obj.name
        
        logger.info(f"Using temporary directory: {temp_dir}")
        
        if input_path.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            audio_file = extract_audio(input_path, temp_dir)
        
        processed_file = convert_to_wav(audio_file, temp_dir)
        
        if config['split_audio']:
            chunk_files = split_audio(processed_file, config['chunk_length'], temp_dir)
        else:
            chunk_files = [processed_file]
        
        transcriber = Transcriber(config)
        
        full_transcript = []
        for i, chunk_path in enumerate(chunk_files):
            if not validate_audio_file(chunk_path):
                logger.warning(f"Skipping invalid audio chunk: {chunk_path}")
                continue
            
            offset = i * config['chunk_length'] if config['split_audio'] else 0
            logger.debug(f"Processing chunk {i+1} with offset {offset}s")
            
            try:
                segments, info = transcriber.transcribe(chunk_path)
                
                if not segments:
                    logger.warning(f"No speech segments detected in {chunk_path}")
                    continue
                
                if config['language'] is None and info.language_probability > 0.5:
                    logger.info(f"Detected language: {info.language} ({info.language_probability:.2f})")
                
                for segment in segments:
                    start_abs = segment.start + offset
                    end_abs = segment.end + offset
                    
                    if config['word_timestamps']:
                        words = " ".join([f"[{format_timestamp(word.start)}-{format_timestamp(word.end)}]{word.word}"
                                        for word in segment.words])
                        line = f"[{format_timestamp(start_abs)} --> {format_timestamp(end_abs)}] {words}"
                    else:
                        line = f"[{format_timestamp(start_abs)} --> {format_timestamp(end_abs)}] {segment.text}"
                    
                    full_transcript.append(line)
            
            except Exception as e:
                logger.error(f"Failed to transcribe {chunk_path}: {e}")
                continue
        
        if config['remove_timestamps']:
            logger.info("Removing timestamps from transcript")
            full_transcript = remove_timestamps(full_transcript)
        
        # Generate output path
#        output_path = generate_output_path(
#            input_path,
#            config.get('output_dir'),
#            config['output_format']
#        )
        
        logger.info(f"Writing transcript to {output_path}")
        output_content = format_output(full_transcript, config['output_format'], config['word_timestamps'])
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_content)
            
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise
    finally:
        if config['cleanup'] and temp_dir and not config['temp_dir']:
            logger.info("Cleaning up temporary files")
            for f in glob.glob(os.path.join(temp_dir, 'chunk_*.wav')):
                try:
                    os.remove(f)
                except Exception as e:
                    logger.warning(f"Failed to delete {f}: {e}")


def generate_output_path(input_path: str, output_dir: str = None, 
                         output_format: str = "txt") -> str:
    """Generate output path based on input filename"""
    input_stem = Path(input_path).stem.replace(" ", "_")
    filename = f"{input_stem}_transcript.{output_format}"
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)
    
    return os.path.join(os.path.dirname(input_path), filename)

def main():
    args = Config.parse_args()
    
    config = Config.load_config(args.config)
    logging.info(f"Final config: {config}")
    for key, value in vars(args).items():
        logging.info(f"{key} : {value}")
        if value is not None:
            if key == "vad_min_silence":
                config["vad_parameters"]["min_silence_duration_ms"] = value
            else:
                config[key] = value
    
    setup_logging(config['verbose'])
    logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    output_path = generate_output_path(
        args.input,
        args.output_dir if hasattr(args, 'output_dir') else None,
        args.format
    )

    process_file(args.input, output_path, config)

if __name__ == "__main__":
    main()
