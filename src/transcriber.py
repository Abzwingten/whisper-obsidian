import torch
import logging
from typing import Tuple, List, Dict, Any
from faster_whisper import WhisperModel, BatchedInferencePipeline
from utils import setup_logging, format_timestamp

logger = logging.getLogger(__name__)

class Transcriber:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._determine_device()
        self.compute_type = self._validate_compute_type()
        self.model = self._load_model()
    
    def _determine_device(self) -> str:
        if self.config['use_cuda'] and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _validate_compute_type(self) -> str:
        if self.config['compute_type'] == "float16":
            if self.device == "cuda" and torch.cuda.is_available():
                capability = torch.cuda.get_device_capability()
                if capability[0] >= 7:
                    return "float16"
                logger.warning(f"Your GPU architecture {capability} may not support float16 efficiently")
        return "default"
    
    def _load_model(self) -> WhisperModel:
        logger.info(f"Loading model: {self.config['model_name']} "
                   f"(Device: {self.device}, Precision: {self.compute_type})")
        
        return WhisperModel(
            self.config['model_name'],
            device=self.device,
            compute_type=self.compute_type
        )
    
    def transcribe(self, audio_path: str) -> Tuple[List[str], Dict[str, Any]]:
        try:
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=self.config['beam_size'],
                vad_filter=self.config['vad_filter'],
                vad_parameters=self.config['vad_parameters'],
                word_timestamps=self.config['word_timestamps']
            )
            
            segments = list(segments)
            return segments, info
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
