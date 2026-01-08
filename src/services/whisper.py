# modules/whisper_recognizer.py
from typing import Optional
import os
import time
import torch
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
import tempfile
import whisper
from src.utils.logging import logger
from src.utils.audio import record_until_silence
from src.core.config import load_config

class WhisperRecognizer:
    def __init__(self, model_name: str = "base", sample_rate: int = 16000, config=None):
        self.config = config if config is not None else load_config()
        logger.info("Loading Whisper model (%s)...", model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_name, device=device)
        self.sample_rate = sample_rate
        logger.info("Whisper model loaded on %s", device)

    def transcribe(self, device: Optional[int] = None) -> str:
        """Record and transcribe audio with error handling"""
        try:
            logger.info("Recording...")
            audio = record_until_silence(
                self.sample_rate,
                device=device or self.config["audio"]["input_device"]
            )
            if audio.size == 0:
                logger.warning("No audio recorded")
                return ""
            # Use a temporary file for audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_file = tmp.name
                wav_write(temp_file, self.sample_rate, audio)
            logger.info("Transcribing...")
            start_time = time.time()
            result = self.model.transcribe(temp_file)
            elapsed = time.time() - start_time
            logger.debug("Transcription took %.2f seconds", elapsed)
            text = result.get("text", "").strip()
            if not text:
                logger.warning("No speech detected in audio")
            try:
                os.remove(temp_file)
            except Exception as e:
                logger.warning("Could not delete temporary file %s: %s", temp_file, str(e))
            return text
        except Exception as e:
            logger.error("Transcription failed: %s", str(e))
            return ""
