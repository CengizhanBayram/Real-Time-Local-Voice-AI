# modules/hotword_detector.py

from typing import Optional
import os
import numpy as np
import sounddevice as sd
import time
import tempfile
import whisper
from scipy.io.wavfile import write as wav_write
from src.utils.logging import logger
from src.core.config import load_config

class HotwordDetector:
    def __init__(self, config=None):
        self.config = config if config is not None else load_config()  # Use passed config if available
        hotword_config = self.config["hotword"]
        audio_config = self.config["audio"]

        self.enabled = hotword_config["enabled"]
        self.phrase = hotword_config["phrase"].lower()
        self.sensitivity = hotword_config["sensitivity"]
        self.timeout = hotword_config["timeout_sec"]
        self.retries = hotword_config["retries"]
        self.sample_rate = audio_config["hotword_sample_rate"]

        logger.info("Loading Whisper model for hotword detection...")
        self.whisper_model = whisper.load_model("tiny.en")
        logger.info("Hotword detector ready (listening for '%s')", self.phrase)

    def listen_for_hotword(self) -> bool:
        if not self.enabled:
            return True  # Bypass if disabled
        logger.info("Listening for hotword: '%s'...", self.phrase)
        for attempt in range(self.retries):
            try:
                # Use the dedicated check_for_hotword with a 1 second recording
                if self.check_for_hotword(timeout=1.0):
                    logger.info("Hotword detected!")
                    return True
                else:
                    logger.debug("Attempt %d: Hotword not detected", attempt + 1)
            except Exception as e:
                logger.error("Hotword detection attempt %d failed: %s", attempt + 1, str(e))
        logger.warning("Hotword not detected")
        return False

    def check_for_hotword(self, timeout: float = 1.0) -> bool:
        """
        Record for the full timeout duration and transcribe.
        Returns True if the hotword is found in the transcription.
        """
        device = self.config["audio"]["input_device"]
        try:
            # Record audio for 'timeout' seconds
            num_samples = int(self.sample_rate * timeout)
            audio = sd.rec(num_samples, samplerate=self.sample_rate, channels=1, dtype='float32', device=device)
            sd.wait()
            # Convert audio to int16 as expected by the transcriber
            audio_int16 = (audio.flatten() * 32767).astype('int16')
            text = self._transcribe_audio(audio_int16)
            logger.debug("Hotword check transcription: '%s'", text)
            return self.phrase in text.lower()
        except Exception as e:
            logger.error("Hotword check failed: %s", str(e))
            return False

    def _transcribe_audio(self, audio):
        if audio.size == 0:
            return ""
        # Use a temporary file for Whisper transcription
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_file = tmp.name
            wav_write(temp_file, self.sample_rate, audio)
        try:
            result = self.whisper_model.transcribe(temp_file)
            return result.get("text", "").strip()
        finally:
            try:
                os.remove(temp_file)
            except Exception as e:
                logger.warning("Could not delete temporary file %s: %s", temp_file, str(e))
