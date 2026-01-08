# modules/audio.py
import os
import time
import wave
import numpy as np
import sounddevice as sd
import webrtcvad
from scipy.io.wavfile import write as wav_write
from src.utils.logging import logger

# Default parameters (can be overridden via configuration)
VAD_MODE = 2
FRAME_DURATION_MS = 30
SILENCE_THRESHOLD_MS = 1000
MIN_RECORD_TIME_MS = 2000
DEFAULT_MAX_WORDS_PER_SEGMENT = 60

def record_until_silence(sample_rate, device=None):
    """
    Records audio until a period of silence is detected.
    """
    try:
        vad_inst = webrtcvad.Vad(VAD_MODE)
        frame_length = int(sample_rate * FRAME_DURATION_MS / 1000)
        silence_frames = int(SILENCE_THRESHOLD_MS / FRAME_DURATION_MS)
        logger.info("Recording until %d ms of silence is detected...", SILENCE_THRESHOLD_MS)
        recorded_frames = []
        consecutive_silence = 0
        start_time = time.time()

        with sd.InputStream(samplerate=sample_rate, channels=1, device=device, blocksize=frame_length) as stream:
            while True:
                frame, _ = stream.read(frame_length)
                frame_int16 = (np.squeeze(frame) * 32767).astype(np.int16).tobytes()
                is_speech = vad_inst.is_speech(frame_int16, sample_rate)
                recorded_frames.append(frame)
                if not is_speech:
                    consecutive_silence += 1
                else:
                    consecutive_silence = 0
                elapsed_ms = (time.time() - start_time) * 1000
                if consecutive_silence >= silence_frames and elapsed_ms > MIN_RECORD_TIME_MS:
                    logger.info("Silence detected; stopping recording (elapsed %.0f ms)", elapsed_ms)
                    break

        audio = np.concatenate(recorded_frames, axis=0)
        return audio
    except Exception as e:
        logger.error("Error during audio recording: %s", str(e))
        raise

def segment_text(text, max_words=DEFAULT_MAX_WORDS_PER_SEGMENT):
    """
    Splits text into segments of at most max_words, attempting to split at sentence boundaries.
    """
    words = text.split()
    segments = []
    while len(words) > max_words:
        segment = " ".join(words[:max_words])
        cutoff = max_words
        for i, word in enumerate(segment.split()):
            if word.endswith((".", "!", "?")):
                cutoff = i + 1
        segments.append(" ".join(words[:cutoff]).strip())
        words = words[cutoff:]
    if words:
        segments.append(" ".join(words).strip())
    return segments

def combine_audio_files(file_list, output_file):
    """
    Combines multiple WAV files into one.
    """
    if not file_list:
        return None
    try:
        with wave.open(file_list[0], "rb") as wf:
            params = wf.getparams()
        with wave.open(output_file, "wb") as out_wf:
            out_wf.setparams(params)
            for f in file_list:
                with wave.open(f, "rb") as wf:
                    out_wf.writeframes(wf.readframes(wf.getnframes()))
        logger.info("Combined audio written to %s", output_file)
        return output_file
    except Exception as e:
        logger.error("Error combining audio files: %s", str(e))
        raise
