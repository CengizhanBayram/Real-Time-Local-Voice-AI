# modules/virtual_assistant.py

import os
import time
import wave
import numpy as np
import sounddevice as sd
from typing import Optional
from src.utils.logging import logger
from src.services.whisper import WhisperRecognizer
from src.services.lm_studio import LMStudioClient
from src.utils.hotword import HotwordDetector
from src.utils.performance import PerformanceMonitor
import asyncio
import queue
from src.core.config import load_config
from src.services.realtime import RealtimeService

class Assistant:
    def __init__(self, config_path: str = "settings.yml"):
        self.config = load_config(config_path)
        
        self.mode = self.config.get("pipeline", {}).get("mode", "local")
        logger.info(f"Initializing Assistant in '{self.mode}' mode.")

        if self.mode == "local":
            self.recognizer = WhisperRecognizer(
                model_name=self.config["whisper"]["model"],
                sample_rate=self.config["whisper"]["sample_rate"],
                config=self.config
            )
            self.lm_client = LMStudioClient(self.config)
            # Always initialize hotword detector if enabled in config
            self.hotword = self.config.get("hotword", {})
            self.hotword_detector = HotwordDetector(config=self.config) if self.hotword.get("enabled", False) else None
        
        elif self.mode == "realtime":
            self.realtime_service = RealtimeService(self.config)
            # Realtime VAD is handled by the server (Server VAD), but we need audio streaming setup.
        
        self.performance = PerformanceMonitor()
        self._running = False
        self.audio_queue = queue.Queue() # For Realtime audio playback

    async def run(self):
        self._running = True
        logger.info("Assistant started.")
        
        if self.mode == "realtime":
            await self._run_realtime()
        else:
            await self._run_local()

    async def _run_local(self):
        """Legacy local pipeline: Whisper -> STT -> LLM -> TTS"""
        self.hotword = self.config.get("hotword", {})
        phrase = self.hotword.get("phrase", "Assistant")
        logger.info("Press ENTER or say '%s' to interact.", phrase)
        
        try:
            while self._running:
                # Need to use asyncio.to_thread for blocking IO in async loop if we want to be fully async proper,
                # but since this is the only thing running, we can just run it synchronously or loop with sleep(0).
                # However, existing code was sync blocking.
                if not await asyncio.to_thread(self._wait_for_activation):
                    continue
                
                # Record and process user input
                user_text = await asyncio.to_thread(self.recognizer.transcribe)
                if not user_text:
                    logger.warning("No speech detected.")
                    continue
                
                # Get and process response
                response_text = await asyncio.to_thread(self.lm_client.chat, user_text)
                self.performance.add_tokens(len(response_text.split()))
                
                # Synthesize speech
                word_count = len(response_text.split())
                if word_count > self.config["segmentation"]["max_words"]:
                    logger.info("Response is long (%d words). Segmenting...", word_count)
                    output_file = await asyncio.to_thread(self.lm_client.synthesize_long_text, response_text)
                else:
                    output_file = await asyncio.to_thread(self.lm_client.synthesize_speech, response_text)
                
                # Play audio
                await asyncio.to_thread(self.play_audio, output_file)
                # Wait again after audio playback (handled by loop start)

        except Exception as e:
            logger.error(f"Error in local loop: {e}")
            self._running = False

    async def _run_realtime(self):
        """Realtime API streaming loop."""
        await self.realtime_service.connect()
        
        # Audio Configuration
        fs = 24000 # Realtime API uses 24kHz
        channels = 1
        dtype = 'int16'
        blocksize = 1024
        
        loop = asyncio.get_running_loop()

        def input_callback(indata, frames, time_info, status):
            """Capture audio from mic and send to Realtime API."""
            if status:
                logger.debug(str(status))
            # Send audio chunk to websocket
            # We need to schedule the async send_audio_chunk from this sync callback
            asyncio.run_coroutine_threadsafe(
                self.realtime_service.send_audio_chunk(indata.tobytes()), 
                loop
            )

        def output_callback(outdata, frames, time_info, status):
             """Play audio received from Realtime API."""
             try:
                 # Fetch audio bytes from queue (blocking get, but we need non-blocking for real-time?)
                 # Actually sounddevice output callback expects us to fill `outdata`.
                 # We need to fill `frames` worth of data.
                 
                 # Simpler approach: Use a queue.
                 needed_bytes = frames * 2 # 16-bit = 2 bytes
                 buffer = bytearray()
                 
                 while len(buffer) < needed_bytes:
                     try:
                         # Non-blocking get
                         chunk = self.audio_queue.get_nowait()
                         buffer.extend(chunk)
                     except queue.Empty:
                         break
                 
                 # If we have data, write it
                 if len(buffer) > 0:
                     # If we have more than needed, put back remainder (naive implementation)
                     # Or just slice what we have.
                     # Better: use a bytebuffer wrapper or stream.
                     
                     # Using a simple approach: Pad with zeros if underflow
                     if len(buffer) < needed_bytes:
                         buffer.extend(b'\x00' * (needed_bytes - len(buffer)))
                     
                     # If too much (unlikely if chunks are small, but possible), just truncate and lose data? 
                     # No, properly we should buffer.
                     # For MVP: Let's assume queue chunks are small.
                     # Actually, `send_audio_chunk` sends raw bytes. 
                     # `on_audio_delta` receives raw bytes.
                     
                     to_play = buffer[:needed_bytes]
                     # If we have leftovers, how to put back? queue.Queue doesn't support push_front.
                     # We should use a persistent buffer/deque for the output stream.
                     pass 

                 # Let's use a simpler `OutputStream` where we just write() to it from the main loop
                 # instead of using a callback.
                 pass
             except Exception:
                 pass

        # Callback helper for incoming audio
        def on_audio_received(audio_bytes):
            # Put into an output stream directly?
            # Or put into queue for a dedicated playback thread/stream?
            # sd.OutputStream.write() is blocking.
            self.audio_queue.put(audio_bytes) 

        self.realtime_service.on_audio_delta = on_audio_received

        # Start Stream
        # using a raw InputStream for sending, and an OutputStream for playback
        # OR a Stream (duplex)
        
        # NOTE: Handling audio buffering for callbacks correctly in Python is tricky.
        # Simplest working method for MVP:
        # Input: Callback sends to WS.
        # Output: Thread reads from Queue and writes to OutputStream (blocking write).
        
        try:
            with sd.InputStream(samplerate=fs, channels=channels, dtype=dtype, 
                                callback=input_callback, blocksize=blocksize):
                
                # Output stream handler
                output_stream = sd.OutputStream(samplerate=fs, channels=channels, dtype=dtype)
                output_stream.start()
                
                logger.info("Listening... (Realtime Mode)")
                
                while self._running:
                    # Check queue for audio to play
                    try:
                        while not self.audio_queue.empty():
                            data = self.audio_queue.get_nowait()
                            # Write to stream (blocking but fast enough for chunks)
                            output_stream.write(np.frombuffer(data, dtype=np.int16))
                    except Exception as e:
                        logger.error(f"Playback error: {e}")
                    
                    await asyncio.sleep(0.01) # Yield to event loop
                    
                output_stream.stop()
                output_stream.close()

        except Exception as e:
            logger.error(f"Realtime loop error: {e}")
        finally:
            await self.realtime_service.disconnect()


    def stop(self):
        self._running = False

    def _flush_stdin(self):
        """Flush any lingering input from stdin."""
        try:
            import sys
            import termios
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except Exception:
            try:
                import msvcrt
                while msvcrt.kbhit():
                    msvcrt.getch()
            except Exception:
                pass

    def _wait_for_activation(self) -> bool:
        """
        Wait for activation either by detecting a keypress (push-to-talk)
        or by detecting the hotword, whichever comes first.
        """
        logger.info("Waiting for activation: press ENTER or say the hotword...")
        hotword_timeout = self.config["hotword"]["timeout_sec"] if self.hotword_detector else 0
        elapsed = 0.0
        check_interval = 0.5
        while elapsed < hotword_timeout:
            if self._check_for_keypress():
                self._flush_stdin()
                return True
            if self.hotword_detector and self.hotword_detector.check_for_hotword(timeout=check_interval):
                return True
            time.sleep(check_interval)
            elapsed += check_interval
        # Fallback to blocking push-to-talk input if neither hotword nor keypress detected within the timeout
        input("Press ENTER to speak...")
        return True

    def _check_for_keypress(self) -> bool:
        """Non-blocking keypress check."""
        try:
            import msvcrt  # Windows
            return msvcrt.kbhit()
        except ImportError:
            import sys
            import select  # Unix
            return sys.stdin in select.select([sys.stdin], [], [], 0)[0]

    def play_audio(self, filename: str):
        """Play audio with normalization and error handling."""
        if not os.path.exists(filename):
            logger.error("Audio file not found: %s", filename)
            return
        try:
            with wave.open(filename, "rb") as wf:
                sample_rate = wf.getframerate()
                audio_data = wf.readframes(wf.getnframes())
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
            if self.config["speech"]["normalize_audio"]:
                max_val = np.max(np.abs(audio_array))
                if max_val > 0:
                    audio_array = audio_array / max_val
            sd.play(audio_array, samplerate=sample_rate)
            sd.wait()
        except Exception as e:
            logger.error("Audio playback error: %s", str(e))
            raise
