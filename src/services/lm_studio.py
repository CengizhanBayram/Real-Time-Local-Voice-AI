# modules/lm_client.py
import os
import time
import json
import wave
import requests
import re
from typing import Optional
from src.utils.logging import logger
from src.utils.audio import segment_text, combine_audio_files
from src.utils.snac_decoder import tokens_decoder_sync
from src.core.config import load_config

def clean_text_for_tts(text: str) -> str:
    """
    Clean the text to be sent to the TTS engine by:
      • Removing newline characters and excessive whitespace.
      • Removing markdown symbols (e.g., asterisks).
      • Removing non-ASCII characters (e.g., emojis).
    """
    # Remove newline characters
    text = text.replace('\n', ' ')
    # Remove markdown formatting
    text = re.sub(r'\*+', '', text)
    # Remove non-ASCII characters (e.g., emojis)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class LMStudioClient:
    def __init__(self, config):
        self.config = config
        lm_config = config["lm"]

        self.api_url = lm_config["api_url"]
        self.chat_endpoint = lm_config["chat"]["endpoint"]
        self.tts_endpoint = lm_config["tts"]["endpoint"]

        # Chat parameters
        chat_config = lm_config["chat"]
        self.chat_model = chat_config["model"]
        self.system_prompt = chat_config["system_prompt"]
        self.chat_max_tokens = chat_config["max_tokens"]
        self.chat_temperature = chat_config["temperature"]
        self.chat_top_p = chat_config["top_p"]
        self.chat_repetition_penalty = chat_config["repetition_penalty"]
        self.max_response_time = chat_config["max_response_time"]

        # TTS parameters
        tts_config = lm_config["tts"]
        self.tts_model = tts_config["model"]
        self.default_voice = tts_config["default_voice"]
        self.tts_max_tokens = tts_config["max_tokens"]
        self.tts_temperature = tts_config["temperature"]
        self.tts_top_p = tts_config["top_p"]
        self.tts_repetition_penalty = tts_config["repetition_penalty"]
        self.speed = tts_config["speed"]
        self.max_segment_duration = tts_config["max_segment_duration"]

        self.headers = {"Content-Type": "application/json"}
        self.retries = config["speech"]["max_retries"]
        self.tts_sample_rate = config["tts"]["sample_rate"]

        # Use a session for connection pooling
        self.session = requests.Session()

    def chat(self, user_input: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "max_tokens": self.chat_max_tokens,
            "temperature": self.chat_temperature,
            "top_p": self.chat_top_p,
            "repeat_penalty": self.chat_repetition_penalty,
            "stream": False
        }
        url = self.api_url + self.chat_endpoint
        logger.debug("Chat request payload: %s", payload)
        for attempt in range(self.retries):
            try:
                start_time = time.time()
                response = self.session.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.max_response_time
                )
                elapsed = time.time() - start_time
                logger.info("Chat response received in %.2f seconds", elapsed)
                logger.debug("LM Studio full response: %s", response.text)
                if response.status_code != 200:
                    logger.error("Chat API error: %s %s", response.status_code, response.text)
                    if attempt < self.retries - 1:
                        delay = 2 ** attempt
                        logger.warning("Chat API error, retrying in %d seconds...", delay)
                        time.sleep(delay)
                        continue
                    raise RuntimeError(f"Chat API error: {response.status_code} {response.text}")
                data = response.json()
                generated_text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                token_count = len(generated_text.split()) * 1.33
                logger.info("Generated %d tokens: %s", int(token_count), generated_text[:50])
                return generated_text
            except requests.exceptions.Timeout:
                delay = 2 ** attempt
                logger.warning("Chat API timeout (attempt %d), retrying in %d seconds", attempt + 1, delay)
                time.sleep(delay)
                if attempt == self.retries - 1:
                    return "I need more time to think about that. Could you ask again?"
            except Exception as e:
                delay = 2 ** attempt
                logger.error("Chat API failed (attempt %d): %s", attempt + 1, str(e))
                time.sleep(delay)
                if attempt == self.retries - 1:
                    return "I'm having trouble responding right now. Please try again later."

    def synthesize_speech(self, text: str, voice: Optional[str] = None, output_file: Optional[str] = None) -> str:
        voice = voice if voice else self.default_voice
        if not voice.isalpha() or len(voice) > 20:
            logger.warning("Invalid voice name '%s', using default", voice)
            voice = self.default_voice

        # Clean the text for TTS
        cleaned_text = clean_text_for_tts(text)
        prompt = f"<|audio|>{voice}: {cleaned_text}<|eot_id|>"
        payload = {
            "model": self.tts_model,
            "prompt": prompt,
            "max_tokens": self.tts_max_tokens,
            "temperature": self.tts_temperature,
            "top_p": self.tts_top_p,
            "repeat_penalty": self.tts_repetition_penalty,
            "speed": self.speed,
            "stream": True
        }
        url = self.api_url + self.tts_endpoint
        logger.debug("TTS request payload: %s", payload)
        for attempt in range(self.retries):
            try:
                response = self.session.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    stream=True,
                    timeout=self.max_segment_duration + 5
                )
                if response.status_code != 200:
                    logger.error("TTS API error: %s %s", response.status_code, response.text)
                    if attempt < self.retries - 1:
                        delay = 2 ** attempt
                        logger.warning("TTS API error, retrying in %d seconds...", delay)
                        time.sleep(delay)
                        continue
                    raise RuntimeError(f"TTS API error: {response.status_code} {response.text}")

                def token_generator():
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode("utf-8")
                            if decoded_line.startswith("data: "):
                                data_str = decoded_line[6:]
                                if data_str.strip() == "[DONE]":
                                    break
                                try:
                                    data = json.loads(data_str)
                                    token_text = data.get("choices", [{}])[0].get("text", "")
                                    yield token_text
                                except json.JSONDecodeError as e:
                                    logger.error("JSON decode error: %s", e)

                audio_bytes = tokens_decoder_sync(token_generator())
                if not output_file:
                    timestamp = int(time.time())
                    output_file = f"outputs/{voice}_{timestamp}.wav"
                os.makedirs("outputs", exist_ok=True)
                with wave.open(output_file, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.tts_sample_rate)
                    wf.writeframes(audio_bytes)
                logger.info("Audio saved to %s", output_file)
                return output_file
            except Exception as e:
                delay = 2 ** attempt
                logger.error("TTS synthesis failed (attempt %d): %s", attempt + 1, str(e))
                time.sleep(delay)
                if attempt == self.retries - 1:
                    raise

    def synthesize_long_text(self, text: str, voice: Optional[str] = None) -> str:
        voice = voice if voice else self.default_voice
        segments = segment_text(text, max_words=self.config["segmentation"]["max_words"])
        logger.info("Text segmented into %d parts", len(segments))
        file_list = []
        for i, seg in enumerate(segments):
            seg_filename = f"outputs/{voice}_{int(time.time())}_{i}.wav"
            try:
                self.synthesize_speech(seg, voice=voice, output_file=seg_filename)
                file_list.append(seg_filename)
                time.sleep(0.2)  # Small delay between segments
            except Exception as e:
                logger.error("Failed to synthesize segment %d: %s", i, str(e))
                if file_list:
                    break
                raise
        combined_filename = f"outputs/{voice}_{int(time.time())}_combined.wav"
        combine_audio_files(file_list, combined_filename)
        for f in file_list:
            try:
                os.remove(f)
            except Exception as e:
                logger.warning("Could not remove temporary file %s: %s", f, str(e))
        return combined_filename
