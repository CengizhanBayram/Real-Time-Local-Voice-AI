import asyncio
import json
import base64
import websockets
from typing import Optional, Callable, Dict, Any
from src.utils.logging import logger

class RealtimeService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config["realtime"]["api_key"]
        self.endpoint = config["realtime"]["endpoint"]
        self.voice = config["realtime"]["voice"]
        self.system_prompt = config["realtime"]["system_prompt"]
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self.on_audio_delta: Optional[Callable[[bytes], None]] = None

    async def connect(self):
        """Connect to the Realtime API WebSocket."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        logger.info(f"Connecting to Realtime API: {self.endpoint}")
        try:
            self.ws = await websockets.connect(self.endpoint, extra_headers=headers)
            self._running = True
            logger.info("Connected to Realtime API.")
            
            # Initial session setup
            await self._send_session_update()
            
            # Start receiving loop
            asyncio.create_task(self._receive_loop())
            
        except Exception as e:
            logger.error(f"Failed to connect to Realtime API: {e}")
            raise

    async def disconnect(self):
        """Disconnect from the WebSocket."""
        self._running = False
        if self.ws:
            await self.ws.close()
            logger.info("Disconnected from Realtime API.")

    async def _send_session_update(self):
        """Send session.update event to configure the session."""
        event = {
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": self.system_prompt,
                "voice": self.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                }
            }
        }
        await self._send(event)

    async def send_audio_chunk(self, audio_bytes: bytes):
        """Send raw PCM16 audio bytes to the API."""
        if not self.ws or not self._running:
            return

        # Encode audio to base64
        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
        
        event = {
            "type": "input_audio_buffer.append",
            "audio": base64_audio
        }
        await self._send(event)

    async def commit_audio(self):
         """Commit the user audio to trigger a response generation (if VAD is not auto-triggering)."""
         await self._send({"type": "input_audio_buffer.commit"})
         await self._send({"type": "response.create"})

    async def _send(self, event: dict):
        if self.ws and self._running:
            try:
                await self.ws.send(json.dumps(event))
            except Exception as e:
                logger.error(f"Error sending event: {e}")

    async def _receive_loop(self):
        """Listen for incoming events."""
        try:
            async for message in self.ws:
                if not self._running:
                    break
                try:
                    event = json.loads(message)
                    event_type = event.get("type")
                    
                    if event_type == "response.audio.delta":
                        audio_content = event.get("delta")
                        if audio_content and self.on_audio_delta:
                            audio_bytes = base64.b64decode(audio_content)
                            self.on_audio_delta(audio_bytes)
                            
                    elif event_type == "response.audio.done":
                        # Audio generation finished for a turn
                        pass
                        
                    elif event_type == "error":
                        logger.error(f"Realtime API Error: {event.get('error')}")

                except json.JSONDecodeError:
                    logger.error("Failed to decode JSON message")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("Realtime API connection closed.")
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")
