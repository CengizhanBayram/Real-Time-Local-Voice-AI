# modules/performance.py
import time
from src.utils.logging import logger

class PerformanceMonitor:
    def __init__(self, report_interval=60):
        self.start_time = time.time()
        self.token_count = 0
        self.audio_chunks = 0
        self.api_calls = 0
        self.errors = 0
        self.last_report_time = time.time()
        self.report_interval = report_interval

    def add_tokens(self, count=1):
        self.token_count += count

    def add_audio_chunk(self):
        self.audio_chunks += 1

    def add_api_call(self):
        self.api_calls += 1

    def add_error(self):
        self.errors += 1

    def report(self, force=False):
        now = time.time()
        if force or (now - self.last_report_time) >= self.report_interval:
            elapsed = now - self.start_time
            tokens_per_sec = self.token_count / elapsed if elapsed > 0 else 0
            logger.info("Performance: %.1f tokens/sec | %d API calls | %d errors | %d audio chunks",
                        tokens_per_sec, self.api_calls, self.errors, self.audio_chunks)
            self.last_report_time = now
