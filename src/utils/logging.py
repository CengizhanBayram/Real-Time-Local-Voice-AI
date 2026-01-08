# modules/logging.py
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = f"{LOG_DIR}/assistant_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
error_log_filename = f"{LOG_DIR}/errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logger = logging.getLogger("AssistantLogger")
logger.setLevel(logging.DEBUG)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)

# Rotating file handler for all logs (max 5MB per file, up to 3 backups)
fh = RotatingFileHandler(log_filename, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
fh.setLevel(logging.DEBUG)

# Rotating file handler for errors only
eh = RotatingFileHandler(error_log_filename, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
eh.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
eh.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)
logger.addHandler(eh)
