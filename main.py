#!/usr/bin/env python
import argparse
import sys
from src.core.assistant import Assistant
from src.utils.logging import logger

import asyncio

def main():
    parser = argparse.ArgumentParser(description="Start the Voice Assistant.")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="settings.yml",
        help="Path to configuration YAML file."
    )
    args = parser.parse_args()

    try:
        assistant = Assistant(config_path=args.config)
        logger.info("Starting virtual assistant...")
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        logger.info("Assistant interrupted by user. Shutting down.")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)

    logger.info("Voice Assistant shutdown complete")

if __name__ == "__main__":
    main()
