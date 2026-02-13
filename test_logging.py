
import logging
import sys
import os

# Configure logging like ask.py
file_handler = logging.FileHandler("test_log.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
logger = logging.getLogger(__name__)

# Log some unicode
logger.info("Testing unicode: 🔧 📦 ▶ ✓ ✗")
logger.info("Testing special chars: ‘Quotes’ and — dashes")

print("Logging test complete. Check test_log.log")
