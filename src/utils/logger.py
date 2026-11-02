"""
Logging utilities for the Medication Management AI system.
"""
import logging
import sys
from pathlib import Path
from .config import LOG_LEVEL, LOG_FORMAT, BASE_DIR

# Create logs directory
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True, parents=True)

def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Setup a logger with console and file handlers.
    
    Args:
        name: Logger name
        log_file: Optional log file name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Console handler (robust to Windows cp1252 consoles): replace non-encodable chars
    class SafeStreamHandler(logging.StreamHandler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                msg = self.format(record)
                try:
                    self.stream.write(msg + self.terminator)
                except UnicodeEncodeError:
                    # Replace non-encodable characters for safety
                    self.stream.write(msg.encode(self.stream.encoding or 'utf-8', errors='replace').decode('utf-8') + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)

    console_handler = SafeStreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Use UTF-8 for files to preserve emojis/unicode
        file_handler = logging.FileHandler(LOGS_DIR / log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger
