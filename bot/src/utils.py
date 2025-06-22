"""
Utility functions for the trading bot.
"""
import logging
import logging.handlers
import sys

def initialize_logging(config):
    """Initializes the logging configuration."""
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO').upper()
    log_file = log_config.get('file', 'trades.log')
    max_size = log_config.get('max_file_size', 10) * 1024 * 1024  # in MB
    backup_count = log_config.get('backup_count', 5)

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console Handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # File Handler
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_size, backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.info("Logging initialized.") 