import logging
import os
from src.config import Config

def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(Config.LOG_LEVEL)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(Config.LOG_LEVEL)
        
        # File handler
        log_file = os.path.join(Config.LOG_DIR, "app.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(Config.LOG_LEVEL)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        
        logger.addHandler(ch)
        logger.addHandler(fh)
        
    return logger
