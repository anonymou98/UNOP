# utils/logger.py
"""
Unified Logging System
"""

import logging
import sys
import yaml
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir, rank=0, use_tensorboard=True):
        self.rank = rank
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("UNOP")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        if rank == 0:
            log_file = self.log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            console_handler = logging.StreamHandler(sys.stdout)
            
            formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
            if use_tensorboard:
                self.writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
            else:
                self.writer = None
        else:
            self.writer = None
    
    def info(self, msg):
        if self.rank == 0:
            self.logger.info(msg)
            
    def warning(self, msg):
        if self.rank == 0:
            self.logger.warning(msg)
            
    def error(self, msg):
        if self.rank == 0:
            self.logger.error(msg)
            
    def log_metrics(self, metrics, step, prefix=''):
        if self.rank == 0 and self.writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"{prefix}{k}", v, step)
                    
    def log_config(self, config):
        if self.rank == 0 and self.writer:
            config_text = yaml.dump(config, default_flow_style=False)
            self.writer.add_text('config', config_text, 0)
            
    def close(self):
        if self.rank == 0 and self.writer:
            self.writer.close()