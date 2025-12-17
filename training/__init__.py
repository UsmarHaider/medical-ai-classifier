"""
Training Module for Medical Image Classification
"""
from .trainer import Trainer
from .callbacks import get_callbacks

__all__ = ['Trainer', 'get_callbacks']
