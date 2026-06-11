"""
Models subpackage — handles model training and evaluation:
  - ModelTrainer: Fits, serialises, and deserialises sklearn models
  - Evaluator:    Computes and persists classification / regression metrics
"""

from .trainer import ModelTrainer
from .evaluator import Evaluator

__all__ = ["ModelTrainer", "Evaluator"]
