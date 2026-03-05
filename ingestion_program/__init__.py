"""
Ingestion program module for the DPE Challenge.
This module contains the main ingestion logic for training and evaluating models.
"""

from .ingestion import main, get_train_data, evaluate_model, clean_features

__all__ = ['main', 'get_train_data', 'evaluate_model', 'clean_features']
