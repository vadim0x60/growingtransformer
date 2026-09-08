"""A transformer that expands independently in depth and attention heads."""

from .model import GrowingTransformer, ModelConfig

__all__ = ["GrowingTransformer", "ModelConfig"]
