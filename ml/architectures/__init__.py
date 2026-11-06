"""Model registry with lazy architecture imports."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    module: str
    class_name: str
    text_encoder: str | None
    description: str

    @property
    def cls(self):
        return getattr(import_module(self.module), self.class_name)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "cnn": ModelSpec("ml.architectures.cnn", "CNNClassifier", None, "CNN image-only classifier"),
    "lstm": ModelSpec("ml.architectures.lstm", "LSTMClassifier", "word", "BiLSTM text-only classifier"),
    "bert": ModelSpec("ml.architectures.bert", "BERTClassifier", "bert", "BERT text-only classifier"),
    "cnn_lstm": ModelSpec("ml.architectures.cnn_lstm", "CNNLSTMClassifier", "word", "CNN + BiLSTM multimodal fusion"),
    "cnn_bert": ModelSpec("ml.architectures.cnn_bert", "CNNBERTClassifier", "bert", "CNN + BERT multimodal fusion"),
}


def build_model(name: str, **params: Any):
    """Instantiate a registered architecture only when requested."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name].cls(**params)


_LAZY_EXPORTS = {
    "BaseSentimentModel": ("ml.architectures.base_model", "BaseSentimentModel"),
    "CNNClassifier": ("ml.architectures.cnn", "CNNClassifier"),
    "LSTMClassifier": ("ml.architectures.lstm", "LSTMClassifier"),
    "BERTClassifier": ("ml.architectures.bert", "BERTClassifier"),
    "CNNLSTMClassifier": ("ml.architectures.cnn_lstm", "CNNLSTMClassifier"),
    "CNNBERTClassifier": ("ml.architectures.cnn_bert", "CNNBERTClassifier"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, class_name = _LAZY_EXPORTS[name]
    return getattr(import_module(module_name), class_name)


__all__ = ["MODEL_REGISTRY", "ModelSpec", "build_model", *_LAZY_EXPORTS]
