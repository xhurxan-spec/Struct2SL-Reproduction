"""MLP model for Struct2SL-Reproduction."""

import torch
import torch.nn as nn


class Struct2SLMLP(nn.Module):
    """
    MLP for synthetic-lethality prediction.

    The model returns raw logits. Use BCEWithLogitsLoss during training
    and torch.sigmoid(logits) to obtain probabilities during evaluation.
    """

    def __init__(
        self,
        input_size=2560,
        hidden_sizes=(256, 128, 64),
        output_size=1,
        dropout=0.5,
    ):
        super().__init__()

        if input_size <= 0:
            raise ValueError("input_size must be positive")

        if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")

        layers = []
        previous_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def save_checkpoint(model, path, metadata=None):
    """Save an independent CPU copy of the model weights and metadata."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    checkpoint = {
        "model_state_dict": state_dict,
        "metadata": metadata or {},
    }

    torch.save(checkpoint, path)


def load_checkpoint(model, path, map_location="cpu"):
    """Load a checkpoint saved by save_checkpoint()."""
    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint.get("metadata", {})
