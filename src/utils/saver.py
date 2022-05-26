import torch
import os


def saver(model: torch.nn.Module, filepath: str):

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if not filepath.endswith(".pt"):
        filepath += ".pt"

    torch.save(model.state_dict(), filepath)
