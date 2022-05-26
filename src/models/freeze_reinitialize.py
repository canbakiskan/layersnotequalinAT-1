import torch
from torch import nn
from typing import Any, Optional


class Freeze_reinitialize_wrapper(nn.Module):
    def __init__(self, model: nn.Module, freeze_earlier: bool, cutoff_before: Optional[str] = None, single_layer_different: Optional[str] = None) -> None:
        super().__init__()

        assert (single_layer_different is not None) ^ (
            cutoff_before is not None)

        self.model = model

        if single_layer_different is not None:
            assert single_layer_different in [name for name, _ in self.model.named_modules(
            )], "single_layer_different must be a module in the model"
            self.single_layer_different: str = single_layer_different
        else:
            assert cutoff_before in [name for name, _ in self.model.named_modules(
            )], "cutoff_before must be a module in the model"
            self.cutoff_before: str = cutoff_before

        self.freeze_earlier: bool = freeze_earlier  # or single layer

    def freeze_reinitialize(self):

        def params_initialize(module):
            if type(module) in [torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d]:
                module.reset_parameters()
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.reset_running_stats()

        # to make sure everything is frozen to begin with
        for p in self.model.parameters():
            p.requires_grad = False

        if hasattr(self, "cutoff_before"):
            freeze = self.freeze_earlier
            for name, module in self.model.named_modules():

                if name == self.cutoff_before:
                    freeze = not freeze

                if not freeze:
                    if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                        module.weight.requires_grad = True
                    if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor):
                        module.bias.requires_grad = True
                    params_initialize(module)
        else:
            single_layer_frozen = self.freeze_earlier
            for name, module in self.model.named_modules():
                if (name == self.single_layer_different) ^ single_layer_frozen:
                    if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                        module.weight.requires_grad = True
                    if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor):
                        module.bias.requires_grad = True
                    params_initialize(module)

    def __getattribute__(self, name: str) -> Any:
        # the last three are used in nn.Module.__setattr__
        if name in ["freeze_reinitialize", "cutoff_before", "freeze_earlier", "single_layer_different",
                    "model", "__dict__", "_parameters", "_buffers", "_non_persistent_buffers_set"]:
            return object.__getattribute__(self, name)
        else:
            return getattr(self.model, name)
