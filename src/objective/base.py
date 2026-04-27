from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class ObjectiveFunction(ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...
