import copy

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self._source = model
        self.model = copy.deepcopy(model)
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def ema(self) -> None:
        """Update the EMA copy's parameters towards the source model's in-place."""
        torch._foreach_lerp_(
            list(self.model.parameters()),
            list(self._source.parameters()),
            1.0 - self.decay,
        )
        for ema_buf, src_buf in zip(self.model.buffers(), self._source.buffers()):
            ema_buf.copy_(src_buf, non_blocking=True)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "model": self.model.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.model.load_state_dict(state["model"])

    def to(self, *args, **kwargs) -> "EMA":
        self.model.to(*args, **kwargs)
        return self
