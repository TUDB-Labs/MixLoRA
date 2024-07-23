import math

import torch
import torch.nn as nn
from transformers.utils import is_bitsandbytes_available

from .config import LoraConfig

if is_bitsandbytes_available():
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit, Linear8bitLt
else:
    from .utils import Linear8bitLt, Linear4bit

from typing import Dict, Optional, Tuple


def dequantize_bnb_weight(weight: torch.nn.Parameter, state=None):
    # BNB requires CUDA weights
    device = weight.device
    is_cpu = device.type == torch.device("cpu").type
    if is_cpu:
        weight = weight.to(torch.device("cuda"))

    cls_name = weight.__class__.__name__
    if cls_name == "Params4bit":
        dequantized = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
        if is_cpu:
            dequantized = dequantized.to(device)
        return dequantized

    if state.SCB is None:
        state.SCB = weight.SCB

    im = torch.eye(weight.data.shape[-1]).contiguous().half().to(weight.device)
    im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
    im, Sim = bnb.functional.transform(im, "col32")
    if state.CxB is None:
        state.CxB, state.SB = bnb.functional.transform(
            weight.data, to_order=state.formatB
        )
    out32, Sout32 = bnb.functional.igemmlt(im, state.CxB, Sim, state.SB)
    dequantized = bnb.functional.mm_dequant(
        out32, Sout32, SCim, state.SCB, bias=None
    ).t()
    if is_cpu:
        dequantized = dequantized.to(device)
    return dequantized


def dequantize_module_weight(module: torch.nn.Module) -> torch.nn.Parameter:
    if hasattr(module, "W_q"):  # For handling HQQ quantized weight
        weight = module.dequantize()
        return weight

    weight = module.weight
    if not isinstance(weight, torch.nn.Parameter):
        raise TypeError(
            f"Input weight should be of type nn.Parameter, got {type(weight)} instead"
        )

    cls_name = weight.__class__.__name__
    if cls_name not in ("Params4bit", "Int8Params"):
        return weight

    quant_state = getattr(module, "state", None)
    device = weight.device
    is_cpu = device.type == torch.device("cpu").type
    weight = dequantize_bnb_weight(weight, state=quant_state)  # no-op if not bnb
    if is_cpu:
        # dequantize_bnb_weight for 8bit moves the device in-place, thus we need to move it back to CPU if necessary
        module.weight = module.weight.to(device)
    return weight


g_cached_range_tensor: Dict[torch.device, torch.Tensor] = {}
# also max batch size
g_max_range = 128


def get_range_tensor(device: torch.device, batch_size: int = 1024):
    global g_cached_range_tensor
    global g_max_range
    if device not in g_cached_range_tensor or batch_size > g_max_range:
        g_max_range = g_max_range if g_max_range > batch_size else batch_size
        g_cached_range_tensor[device] = torch.arange(
            0, g_max_range, step=1, device=device
        )
    return g_cached_range_tensor[device]


class Lora(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        shape: Tuple[int, int],
        config: LoraConfig,
        device: str,
    ):

        super().__init__()

        self.base_layer_ = base_layer
        self.device_ = torch.device(device)
        self.dtype_ = config.dtype_

        self.initializer_ = config.lora_init_
        self.r_ = config.lora_r_
        self.alpha_ = config.lora_alpha_

        if config.use_rslora_:
            self.scaling_ = self.alpha_ / math.sqrt(self.r_)
        else:
            self.scaling_ = self.alpha_ / self.r_

        self.in_features_, self.out_features_ = shape

        assert config.lora_dropout_ > 0.0
        self.dropout_ = nn.Dropout(p=config.lora_dropout_)

        self.lora_a_ = nn.Linear(
            self.in_features_,
            self.r_,
            bias=False,
            dtype=self.dtype_,
            device=self.device_,
        )
        self.lora_b_ = nn.Linear(
            self.r_,
            self.out_features_,
            bias=False,
            dtype=self.dtype_,
            device=self.device_,
        )

        self.use_dora_: bool = config.use_dora_
        self.magnitude_vector_: nn.Parameter = None

    def _get_weight_norm(self) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        weight = dequantize_module_weight(self.base_layer_).to(self.dtype_)
        lora_weight = self.lora_b_.weight @ self.lora_a_.weight
        weight = weight + self.scaling_ * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    def reset_parameters(self, lora_tensor=(None, None)) -> None:
        # if the lora_tensor is not (None, None), use it to init the lora weight
        assert isinstance(lora_tensor, Tuple)
        assert len(lora_tensor) == 2
        assert ((lora_tensor[0] is None) and (lora_tensor[1] is None)) or (
            isinstance(lora_tensor[0], torch.Tensor)
            and isinstance(lora_tensor[1], torch.Tensor)
        )

        if lora_tensor == (None, None):
            if self.initializer_ == "original":
                nn.init.kaiming_uniform_(self.lora_a_.weight, a=math.sqrt(5))
            elif self.initializer_ == "gaussian":
                nn.init.normal_(self.lora_a_.weight, std=1 / self.r_)
            else:
                raise ValueError(f"Unknown initialization {self.initializer_}")
            nn.init.zeros_(self.lora_b_.weight)
        else:
            with torch.no_grad():
                self.lora_a_.weight.copy_(lora_tensor[0])
                self.lora_b_.weight.copy_(lora_tensor[1])

        if self.use_dora_:
            self.magnitude_vector_ = nn.Parameter(
                self._get_weight_norm(), requires_grad=True
            )

    def apply_dora(
        self,
        residual: torch.Tensor,
        result_lora: torch.Tensor,
    ):
        weight_norm = self._get_weight_norm().detach()
        mag_norm_scale = (self.magnitude_vector_ / weight_norm).view(1, -1)
        return mag_norm_scale * residual + mag_norm_scale * result_lora

    def forward(
        self, residual: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        result_lora = (
            self.lora_b_(self.lora_a_(self.dropout_(hidden_states.to(self.dtype_))))
            * self.scaling_
        )
        if self.use_dora_:
            return self.apply_dora(residual, result_lora).to(hidden_states.dtype)
        else:
            return residual + result_lora.to(residual.dtype)


def init_lora_weight(
    base_layer: nn.Module,
    lora_config: LoraConfig,
    lora_tensor=(None, None),
    device: Optional[str] = None,
) -> Lora:
    if not isinstance(base_layer, nn.Linear):
        assert isinstance(base_layer, Linear8bitLt) or isinstance(
            base_layer, Linear4bit
        ), f"Unsupported base layer type '{type(base_layer)}'."

    if isinstance(base_layer, Linear4bit):
        out_dim, in_dim = (
            base_layer.out_features,
            base_layer.in_features,
        )
    else:
        out_dim, in_dim = base_layer.weight.shape

    lora_layer = Lora(
        base_layer,
        (in_dim, out_dim),
        lora_config,
        device if device is not None else base_layer.weight.device,
    )

    lora_layer.reset_parameters(lora_tensor)

    return lora_layer


class Linear(nn.Module):
    def __init__(self, base_layer: nn.Module, lora_layer: Lora) -> None:
        super().__init__()
        self.base_layer_ = base_layer
        self.lora_layer_ = lora_layer

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        result = self.base_layer_(hidden_states)
        return self.lora_layer_(result, hidden_states)
