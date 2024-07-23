import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import PreTrainedModel
from transformers.activations import ACT2FN

from .adapter import Linear, Lora, init_lora_weight
from .config import MixLoraConfig


def _mixtral_slice_tensor(
    data: torch.Tensor,
    slice: torch.Tensor,
    dtype: torch.dtype,
    last_value: Optional[torch.Tensor] = None,
):
    if last_value is None:
        # for macOS debugging, please uncomment this line
        # assert data.dtype in (torch.float, torch.int, torch.bool)
        return data[None, slice].reshape(-1, data.shape[-1]).to(dtype)
    else:
        return last_value


_compatible_model_types = {
    "llama": "_llama_forward",
    "gemma": "_llama_forward",
    "gemma2": "_llama_forward",
    "qwen2": "_llama_forward",
    "mistral": "_llama_forward",
}


class MixLoraSparseMoe(torch.nn.Module):
    def __init__(
        self,
        base_layer: torch.nn.Module,
        config: MixLoraConfig,
        device: str,
    ) -> None:
        super().__init__()

        self.dtype_: torch.dtype = config.dtype_
        self.gate_ = torch.nn.Linear(
            config.hidden_size_,
            config.num_experts_,
            bias=False,
            device=device,
            dtype=self.dtype_,
        )
        self.base_layer_: torch.nn.Module = base_layer
        self.experts_: Dict[str, Lora] = {}
        self.act_ = ACT2FN[config.act_fn_]
        self.num_experts_: int = config.num_experts_
        self.topk_: int = config.top_k_
        self.jitter_noise_: float = config.jitter_noise_
        if config.model_type_ not in _compatible_model_types:
            raise NotImplementedError()
        self.forward_fn_ = getattr(self, _compatible_model_types[config.model_type_])

    def _llama_forward(
        self, expert_mask: torch.Tensor, hidden_states: torch.Tensor, input_dtype
    ):
        common_w1 = self.base_layer_.gate_proj(hidden_states.to(input_dtype)).to(
            hidden_states.dtype
        )
        common_w3 = self.base_layer_.up_proj(hidden_states.to(input_dtype)).to(
            hidden_states.dtype
        )
        final_expert_states = []
        for expert_idx in range(self.num_experts_):
            _, top_x = torch.where(expert_mask[expert_idx])
            lora_w1: Optional[Lora] = self.experts_.get(
                f"experts.{expert_idx}.gate_proj", None
            )
            lora_w2: Optional[Lora] = self.experts_.get(
                f"experts.{expert_idx}.down_proj", None
            )
            lora_w3: Optional[Lora] = self.experts_.get(
                f"experts.{expert_idx}.up_proj", None
            )
            if lora_w1 is not None:
                lora_data = _mixtral_slice_tensor(hidden_states, top_x, input_dtype)
                w1 = lora_w1(
                    _mixtral_slice_tensor(common_w1, top_x, input_dtype), lora_data
                )
            else:
                lora_data = None
                w1 = _mixtral_slice_tensor(common_w1, top_x, input_dtype)

            if lora_w3 is not None:
                lora_data = _mixtral_slice_tensor(hidden_states, top_x, input_dtype)
                w3 = lora_w3(
                    _mixtral_slice_tensor(common_w3, top_x, input_dtype), lora_data
                )
            else:
                lora_data = None
                w3 = _mixtral_slice_tensor(common_w3, top_x, input_dtype)

            act_result = self.act_(w1) * w3

            if lora_w2 is not None:
                final_expert_states.append(
                    lora_w2(self.base_layer_.down_proj(act_result), act_result)
                )
            else:
                final_expert_states.append(self.base_layer_.down_proj(act_result))

        return final_expert_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        if self.jitter_noise_ > 0:
            # Multiply the token inputs by the uniform distribution - adding some noise
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise_, 1.0 + self.jitter_noise_
            )

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.view(-1, hidden_dim).to(self.dtype_)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate_(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=self.dtype_)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.topk_, dim=-1
        )

        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=self.dtype_,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts_
        ).permute(2, 1, 0)

        # Perform the computation on each expert
        expert_states = self.forward_fn_(
            expert_mask,
            hidden_states,
            input_dtype,
        )

        # Unpack
        for expert_idx in range(self.num_experts_):
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_hidden_states = (
                expert_states[expert_idx] * routing_weights[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(self.dtype_)
            )

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        ).to(input_dtype)

        return final_hidden_states


def _inject_attn_module(
    layer_idx: int,
    self_attn: torch.nn.Module,
    config: MixLoraConfig,
    weights: Dict[str, torch.Tensor],
):
    for proj_name, inject in config.target_modules_.items():
        if not inject or not hasattr(self_attn, proj_name):
            continue
        base_layer = getattr(self_attn, proj_name)
        layer_prefix_name = f"mixlora.layers.{layer_idx}.self_attn.{proj_name}"
        lora_weights = (
            weights[f"{layer_prefix_name}.lora_A.weight"],
            weights[f"{layer_prefix_name}.lora_B.weight"],
        )
        setattr(
            self_attn,
            proj_name,
            Linear(
                base_layer,
                init_lora_weight(
                    base_layer, config, lora_weights, base_layer.weight.device
                ),
            ),
        )


def _inject_mlp_module(
    layer_idx: int,
    mlp: torch.nn.Module,
    config: MixLoraConfig,
    weights: Dict[str, torch.Tensor],
    device: str,
):
    moe_layer = MixLoraSparseMoe(mlp, config, device)
    mlp._mixlora_moe = moe_layer
    mlp.forward = moe_layer.forward
    gate_weight = weights[f"mixlora.layers.{layer_idx}.gate.weight"]
    with torch.no_grad():
        moe_layer.gate_.weight.copy_(gate_weight)
    for proj_name, inject in config.target_modules_.items():
        if not inject or not hasattr(mlp, proj_name):
            continue
        base_layer = getattr(mlp, proj_name)
        for expert_idx in range(config.num_experts_):
            layer_prefix_name = (
                f"mixlora.layers.{layer_idx}.experts.{expert_idx}.{proj_name}"
            )
            lora_weights = (
                weights[f"{layer_prefix_name}.lora_A.weight"],
                weights[f"{layer_prefix_name}.lora_B.weight"],
            )
            moe_layer.experts_[f"experts.{expert_idx}.{proj_name}"] = init_lora_weight(
                base_layer, config, lora_weights, base_layer.weight.device
            )


def inject_pretrained(
    model: PreTrainedModel,
    config: MixLoraConfig,
    weights: Dict[str, torch.Tensor],
    device: str,
):
    config.hidden_size_ = model.config.hidden_size
    config.model_type_ = model.config.model_type
    model._mixlora_config = config
    for idx, layer in enumerate(model.model.layers):
        _inject_attn_module(idx, layer.self_attn, config, weights)
        _inject_mlp_module(idx, layer.mlp, config, weights, device)


def load_adapter_weights(
    name_or_path: str,
    adapter_name: str,
    device: str,
    dtype: torch.dtype,
):
    if not os.path.exists(name_or_path):
        name_or_path = snapshot_download(repo_id=name_or_path, repo_type="model")

    with open(
        name_or_path + os.sep + "adapter_config.json", "r", encoding="utf8"
    ) as fp:
        config = MixLoraConfig(adapter_name_=adapter_name, dtype_=dtype).from_config(
            json.load(fp)
        )

    weights = torch.load(
        name_or_path + os.sep + "adapter_model.bin", map_location=device
    )

    return config, weights


@dataclass
class MixLoraModel:
    @staticmethod
    def from_pretrained(
        model: PreTrainedModel,
        name_or_path: str,
        adapter_name: str = "default",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> PreTrainedModel:
        if device is None:
            device = model.device
        config, weights = load_adapter_weights(
            name_or_path,
            adapter_name,
            device,
            dtype,
        )

        inject_pretrained(model, config, weights, device)

        return model
