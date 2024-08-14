import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.activations import ACT2FN

from .config import MixLoraConfig
from .lora_linear import LoraLinear
from .utils import infer_device


def _slice_tensor(
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
    "phi": "_phi_forward",
    "phi3": "_phi3_forward",
}


class MixLoraSparseMoe(torch.nn.Module):
    def __init__(
        self,
        base_layer: torch.nn.Module,
        config: MixLoraConfig,
    ) -> None:
        super().__init__()

        self.dtype_: torch.dtype = config.dtype_
        self.gate_: torch.Tensor = None
        self.base_layer_: torch.nn.Module = base_layer
        self.experts_: Dict[str, LoraLinear] = {}
        self.act_fn_ = (
            ACT2FN[config.act_fn_]
            if isinstance(config.act_fn_, str)
            else config.act_fn_
        )
        self.num_experts_: int = config.num_experts_
        self.topk_: int = config.top_k_
        self.jitter_noise_: float = config.jitter_noise_
        if config.model_type_ not in _compatible_model_types:
            raise NotImplementedError()
        self.forward_fn_ = getattr(self, _compatible_model_types[config.model_type_])

    def _llama_forward(
        self,
        expert_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        input_dtype: torch.dtype,
    ):
        common_gate = self.base_layer_.gate_proj(hidden_states.to(input_dtype)).to(
            hidden_states.dtype
        )
        common_up = self.base_layer_.up_proj(hidden_states.to(input_dtype)).to(
            hidden_states.dtype
        )
        final_expert_states = []
        for expert_idx in range(self.num_experts_):
            _, top_x = torch.where(expert_mask[expert_idx])
            lora_gate: Optional[LoraLinear] = self.experts_.get(
                f"experts.{expert_idx}.gate_proj", None
            )
            lora_down: Optional[LoraLinear] = self.experts_.get(
                f"experts.{expert_idx}.down_proj", None
            )
            lora_up: Optional[LoraLinear] = self.experts_.get(
                f"experts.{expert_idx}.up_proj", None
            )
            if lora_gate is not None:
                lora_data = _slice_tensor(hidden_states, top_x, input_dtype)
                gate_states = lora_gate.lora_forward(
                    _slice_tensor(common_gate, top_x, input_dtype), lora_data
                )
            else:
                lora_data = None
                gate_states = _slice_tensor(common_gate, top_x, input_dtype)

            if lora_up is not None:
                lora_data = _slice_tensor(hidden_states, top_x, input_dtype)
                up_states = lora_up.lora_forward(
                    _slice_tensor(common_up, top_x, input_dtype), lora_data
                )
            else:
                lora_data = None
                up_states = _slice_tensor(common_up, top_x, input_dtype)

            act_result = self.act_fn_(gate_states) * up_states

            if lora_down is not None:
                final_expert_states.append(
                    lora_down.lora_forward(
                        self.base_layer_.down_proj(act_result), act_result
                    )
                )
            else:
                final_expert_states.append(self.base_layer_.down_proj(act_result))

        return final_expert_states

    def _phi_forward(
        self,
        expert_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        input_dtype: torch.dtype,
    ):
        common_fc1 = self.base_layer_.fc1(hidden_states.to(input_dtype)).to(
            hidden_states.dtype
        )
        final_expert_states = []
        for expert_idx in range(self.num_experts_):
            _, top_x = torch.where(expert_mask[expert_idx])
            lora_fc1: Optional[LoraLinear] = self.experts_.get(
                f"experts.{expert_idx}.fc1", None
            )
            lora_fc2: Optional[LoraLinear] = self.experts_.get(
                f"experts.{expert_idx}.fc2", None
            )
            if lora_fc1 is not None:
                lora_data = _slice_tensor(hidden_states, top_x, input_dtype)
                act_result = self.act_fn_(
                    lora_fc1.lora_forward(
                        _slice_tensor(common_fc1, top_x, input_dtype), lora_data
                    )
                )
            else:
                act_result = self.act_fn_(_slice_tensor(common_fc1, top_x, input_dtype))

            if lora_fc2 is not None:
                final_expert_states.append(
                    lora_fc2.lora_forward(self.base_layer_.fc2(act_result), act_result)
                )
            else:
                final_expert_states.append(self.base_layer_.fc2(act_result))

        return final_expert_states

    def _phi3_forward(
        self,
        expert_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        input_dtype: torch.dtype,
    ):
        common_gate_up = self.base_layer_.gate_up_proj(
            hidden_states.to(input_dtype)
        ).to(hidden_states.dtype)
        final_expert_states = []
        for expert_idx in range(self.num_experts_):
            _, top_x = torch.where(expert_mask[expert_idx])
            lora_gate_up: Optional[LoraLinear] = self.experts_.get(
                f"experts.{expert_idx}.gate_up_proj", None
            )
            lora_down: Optional[LoraLinear] = self.experts_.get(
                f"experts.{expert_idx}.down_proj", None
            )
            if lora_gate_up is not None:
                gate_up_states = lora_gate_up.lora_forward(
                    _slice_tensor(common_gate_up, top_x, input_dtype),
                    _slice_tensor(hidden_states, top_x, input_dtype),
                )
            else:
                gate_up_states = _slice_tensor(common_gate_up, top_x, input_dtype)

            gate_states, up_states = gate_up_states.chunk(2, dim=-1)
            act_result = up_states * self.act_fn_(gate_states)

            if lora_down is not None:
                final_expert_states.append(
                    lora_down.lora_forward(
                        self.base_layer_.down_proj(act_result), act_result
                    )
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
        self.gate_ = self.gate_.to(hidden_states)
        router_logits = F.linear(hidden_states, self.gate_)

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
        setattr(
            self_attn,
            proj_name,
            LoraLinear(
                base_layer,
                config,
                (
                    weights[f"{layer_prefix_name}.lora_A.weight"],
                    weights[f"{layer_prefix_name}.lora_B.weight"],
                ),
            ),
        )


def _inject_mlp_module(
    layer_idx: int,
    mlp: torch.nn.Module,
    config: MixLoraConfig,
    weights: Dict[str, torch.Tensor],
):
    moe_layer = MixLoraSparseMoe(mlp, config)
    moe_layer.gate_ = weights[f"mixlora.layers.{layer_idx}.mlp.moe_gate.weight"].to(
        config.dtype_
    )

    if not hasattr(mlp, "mixlora_moes"):
        mlp.mixlora_moes = {}

    mlp.mixlora_moes[config.adapter_name_] = moe_layer
    mlp.forward = moe_layer.forward

    for proj_name, inject in config.target_modules_.items():
        if not inject or not hasattr(mlp, proj_name):
            continue
        base_layer = getattr(mlp, proj_name)
        for expert_idx in range(config.num_experts_):
            layer_prefix_name = (
                f"mixlora.layers.{layer_idx}.mlp.{proj_name}.experts.{expert_idx}"
            )
            moe_layer.experts_[f"experts.{expert_idx}.{proj_name}"] = LoraLinear(
                base_layer,
                config,
                (
                    weights[f"{layer_prefix_name}.lora_A.weight"],
                    weights[f"{layer_prefix_name}.lora_B.weight"],
                ),
            )


def inject_adapter_in_model(
    model: PreTrainedModel,
    config: MixLoraConfig,
    weights: Dict[str, torch.Tensor],
):
    config.model_type_ = model.config.model_type
    model._mixlora_config = config
    for idx, layer in enumerate(model.model.layers):
        _inject_attn_module(idx, layer.self_attn, config, weights)
        _inject_mlp_module(idx, layer.mlp, config, weights)


def load_adapter_weights(
    name_or_path: str,
    adapter_name: str = "default",
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
):
    if not os.path.exists(name_or_path):
        name_or_path = snapshot_download(repo_id=name_or_path, repo_type="model")

    if device is None:
        device = infer_device()

    with open(
        name_or_path + os.sep + "adapter_config.json", "r", encoding="utf8"
    ) as fp:
        config = MixLoraConfig.from_config(json.load(fp))
        config.adapter_name_ = adapter_name
        config.dtype_ = dtype

    config.check()

    weights: Dict[str, torch.Tensor] = torch.load(
        name_or_path + os.sep + "adapter_model.bin", map_location=device
    )

    return config, weights


_compatible_task_types = ["CAUSAL_LM", "QUESTION_ANS"]


@dataclass
class MixLoraModelForCausalLM:
    @staticmethod
    def from_pretrained(
        name_or_path: str,
        *model_args,
        **kwargs,
    ) -> Tuple[PreTrainedModel, MixLoraConfig]:
        config, weights = load_adapter_weights(
            name_or_path,
            adapter_name=kwargs.pop("adapter_name", "default"),
            dtype=kwargs.get("torch_dtype", torch.float32),
        )

        assert config.task_type_ in _compatible_task_types

        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_, *model_args, **kwargs
        )

        inject_adapter_in_model(model, config, weights)

        return model, config
