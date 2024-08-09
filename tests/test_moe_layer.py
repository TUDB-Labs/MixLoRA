import unittest
from typing import List

import torch
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaMLP
from transformers.models.phi3.modeling_phi3 import Phi3Config, Phi3MLP
from transformers.models.phi.modeling_phi import PhiConfig, PhiMLP

from mixlora.model import LoraLinear, MixLoraConfig, MixLoraSparseMoe


def dummy_moe_layer(
    model_type: str,
    mlp_layer: torch.nn.Module,
    hidden_size: int,
    mlp_projections: List[str],
):
    config = MixLoraConfig.from_config(
        {
            "bias": "none",
            "peft_type": "MIXLORA",
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": [],
            "routing_strategy": "mixlora",
            "num_experts": 8,
            "act_fn": "silu",
            "top_k": 2,
            "base_model_name_or_path": "DUMMY",
            "task_type": "CAUSAL_LM",
        }
    )
    config.model_type_ = model_type
    moe_layer = MixLoraSparseMoe(mlp_layer, config)
    gate_layer = torch.nn.Linear(hidden_size, config.num_experts_, bias=False)
    torch.nn.init.normal_(gate_layer.weight)
    moe_layer.gate_ = gate_layer.weight
    for proj_name in mlp_projections:
        base_layer: torch.nn.Linear = getattr(mlp_layer, proj_name)
        torch.nn.init.normal_(base_layer.weight)
        for expert_idx in range(config.num_experts_):
            moe_layer.experts_[f"experts.{expert_idx}.{proj_name}"] = LoraLinear(
                base_layer, config
            )

    return moe_layer


def dummy_test_shapes(hidden_size: int):
    return [(2, 8, hidden_size), (1, 16, hidden_size), (4, 4, hidden_size)]


hidden_size = 16


class MoeLayerTestCase(unittest.TestCase):
    def test_llama_forward(self):
        mlp_layer = LlamaMLP(
            LlamaConfig(
                vocab_size=128,
                hidden_size=hidden_size,
                intermediate_size=hidden_size * 2,
                num_hidden_layers=8,
                num_attention_heads=2,
            )
        )
        moe_layer = dummy_moe_layer(
            "llama", mlp_layer, hidden_size, ["gate_proj", "down_proj", "up_proj"]
        )
        for shape in dummy_test_shapes(hidden_size):
            with self.subTest(f"test for shape = {shape}"):
                input = torch.zeros(shape)
                output: torch.Tensor = moe_layer(input)
                self.assertEqual(output.shape, shape)

    def test_phi_forward(self):
        mlp_layer = PhiMLP(
            PhiConfig(
                vocab_size=128,
                hidden_size=hidden_size,
                intermediate_size=hidden_size * 2,
                num_hidden_layers=8,
                num_attention_heads=2,
            )
        )
        moe_layer = dummy_moe_layer("phi", mlp_layer, hidden_size, ["fc1", "fc2"])
        for shape in dummy_test_shapes(hidden_size):
            with self.subTest(f"test for shape = {shape}"):
                input = torch.zeros(shape)
                output: torch.Tensor = moe_layer(input)
                self.assertEqual(output.shape, shape)

    def test_phi3_forward(self):
        mlp_layer = Phi3MLP(
            Phi3Config(
                vocab_size=128,
                hidden_size=hidden_size,
                intermediate_size=hidden_size * 2,
                num_hidden_layers=8,
                num_attention_heads=2,
            )
        )
        moe_layer = dummy_moe_layer(
            "phi3", mlp_layer, hidden_size, ["gate_up_proj", "down_proj"]
        )
        for shape in dummy_test_shapes(hidden_size):
            with self.subTest(f"test for shape = {shape}"):
                input = torch.zeros(shape)
                output: torch.Tensor = moe_layer(input)
                self.assertEqual(output.shape, shape)


if __name__ == "__main__":
    unittest.main()
