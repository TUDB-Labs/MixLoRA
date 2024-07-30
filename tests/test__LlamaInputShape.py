import unittest

import torch

from mixlora.model import LoraLinear, MixLoraConfig, MixLoraSparseMoe


class DummyLlamaMLP(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = torch.nn.SiLU()


config = MixLoraConfig.from_config(
    {
        "bias": "none",
        "peft_type": "MIXLORA",
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
        "routing_strategy": "mixtral",
        "num_experts": 8,
        "act_fn": "silu",
        "top_k": 2,
        "base_model_name_or_path": "DUMMY",
        "task_type": "CAUSAL_LM",
    }
)

config.model_type_ = "llama"

hidden_size = 8
intermediate_size = hidden_size * 2
dummy_mlp = DummyLlamaMLP(hidden_size, intermediate_size)
moe_layer = MixLoraSparseMoe(dummy_mlp, config)
gate_layer = torch.nn.Linear(hidden_size, config.num_experts_, bias=False)
moe_layer.gate_ = gate_layer.weight
mlp_projections = ["gate_proj", "down_proj", "up_proj"]
for proj_name in mlp_projections:
    base_layer: torch.nn.Linear = getattr(dummy_mlp, proj_name)
    torch.nn.init.zeros_(base_layer.weight)
    for expert_idx in range(config.num_experts_):
        moe_layer.experts_[f"experts.{expert_idx}.{proj_name}"] = LoraLinear(
            base_layer, config
        )


class LlamaInputShapeTest(unittest.TestCase):
    def test_forward_with_different_shape(self):
        input_shapes = [
            (2, 8, hidden_size),
            (1, 16, hidden_size),
            (4, 4, hidden_size)
        ]
        
        for shape in input_shapes:
            with self.subTest(shape=shape):
                input = torch.zeros(shape)
                output: torch.Tensor = moe_layer(input)
                self.assertEqual(output.shape, shape)

if __name__ == "__main__":
    unittest.main()

