import json
import os
from typing import Dict, List, Optional

import fire
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

import mixlora

legacy_proj_names = {
    "w1_proj": "gate_proj",
    "w2_proj": "down_proj",
    "w3_proj": "up_proj",
}

modern_proj_names = {
    "gate_proj": "w1_proj",
    "down_proj": "w2_proj",
    "up_proj": "w3_proj",
}


def from_legacy(name_or_path: str, output_dir: Optional[str] = None):
    if not os.path.exists(name_or_path):
        assert output_dir is not None
        name_or_path = snapshot_download(repo_id=name_or_path, repo_type="model")

    if output_dir is None:
        output_dir = name_or_path

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(
        name_or_path + os.sep + "adapter_config.json", "r", encoding="utf8"
    ) as fp:
        config = json.load(fp)
        assert "routing_strategy" in config and config["routing_strategy"] == "mixtral"
        config["routing_strategy"] = "mixlora"
        target_modules: List[str] = []
        assert isinstance(config["target_modules"], List)
        for target in config["target_modules"]:
            if target in legacy_proj_names:
                target = legacy_proj_names[target]
            if target in mixlora.config.lora_target_modules:
                target_modules.append(target)
        config["target_modules"] = target_modules
        config = mixlora.MixLoraConfig.from_config(config)

    config.check()

    weights: Dict[str, torch.Tensor] = torch.load(
        name_or_path + os.sep + "adapter_model.bin",
        map_location="cpu",
        weights_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    for layer_idx, layer in enumerate(model.model.layers):
        weights[f"mixlora.layers.{layer_idx}.mlp.moe_gate.weight"] = weights.pop(
            f"mixlora.layers.{layer_idx}.gate.weight"
        )
        for proj_name, inject in config.target_modules_.items():
            if not inject or not hasattr(layer.mlp, proj_name):
                continue
            for expert_idx in range(config.num_experts_):
                new_layer_prefix_name = (
                    f"mixlora.layers.{layer_idx}.mlp.{proj_name}.experts.{expert_idx}"
                )
                old_layer_prefix_name = (
                    f"mixlora.layers.{layer_idx}.experts.{expert_idx}.{proj_name}"
                )
                if f"{old_layer_prefix_name}.lora_A.weight" not in weights:
                    old_layer_prefix_name = f"mixlora.layers.{layer_idx}.experts.{expert_idx}.{modern_proj_names[proj_name]}"
                weights[f"{new_layer_prefix_name}.lora_A.weight"] = weights.pop(
                    f"{old_layer_prefix_name}.lora_A.weight"
                )
                weights[f"{new_layer_prefix_name}.lora_B.weight"] = weights.pop(
                    f"{old_layer_prefix_name}.lora_B.weight"
                )

    torch.save(weights, output_dir + os.sep + "adapter_model.bin")

    with open(output_dir + os.sep + "adapter_config.json", "w") as f:
        json.dump(config.export(), f, indent=4)


if __name__ == "__main__":
    fire.Fire(from_legacy)
