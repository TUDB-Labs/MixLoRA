import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from transformers.activations import ACT2FN


@dataclass
class AdapterConfig:
    base_model_: str = None
    task_type_: str = None
    peft_type_: str = None
    adapter_name_: str = None
    model_type_: str = None
    dtype_: torch.dtype = None

    @property
    def base_model_name_or_path(self):
        return self.base_model_

    @property
    def adapter_name(self):
        return self.adapter_name_

    def check(self) -> "AdapterConfig":
        assert isinstance(self.base_model_, str)
        assert isinstance(self.task_type_, str)
        assert isinstance(self.peft_type_, str)

        return self

    @staticmethod
    def from_config(config: Dict[str, any]) -> "AdapterConfig":
        return AdapterConfig(
            base_model_=config["base_model_name_or_path"],
            task_type_=config["task_type"],
            peft_type_=config["peft_type"],
        )

    def export(self) -> Dict[str, any]:
        config = {}
        config["bias"] = "none"
        config["peft_type"] = self.peft_type_
        config["task_type"] = self.task_type_
        config["base_model_name_or_path"] = self.base_model_

        return config


lora_target_modules = {
    # LLaMA names
    "q_proj": False,
    "k_proj": False,
    "v_proj": False,
    "o_proj": False,
    "gate_proj": False,
    "down_proj": False,
    "up_proj": False,
    # Phi names
    "q_proj": False,
    "k_proj": False,
    "v_proj": False,
    "dense": False,
    "fc1": False,
    "fc2": False,
    # Phi3 names
    "qkv_proj": False,
    "o_proj": False,
    "gate_up_proj": False,
    "down_proj": False,
}


@dataclass
class LoraConfig(AdapterConfig):
    # Weight-Decomposed Low-Rank Adaptation
    use_dora_: bool = False
    # Rank-Stabilized LoRA
    # sets the adapter scaling factor to `alpha/math.sqrt(r)`
    use_rslora_: bool = False
    # can be original or gaussian
    lora_init_: str = "original"
    lora_r_: int = None
    lora_alpha_: int = None
    lora_dropout_: float = None
    target_modules_: Dict[str, bool] = None

    def check(self) -> "LoraConfig":
        super().check()
        assert isinstance(self.use_dora_, bool)
        assert isinstance(self.use_rslora_, bool)
        assert isinstance(self.lora_init_, str) and self.lora_init_ in [
            "original",
            "gaussian",
        ]
        assert isinstance(self.lora_r_, int) and self.lora_r_ > 0
        assert isinstance(self.lora_alpha_, int) and self.lora_alpha_ > 0
        assert isinstance(self.lora_dropout_, float) and self.lora_dropout_ >= 0
        assert isinstance(self.target_modules_, Dict)
        for key, value in self.target_modules_.items():
            assert isinstance(key, str) and len(key) > 0
            assert isinstance(value, bool)

        return self

    @staticmethod
    def from_config(config: Dict[str, any]) -> "LoraConfig":
        lora_config = LoraConfig(**AdapterConfig.from_config(config).__dict__)
        lora_config.use_dora_ = config.get("use_dora", False)
        lora_config.use_rslora_ = config.get("use_rslora", False)
        lora_config.lora_init_ = config.get("lora_init", "original")
        lora_config.lora_r_ = config["r"]
        lora_config.lora_alpha_ = config["lora_alpha"]
        lora_config.lora_dropout_ = config["lora_dropout"]
        lora_config.target_modules_ = copy.deepcopy(lora_target_modules)
        if isinstance(config["target_modules"], List):
            for target in config["target_modules"]:
                if target in lora_target_modules:
                    lora_config.target_modules_[target] = True
        elif isinstance(config["target_modules"], Dict):
            for target, value in config["target_modules"].items():
                if target in lora_target_modules:
                    lora_config.target_modules_[target] = value
        else:
            raise ValueError("broken config item: target_modules")

        return lora_config

    def export(self) -> Dict[str, any]:
        config = super().export()
        if self.use_dora_:
            config["use_dora"] = True
        if self.use_rslora_:
            config["use_rslora"] = True
        config["r"] = self.lora_r_
        config["lora_alpha"] = self.lora_alpha_
        config["lora_dropout"] = self.lora_dropout_
        tgt_list = []
        for target, value in self.target_modules_.items():
            if value:
                tgt_list.append(target)
        config["target_modules"] = tgt_list

        return config


available_routing_strategies = ["mixlora"]


@dataclass
class MixLoraConfig(LoraConfig):
    # expert lora
    expert_config_: LoraConfig = None
    # router config
    router_aux_loss_coef_: float = None
    router_init_range_: float = None
    routing_strategy_: str = None
    jitter_noise_: float = None
    router_loss_: bool = True
    num_experts_: int = None
    act_fn_: Optional[Union[str, torch.nn.Module]] = None
    top_k_: int = None

    def check(self) -> "MixLoraConfig":
        super().check()
        if self.expert_config_ is not None:
            self.expert_config_.check()
        assert (
            isinstance(self.router_aux_loss_coef_, float)
            and self.router_aux_loss_coef_ >= 0
        )
        assert (
            isinstance(self.router_init_range_, float) and self.router_init_range_ >= 0
        )
        assert (
            isinstance(self.routing_strategy_, str)
            and self.routing_strategy_ in available_routing_strategies
        )
        assert isinstance(self.jitter_noise_, float) and self.jitter_noise_ >= 0
        assert isinstance(self.router_loss_, bool)
        assert isinstance(self.num_experts_, int) and self.num_experts_ > 0
        assert self.act_fn_ is None or (
            isinstance(self.act_fn_, str) and self.act_fn_ in ACT2FN
        )
        if self.routing_strategy_ == "mixlora":
            assert isinstance(self.top_k_, int) and self.top_k_ > 0
        else:
            raise NotImplementedError()

        return self

    @staticmethod
    def from_config(config: Dict[str, any]) -> "MixLoraConfig":
        lora_config = MixLoraConfig(**LoraConfig.from_config(config).__dict__)
        lora_config.routing_strategy_ = config.get("routing_strategy", None)
        assert (
            lora_config.peft_type_ == "MIXLORA"
            and lora_config.routing_strategy_ is not None
            and lora_config.routing_strategy_ == "mixlora"
        ), "MixLoraConfig only supports MixLoRA models with 'mixlora' routing_strategy."
        if "expert_lora" in config:
            expert_config = copy.deepcopy(config)
            expert_config.update(config["expert_lora"])
            lora_config.expert_config_ = LoraConfig().from_config(expert_config)
        lora_config.router_aux_loss_coef_ = config.get(
            "router_aux_loss_coef", 0.001
        )  # for training
        lora_config.router_loss_ = config.get("router_loss", True)
        lora_config.num_experts_ = config["num_experts"]
        # left blank to automatically use the original act_fn of FFN
        lora_config.act_fn_ = config.get("act_fn", None)
        if lora_config.routing_strategy_ == "mixlora":
            lora_config.router_init_range_ = config.get("router_init_range", 0.02)
            lora_config.jitter_noise_ = config.get("jitter_noise", 0.0)
            lora_config.top_k_ = config.get("top_k", 2)
        else:
            raise NotImplementedError()

        return lora_config

    def export(self) -> Dict[str, any]:
        config = super().export()
        config["peft_type"] = "MIXLORA"
        if self.expert_config_ is not None:
            expert_config = self.expert_config_.export()
            expert_config.pop("peft_type")
            expert_config.pop("target_modules")
            config["expert_lora"] = expert_config
        config["routing_strategy"] = self.routing_strategy_
        config["num_experts"] = self.num_experts_
        if self.act_fn_ is not None and isinstance(self.act_fn_, str):
            config["act_fn"] = self.act_fn_
        if self.routing_strategy_ == "mixlora":
            config["top_k"] = self.top_k_
        else:
            raise NotImplementedError()

        return config

    def expert_config(self, expert_idx: int) -> LoraConfig:
        if self.expert_config_ is None:
            config = copy.deepcopy(super())
        else:
            config = copy.deepcopy(self.expert_config_)
        return config
