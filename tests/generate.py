from typing import Optional

import fire
import torch
from transformers import AutoTokenizer
from transformers.utils import is_torch_bf16_available_on_device

from mixlora import MixLoraModelForCausalLM, Prompter
from mixlora.utils import infer_device


def main(
    adapter_model: str,
    instruction: str,
    template: str = "alpaca",
    device: Optional[str] = None,
):
    if device is None:
        device = infer_device()

    model, config = MixLoraModelForCausalLM.from_pretrained(
        adapter_model,
        torch_dtype=(
            torch.bfloat16
            if is_torch_bf16_available_on_device(device)
            else torch.float16
        ),
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    prompter = Prompter(template)

    input_kwargs = tokenizer(prompter.generate_prompt(instruction), return_tensors="pt")
    # send tensors into correct device
    for key, value in input_kwargs.items():
        if isinstance(value, torch.Tensor):
            input_kwargs[key] = value.to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **input_kwargs,
            max_new_tokens=100,
        )
        output = tokenizer.batch_decode(
            outputs.detach().cpu().numpy(), skip_special_tokens=True
        )[0][input_kwargs["input_ids"].shape[-1] :]

        print(f"\nOutput: {prompter.get_response(output)}\n")


if __name__ == "__main__":
    fire.Fire(main)
