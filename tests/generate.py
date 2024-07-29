import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mixlora import MixLoraModel, Prompter


def main(
    base_model: str, instruction: str, lora_weights: str = None, device: str = "cuda:0"
):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model = MixLoraModel.from_pretrained(
        model, lora_weights, device=device, dtype=torch.float16
    )
    prompter = Prompter("alpaca")
    input_ids = tokenizer(
        prompter.generate_prompt(instruction), return_tensors="pt"
    ).input_ids.to(device)
    output = ""
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=100,
        )
        output = tokenizer.batch_decode(
            outputs.detach().cpu().numpy(), skip_special_tokens=True
        )[0][input_ids.shape[-1] :]

        print(output)


if __name__ == "__main__":
    fire.Fire(main)
