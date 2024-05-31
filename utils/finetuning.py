import torch
from peft import LoraConfig, LoraModel


def find_linear_module_names(model):
    linear_module_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_module_names.append(name)
    return linear_module_names

def add_lora(model, task_type = "SEQ_2_SEQ_LM", r=8, lora_alpha=16, lora_dropout=0.1):
    lora_config = LoraConfig(
            task_type=task_type,
            target_modules=find_linear_module_names(model),
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
    model = LoraModel(model, lora_config, "default")

    return model

def model_printer(model):
    for name, module in model.named_modules():
        print(module)
    