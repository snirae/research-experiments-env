import torch
from torch.nn.modules.module import _addindent
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

# neuralforecast models don't use the original pytorch module __repr__
def torch_repr(model):
    # We treat the extra repr like the sub-module, one item per line
    extra_lines = []
    extra_repr = model.extra_repr()
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split('\n')
    child_lines = []
    for key, module in model._modules.items():
        mod_str = repr(module)
        mod_str = _addindent(mod_str, 2)
        child_lines.append('(' + key + '): ' + mod_str)
    lines = extra_lines + child_lines

    main_str = model._get_name() + '('
    if lines:
        # simple one-liner info, which most builtin Modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'

    main_str += ')'
    return main_str
    