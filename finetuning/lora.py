from peft import LoraConfig, get_peft_model
import pytorch_lightning as pl


def lora_finetune(model):
        
        target_modules_base = 'module.encoder.layers'
        num_layers = 6
        target_modules_intermediate = {'self_attn':
                                        {'q_proj', 'k_proj', 'v_proj', 'out_proj'},
                                        'ffn': {'fc1', 'fc2', 'fc_gate'}}

        target_mods = [
            f'{target_modules_base}.{i}.{component}.{sub_component}'
            for i in range(num_layers)
            for component, sub_components in target_modules_intermediate.items()
            for sub_component in sub_components
            ]   
        
        
        lora_config = LoraConfig(r=8, lora_alpha=8,
                                 target_modules=target_mods,
                                 lora_dropout=0.1,
                                 bias="lora_only")
        model = get_peft_model(model, lora_config)
        print('trainable parameters:')
        model.print_trainable_parameters()
        
        # wrap the model in LightningModule
        # model = pl.LightningModule(model)
        return model