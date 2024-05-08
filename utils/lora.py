from peft import LoraConfig, get_peft_model


def lora_finetune(model, lora_config = LoraConfig(r=8, lora_alpha=8,
                                                target_modules="encoder", lora_dropout=0.1,
                                                bias="lora_only")):
        
        model = get_peft_model(model, lora_config)
        print('trainable parameters:')
        model.print_trainable_parameters()
        
        return model