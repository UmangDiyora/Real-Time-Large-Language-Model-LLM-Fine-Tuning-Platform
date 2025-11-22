import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

def create_lora_model(base_model, r=16, lora_alpha=32, lora_dropout=0.05, target_modules=None):
    """
    Wrap a base model with LoRA adapters.
    
    Args:
        base_model: The pre-trained base model.
        r: LoRA attention dimension.
        lora_alpha: The alpha parameter for LoRA scaling.
        lora_dropout: The dropout probability for LoRA layers.
        target_modules: List of module names to apply LoRA to.
        
    Returns:
        The model with LoRA adapters.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
        
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
    return model
