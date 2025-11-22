import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType

def create_qlora_model(model_name, r=64, lora_alpha=16, lora_dropout=0.05, target_modules=None):
    """
    Load a model with 4-bit quantization and wrap it with LoRA adapters (QLoRA).
    
    Args:
        model_name: Name or path of the pre-trained model.
        r: LoRA attention dimension.
        lora_alpha: The alpha parameter for LoRA scaling.
        lora_dropout: The dropout probability for LoRA layers.
        target_modules: List of module names to apply LoRA to.
        
    Returns:
        The QLoRA model.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # Load model in 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Add adapters
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model
