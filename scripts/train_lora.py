import argparse
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from src.models.lora_model import create_lora_model
from src.data.preprocessing import load_json_dataset, prepare_instruction_dataset
from src.data.tokenization import tokenize_dataset
from datasets import Dataset

def train(config_path, model_name, data_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load and prepare data
    raw_data = load_json_dataset(data_path)
    formatted_data = prepare_instruction_dataset(raw_data)
    
    # Create Hugging Face Dataset
    hf_dataset = Dataset.from_list(formatted_data)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenize_dataset(examples, tokenizer)
        
    tokenized_datasets = hf_dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "completion"])
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if config["training"]["fp16"] else torch.float32,
        device_map="auto"
    )
    
    # Add LoRA adapters
    model = create_lora_model(
        base_model,
        r=config["lora_config"]["r"],
        lora_alpha=config["lora_config"]["lora_alpha"],
        lora_dropout=config["lora_config"]["lora_dropout"],
        target_modules=config["lora_config"]["target_modules"]
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        fp16=config["training"]["fp16"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        evaluation_strategy="steps",
        eval_steps=config["training"]["eval_steps"],
        warmup_steps=config["training"]["warmup_steps"],
        save_total_limit=3,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets, # Using same for demo, should split
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    trainer.train()
    
    # Save model
    model.save_pretrained(f"{config['training']['output_dir']}/final")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training_configs/lora_config.yaml")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    
    train(args.config, args.model_name, args.data_path)
