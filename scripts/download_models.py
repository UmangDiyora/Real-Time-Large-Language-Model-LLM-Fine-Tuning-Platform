import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def download_model(model_name, cache_dir):
    print(f"Downloading model: {model_name}...")
    
    try:
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer.save_pretrained(os.path.join(cache_dir, model_name.split('/')[-1]))
        
        # Download model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        model.save_pretrained(os.path.join(cache_dir, model_name.split('/')[-1]))
        
        print(f"Successfully downloaded {model_name} to {cache_dir}")
        
    except Exception as e:
        print(f"Error downloading {model_name}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pre-trained LLMs")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Hugging Face model name")
    parser.add_argument("--cache_dir", type=str, default="./models/pretrained", help="Directory to save models")
    
    args = parser.parse_args()
    
    os.makedirs(args.cache_dir, exist_ok=True)
    download_model(args.model_name, args.cache_dir)
