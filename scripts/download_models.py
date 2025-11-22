import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model(model_name, output_dir):
    print(f"Downloading {model_name} to {output_dir}...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(output_dir)
        
        # Download model
        # Use torch_dtype=torch.float16 to save memory and storage
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        model.save_pretrained(output_dir)
        
        print(f"Successfully downloaded {model_name}")
        
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pre-trained LLMs")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Hugging Face model name")
    parser.add_argument("--output_dir", type=str, default="./models/pretrained", help="Output directory")
    
    args = parser.parse_args()
    
    download_model(args.model_name, args.output_dir)
