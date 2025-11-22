import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

def quantize_model(model_path, output_path):
    print(f"Quantizing model from {model_path}...")
    
    # Load in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Saving quantized model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    
    quantize_model(args.model_path, args.output_path)
