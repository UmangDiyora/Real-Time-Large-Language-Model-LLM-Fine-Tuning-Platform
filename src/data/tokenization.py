from transformers import PreTrainedTokenizer
from typing import Dict, List, Any
import torch

def tokenize_dataset(examples: Dict[str, List[str]], tokenizer: PreTrainedTokenizer, max_length: int = 2048) -> Dict[str, torch.Tensor]:
    """
    Tokenize dataset for causal language modeling.
    
    Args:
        examples: Dictionary containing 'prompt' and 'completion' lists.
        tokenizer: The tokenizer to use.
        max_length: Maximum sequence length.
        
    Returns:
        Dictionary with 'input_ids', 'attention_mask', and 'labels'.
    """
    prompts = examples['prompt']
    completions = examples['completion']
    
    # Combine prompt + completion
    texts = [p + c for p, c in zip(prompts, completions)]
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    input_ids = tokenized['input_ids']
    labels = input_ids.clone()
    
    # Mask prompt tokens in labels so we don't train on them
    for i, prompt in enumerate(prompts):
        prompt_tokens = tokenizer(prompt, truncation=True, max_length=max_length, add_special_tokens=False)['input_ids']
        prompt_len = len(prompt_tokens)
        
        # Ensure we don't mask the entire sequence if prompt is too long (though truncation should handle it)
        if prompt_len < max_length:
             labels[i, :prompt_len] = -100
    
    tokenized['labels'] = labels
    
    return tokenized
