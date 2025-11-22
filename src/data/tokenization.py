from typing import Dict, List, Any
import torch
from transformers import PreTrainedTokenizer

def tokenize_dataset(examples: Dict[str, List[str]], tokenizer: PreTrainedTokenizer, max_length: int = 2048) -> Dict[str, torch.Tensor]:
    """
    Tokenize dataset for causal language modeling.
    
    Args:
        examples: Dictionary containing 'prompt' and 'completion' lists.
        tokenizer: Hugging Face tokenizer.
        max_length: Maximum sequence length.
        
    Returns:
        Dictionary with 'input_ids', 'attention_mask', and 'labels'.
    """
    # Add special tokens
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
    
    # Create labels (mask prompt tokens)
    labels = tokenized['input_ids'].clone()
    
    # For each example, find where completion starts
    for i, (prompt, full_text) in enumerate(zip(prompts, texts)):
        prompt_tokens = tokenizer(prompt, truncation=True, max_length=max_length)['input_ids']
        prompt_len = len(prompt_tokens)
        
        # Mask prompt tokens in labels (set to -100)
        # Ensure we don't mask the entire sequence if prompt is too long (though truncation handles this)
        if prompt_len < max_length:
            labels[i, :prompt_len] = -100
        else:
            # If prompt is longer than max_length, the whole thing is masked effectively
            labels[i, :] = -100
            
    tokenized['labels'] = labels
    
    return tokenized
