import json
from typing import List, Dict, Any

def prepare_instruction_dataset(raw_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Format raw data into instruction-following format.
    
    Args:
        raw_data: List of dictionaries containing 'instruction', 'input' (optional), and 'output'.
        
    Returns:
        List of dictionaries with 'prompt' and 'completion'.
    """
    formatted_data = []
    
    for item in raw_data:
        # Format as instruction-following
        prompt = f"### Instruction:\n{item.get('instruction', '')}\n"
        if item.get('input'):
            prompt += f"### Input:\n{item['input']}\n"
        prompt += "### Response:\n"
        
        formatted_data.append({
            "prompt": prompt,
            "completion": item.get('output', '')
        })
    
    return formatted_data

def load_json_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_formatted_dataset(data: List[Dict[str, str]], output_path: str):
    """Save formatted dataset to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
