import json
from typing import List, Dict, Any

def prepare_instruction_dataset(raw_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Format raw data into instruction-following format.
    
    Args:
        raw_data: List of dictionaries containing 'instruction', 'input' (optional), and 'output'.
        
    Returns:
        List of dictionaries with 'prompt' and 'completion' keys.
    """
    formatted_data = []
    
    for item in raw_data:
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')
        
        if not instruction:
            continue
            
        # Format as instruction-following
        prompt = f"### Instruction:\n{instruction}\n"
        if input_text:
            prompt += f"### Input:\n{input_text}\n"
        prompt += "### Response:\n"
        
        formatted_data.append({
            "prompt": prompt,
            "completion": output
        })
    
    return formatted_data

def load_json_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Save data to JSONL format."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
