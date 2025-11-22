from vllm import LLM, SamplingParams

class vLLMServer:
    def __init__(self, model_path, tensor_parallel_size=1):
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype='float16',
            max_model_len=4096,
            gpu_memory_utilization=0.9
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=512
        )
    
    def generate(self, prompts, sampling_params=None):
        if sampling_params is None:
            sampling_params = self.sampling_params
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        results = []
        for output in outputs:
            results.append({
                'text': output.outputs[0].text,
                'tokens': len(output.outputs[0].token_ids),
                'finish_reason': output.outputs[0].finish_reason
            })
        
        return results
