class CostTracker:
    def __init__(self):
        self.costs = {
            'gpu_hour': 1.00,
            'token_input': 0.000001,
            'token_output': 0.000002,
        }
        
        self.usage = {
            'total_requests': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'gpu_hours': 0.0
        }
    
    def track_request(self, input_tokens, output_tokens, duration_seconds):
        self.usage['total_requests'] += 1
        self.usage['total_input_tokens'] += input_tokens
        self.usage['total_output_tokens'] += output_tokens
        self.usage['gpu_hours'] += duration_seconds / 3600
    
    def calculate_costs(self):
        input_cost = self.usage['total_input_tokens'] * self.costs['token_input']
        output_cost = self.usage['total_output_tokens'] * self.costs['token_output']
        gpu_cost = self.usage['gpu_hours'] * self.costs['gpu_hour']
        
        return {
            'input_cost': input_cost,
            'output_cost': output_cost,
            'gpu_cost': gpu_cost,
            'total_cost': input_cost + output_cost + gpu_cost
        }
