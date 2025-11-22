class HumanEvaluationSystem:
    def __init__(self):
        self.criteria = ['helpfulness', 'harmlessness', 'honesty', 'relevance', 'coherence']
    
    def create_evaluation_task(self, prompt, responses):
        """
        Create a task for human annotators.
        """
        task = {
            'prompt': prompt,
            'response_a': responses[0],
            'response_b': responses[1],
            'criteria': self.criteria
        }
        return task
    
    def collect_ratings(self, task):
        """
        Placeholder for collecting ratings.
        """
        pass
