from transformers import pipeline
import re

class ContentFilter:
    def __init__(self):
        # Placeholder for toxicity model
        # self.toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert")
        self.banned_patterns = [
            r"bad_word_1",
            r"bad_word_2"
        ]
    
    def is_safe(self, text):
        # Check banned patterns
        for pattern in self.banned_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        
        # Check toxicity (mocked for now to avoid downloading model)
        # toxicity_score = self.toxicity_classifier(text)[0]
        # if toxicity_score['label'] == 'toxic' and toxicity_score['score'] > 0.7:
        #     return False
            
        return True
    
    def filter_output(self, text):
        if not self.is_safe(text):
            return "I apologize, but I cannot generate that response."
        return text
