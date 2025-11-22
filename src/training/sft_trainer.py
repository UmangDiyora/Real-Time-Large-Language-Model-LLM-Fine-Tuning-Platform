from transformers import Trainer
import torch

class SFTTrainer(Trainer):
    """
    Custom Trainer for Supervised Fine-Tuning.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation.
        """
        outputs = model(**inputs)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model, inputs):
        """
        Custom training step with logging.
        """
        loss = super().training_step(model, inputs)
        
        # Log custom metrics if needed
        # if self.state.global_step % 10 == 0:
        #     self.log_metrics()
        
        return loss
