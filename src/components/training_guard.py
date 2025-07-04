"""
Training Guard to prevent automatic training execution
"""
import os

class TrainingGuard:
    """Guard to prevent automatic training execution"""
    
    def __init__(self):
        self.auto_training_enabled = os.getenv("AUTO_TRAINING_ENABLED", "false").lower() == "true"
        self.training_in_progress = False
    
    def should_allow_training(self):
        """Check if training should be allowed"""
        return self.auto_training_enabled and not self.training_in_progress
    
    def start_training(self):
        """Mark training as started"""
        self.training_in_progress = True
    
    def stop_training(self):
        """Mark training as stopped"""
        self.training_in_progress = False

# Global instance
training_guard = TrainingGuard()