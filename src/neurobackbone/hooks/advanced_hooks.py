from .hooks import *

#region advanced hooks
class EarlyStoppingValidLossHook(NewValidLossHook):
    """Hook that stops the training when the validation loss does not improve.
    """
    def __init__(self,patience: int = 3, margin: float = 0.01):
        """
        Args:
            patience (int, optional): number of epochs with no improvement in the validation loss, after which training will be stopped. Defaults to 3.
            margin (float, optional): required margin of improvement for the validation loss. Defaults to 0.01.
        """
        self.patience = max(1,patience)
        self.margin = margin
        self.checks = 0
        self.best_so_far = float("inf")
        hook = lambda loss, stage: self.hook(loss, stage)
        super().__init__(hook)
    
    def hook(self, loss, stage):
        if loss > self.best_so_far-self.margin:
            self.checks += 1
        else:
            self.checks = 0
            self.best_so_far = loss
        if self.checks >= self.patience:
            self.trainer().stop_training()
            
class EarlyStoppingNamedScoreHook(NewNamedScoreHook):
    """Hook that stops the training when a specific score does not improve.
    """
    def __init__(self, score_name, patience: int = 3, margin: float = 0.01):
        """
        Args:
            score_name (str): the name of the score to observe
            patience (int, optional): number of epochs with no improvement in the validation loss, after which training will be stopped. Defaults to 3.
            margin (float, optional): required margin of improvement for the validation loss. Defaults to 0.01.
        """
        self.patience = max(1,patience)
        self.margin = margin
        self.checks = 0
        self.best_so_far = float("inf")
        hook = lambda score_name, score_val, stage: self.hook(score_name, score_val, stage)
        super().__init__(score_name, hook)
    
    def hook(self, score_name, score_val, stage):
        if score_val > self.best_so_far-self.margin:
            self.checks += 1
        else:
            self.checks = 0
            self.best_so_far = score_val
        if self.checks >= self.patience:
            self.trainer().stop_training()

#endregion
