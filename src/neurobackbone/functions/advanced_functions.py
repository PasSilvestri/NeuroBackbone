import torch
from .functions import *

class BinaryAccuracy(BackboneFunction):
    def __init__(self, name = "BinaryAccuracy", logits = False, threshold = 0.5, **kwargs):
        """
        Initialize the BinaryAccuracy class.
        When called, pass the predictions and targets tensors to the function. Both predictions and targets should be of shape (N,2), (N,1) or (N).         

        Parameters:
            name (str): The name of the function. Default is "BinaryAccuracy".
            logits (bool): Whether the model handles logits or probabilities. If set to True, the predictions are passed through a sigmoid function.
            threshold (float): Decision threshold for computing binary accuracy.
        """
        super().__init__(name = name, function = self.__binary_accuracy, logits = logits, threshold = threshold, **kwargs)
    
    @staticmethod
    def __binary_accuracy(preds: torch.Tensor, targets: torch.Tensor, logits = False, threshold = 0.5, **kwargs):
        """
        Calculate the binary accuracy of predictions compared to targets.

        Args:
            preds (torch.Tensor): The predictions tensor. Possible shapes are (N,2), (N,1) or (N).
            targets (torch.Tensor): The targets tensor. Possible shapes are (N,2), (N,1) or (N).
            logits (bool): Whether the predictions are logits or probabilities (default is False).
            threshold (float): The threshold for binarizing the predictions (default is 0.5).

        Returns:
             (torch.Tensor) A tensor representing the mean binary accuracy.
        """
                
        # Processing of the predictions
        if len(preds.shape) > 1 and preds.shape[-1] > 1:
            # The tensor has shape (N,2)
            preds = preds.argmax(dim=1)
        else:
            # The tensor has shape (N,1) or (N)
            preds = preds.squeeze()
            if logits:
                preds = torch.sigmoid(preds)
            preds = (preds >= threshold).long()
        
        # Processing of the targets
        if len(targets.shape) > 1 and targets.shape[-1] > 1:
            # The tensor has shape (N,2)
            targets = targets.argmax(dim=1)
        else:
            # The tensor has shape (N,1) or (N)
            targets = targets.squeeze().long()
        
        targets.to(preds.device)
        return (preds == targets).float().mean()

class MulticlassAccuracy(BackboneFunction):
    def __init__(self, name = "MulticlassAccuracy", **kwargs):
        """
        Initializes the MulticlassAccuracy class.
        When called, pass the predictions and targets tensors to the function. Possible shapes for both predicitons and targets are: (N,C) with class probabilities or logits or (N,1) or (N) with class indices.
        
        Args:
            name (str): The name of the function. Default is "MulticlassAccuracy".
        """
        super().__init__(name = name, function = self.__multiclass_accuracy, **kwargs)
    
    @staticmethod
    def __multiclass_accuracy(preds: torch.Tensor, targets: torch.Tensor, **kwargs):
        """
        Calculate the multiclass accuracy of predictions compared to targets.

        Args:
            preds (torch.Tensor): The predictions tensor. Possible shapes are: (N,C) with class probabilities or logits or (N,1) or (N) with class indices.
            targets (torch.Tensor): The targets tensor. Possible shapes are: (N,C) with class probabilities or logits or (N,1) or (N) with class indices.

        Returns:
             (torch.Tensor) A tensor representing the mean multiclass accuracy.
        """
        # Processing of the predictions
        if len(preds.shape) > 1 and preds.shape[-1] > 1:
            # The tensor has shape (N,C)
            preds = preds.argmax(dim=1)
        else:
            # The tensor has shape (N,1) or (N), being the indices
            preds = preds.squeeze().long()
        
        # Processing of the targets
        if len(targets.shape) > 1 and targets.shape[-1] > 1:
            # The tensor has shape (N,C)
            targets = targets.argmax(dim=1)
        else:
            # The tensor has shape (N,1) or (N), being the indices
            targets = targets.squeeze().long()
        
        targets.to(preds.device)
        return (preds == targets).float().mean()