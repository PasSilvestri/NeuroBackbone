__package__ = "backbone"
def seed_everything(seed: int = 1749274):
    """
    Set the random seed for reproducibility in Python, NumPy, and PyTorch.
    
    Args:
        seed (int): The seed value to use for random number generation.
    """
    import random, os
    import numpy as np
    import torch
    
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True