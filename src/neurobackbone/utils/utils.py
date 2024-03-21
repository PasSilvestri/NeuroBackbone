__package__ = "neurobackbone"
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
    
# def saved_model_sorted_list(path,score=None):
#     model_list = []
#     model_dirs = os.listdir(path)
#     for model_dir in model_dirs:
#         model_dir_path = os.path.join(path,model_dir)
#         with open(f"{model_dir_path}/evolution.json","r") as evol_f:
#             evol = json.load(evol_f)
#             best_score = evol.get("best_score",0)
#             score_name = evol.get("score_name",None)
#             if score == None or score_name == None or score==score_name:
#                 model_list.append({
#                     "best_score": best_score,
#                     "name": model_dir
#                 })
#     model_list = sorted(model_list, key=lambda x: x["best_score"], reverse=True)
#     return model_list