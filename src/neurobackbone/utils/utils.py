import os, math
from typing import List
import matplotlib.pyplot as plt

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

def save_losses_graph(path: str, train_loss_evolution: List(float), valid_loss_evolution: List(float) = None, filename: str ="loss"):
    """
    Save a graph showing the evolution of training and validation losses over epochs.

    Args:
        path (str): The directory path where the graph will be saved.
        train_loss_evolution (list): List of training losses over epochs.
        valid_loss_evolution (list): List of validation losses over epochs.
        filename (str, optional): The name of the file to save the graph as. Default is "loss". Don't add the file extension.
    """
    plt.figure(figsize=(8,6),dpi=150)
    plt.plot(train_loss_evolution, label="train")
    if valid_loss_evolution is not None: plt.plot(valid_loss_evolution, label="val")
    plt.ylabel("loss")
    plt.xticks(range(1,len(train_loss_evolution)+1, math.ceil(len(valid_loss_evolution)/25)))
    plt.xlabel("epochs")
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(path,filename+".png"))
    plt.close()

def save_score_graph(path: str,  score_evolution: List(float), score_name: str = "score", filename: str ="score"):
    """
    Save a graph showing the evolution of validation scores over epochs.

    Args:
        path (str): The directory path where the graph will be saved.
        score_evolution (list): List of scores over epochs.
        score_name (str, optional): The name of the score. Default is "score".
        filename (str, optional): The name of the file to save the graph as. Default is "score". Don't add the file extension.
    """
    plt.figure(figsize=(8,6),dpi=150)
    plt.plot(score_evolution)
    plt.ylabel(f"{score_name}")
    plt.xlabel("epochs")
    plt.xticks(range(1,len(score_evolution)+1, math.ceil(len(score_evolution)/25)))
    plt.savefig(os.path.join(path,filename+".png"))
    plt.close()
    
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