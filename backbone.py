import inspect
import sys
import os
import json
import pickle
import math
import random
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt


def seed_everything(seed: int):
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

class BackboneModule(nn.Module):
    def __init__(self):
        super().__init__()

        _,_,_,kwargs = inspect.getargvalues(inspect.currentframe().f_back)
        kwargs.pop("self"); kwargs.pop("__class__")
        kwargs.update(kwargs.pop("kwargs",{}))
        args = kwargs.pop("args",[]) # Those aren't supported, too much trouble for a never occuring case

        self.device = torch.device("cpu")
        self.uuid = f"{int(datetime.now().timestamp()*1000)}"

        # construction data
        self.construction_args = {
            "current": {
                "class": self.__class__.__name__,
                "module": inspect.getmodule(self).__name__,
                "uuid": self.uuid,
                "kwargs": {},
                # "args": []
            },
            "modules": {}
        }

        for key, value in kwargs.items():
            if isinstance(value,BackboneModule):
                self.construction_args["modules"][key] = value
            else:
                self.construction_args["current"]["kwargs"][key] = value
        
    def name(self):
        return f"{self.__class__.__name__}_{self.uuid}"
    
    def to(self, device):
        super().to(device)
        self.device = device
        for val in self._modules.values():
            if isinstance(val,(BackboneModule)):
                val.to(device)
        return self

    def set_uuid(self,uuid):
        self.uuid = uuid
        self.construction_args["current"]["uuid"] = uuid
    
    def serialize(self, path:str=None):
        self.on_save(path)

        out = {
            "current": self.construction_args["current"],
            "modules": {}
        }
        for module_name, module in self.construction_args["modules"].items():
            out["modules"][module_name] = module.serialize()

        return out
    
    def on_save(self,path):
        """Called before the module gets serialized

        Args:
            path (str): the directory where the module has been saved
        """
        return
    
    @classmethod
    def load_module(cls, construction_args):
        kwargs = {}
        kwargs.update(construction_args["current"]["kwargs"])
        for module_name, module_construction_args in construction_args["modules"].items():
            kwargs[module_name] = BackboneModule.get_class(module_construction_args).load_module(module_construction_args)
        module = cls(**kwargs)
        module.set_uuid(construction_args["current"]["uuid"])
        module.on_load()
        return module

    def on_load(self):
        """Called after the module gets loaded from memory"""
        return

    @staticmethod
    def get_class(module_construction_args):
        mod = sys.modules[module_construction_args["current"]["module"]]
        return getattr(mod,module_construction_args["current"]["class"])
    
    def save(self, path=""):
        to_save = self.serialize(path)
        
        with open(os.path.join(path,"model_structure.txt"), "w") as f1:
            f1.write(f"{self}\n")
        try:
            with open(os.path.join(path,"model_arguments.json"), "w") as f2:
                json.dump(to_save, f2, indent="\t")
        except:
            with open(os.path.join(path,"model_arguments.json"), "wb") as f2:
                pickle.dump(to_save, f2)
        return torch.save(self.state_dict(),os.path.join(path,"model.pth"))
    
    @classmethod
    def load(cls, path, strict:bool = True):
        state_dict = torch.load(os.path.join(path,"model.pth"), map_location=torch.device('cpu'))
        construction_args = {
            "current": {
                "class": cls.__name__,
                "module": inspect.getmodule(cls).__name__,
                "uuid": f"{int(datetime.now().timestamp()*1000)}",
                "kwargs": {},
                # "args": []
            },
            "modules": {}
        }
        try:
            with open(os.path.join(path,"model_arguments.json"), "r") as f:
                construction_args = json.load(f) 
        except:
            with open(os.path.join(path,"model_arguments.json"), "rb") as f:
                construction_args = pickle.load(f) 
 
        model = cls.load_module(construction_args)
        model.load_state_dict(state_dict, strict=strict)
        return model

#region trainer
class BackboneTrainer():
    """Utility class to train and evaluate a model."""

    def __init__(self, model: BackboneModule, optimizer, loss_fn,
                 evaluation_fn = lambda y_hat,y: torch.zeros(1), score_name = "Score",
                 process_samples = None, process_output = None, **kwargs):
        """
        Args:
            model (BackboneModule): the model we want to train.
            optimizer (torch.optim): the optimizer used to minimize the sum of the loss functions.
            loss_fn (callable): function that computes the loss for the model. Inputs: (output, target). Output: Tensor value
            evaluation_fn (callable): function that computes the score for the model. Inputs: (output, target). Output: Tensor value
            score_name (str): name of the score used. E.g. "F1-score", "Accuracy", ...
            process_samples (callable): function to process the samples and targets before feeding them to the model. Inputs: (samples, targets). Output: (samples, targets)
            process_output (callable): function to process the output of the model. Inputs: (output). Output: (output)
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.evaluation_fn = evaluation_fn
        self.score_name = score_name
        self.process_samples = process_samples
        self.process_output = process_output
        self.epoch_loss_evolution = []
        self.valid_loss_evolution = []
        self.valid_score_evolution = []
        self.best_score = 0

    def train(self, train_dataset: torch.utils.data.Dataset, valid_dataset: torch.utils.data.Dataset,
              epochs: int = 1, starting_epoch:int = 1, batch_size: int = 32, shuffle: bool = False,
              save_path:str = None, collate_fn=None, save_current_graphs:bool = False):
        """
        Args:
            train_dataset (Dataset): a Dataset instance containing the training instances.
            valid_dataset (Dataset):a Dataset instance used to evaluate learning progress.
            epochs (int, optional): the number of times to iterate over train_dataset, it includes the already done starting_epoch. Defaults to 1.
            starting_epoch (int, optional): the number of already done epochs. Defaults to 0.
            batch_size (int, optional): the size of a single batch. Defaults to 32.
            shuffle (bool, optional): wheter to shuffle or not the batches. Defaults to False.
            save_path (str, optional): The path to save the model to. Defaults to None.
            collate_fn (function, optional): the dataloader collate function. Defaults to None.
            save_current_graphs (bool, optional): whether to save the current loss and score to a png file while training. They get deleted when the training is over

        Returns:
            epoch_loss_evolution (list(float)): The training loss epoch per epoch
            valid_loss_evolution (list(float)): The validation loss epoch per epoch
            valid_score_evolution (list(float)): The score epoch per epoch
            best_score (float): The best score over all epochs
        """
        epochs = max(1,int(epochs))
        batch_size = max(1,int(batch_size))

        path = None
        if save_path != None:
            path = f"{save_path}/{self.model.name()}/"

        # If best score for a previously saved checkpoint is still better, don't save this one
        old_best_score = self.best_score
        if path != None and os.path.exists(path):
            with open(os.path.join(path,"evolution.json"), "r") as f:
                evolution_data = json.load(f)
                old_best_score = evolution_data["best_score"]

        # train_dataset.to(self.model.device)
        # valid_dataset.to(self.model.device)

        dataloader =  torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle ) # num_workers = max(os.cpu_count()-2,1)

        for epoch in range(starting_epoch-1, epochs):

            epoch_loss = 0.0
            self.model.train()

            with tqdm(dataloader,
                      desc=f"Training epoch {epoch+1:0>3}/{epochs:0>3}",
                      bar_format="{desc}: |{bar}|{percentage:3.0f}% [{elapsed} ({remaining}), {rate_fmt}{postfix}]") as dataloader_bar:
                for batch_idx, dataset_items in enumerate(dataloader_bar):
                    if (type(dataset_items) is list or type(dataset_items) is tuple) and len(dataset_items) == 2:  
                        samples, targets = dataset_items
                    else:
                        samples, targets = dataset_items, dataset_items

                    # Moving tensors to the same device of the model
                    if isinstance(samples,torch.Tensor) and samples.device != self.model.device: samples = samples.to(self.model.device)
                    if isinstance(targets,torch.Tensor) and targets.device != self.model.device: targets = targets.to(self.model.device)

                    self.optimizer.zero_grad()

                    if self.process_samples != None: samples, targets = self.process_samples(samples,targets)
                    output = self.model(samples)
                    if self.process_output != None: output = self.process_output(output)

                    loss = self.loss_fn(output, targets)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    dataloader_bar.set_postfix({'loss': epoch_loss/(batch_idx+1)})

            epoch_loss = epoch_loss/len(dataloader)
            self.epoch_loss_evolution.append(epoch_loss)

            valid_loss, valid_score = self.evaluate(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)
            self.valid_loss_evolution.append(valid_loss)
            self.valid_score_evolution.append(valid_score)

            # TODO: add lr scheduler support

            print(f"Epoch: {epoch+1}/{epochs} | Training loss: {epoch_loss:.8f} | Validation loss: {valid_loss:.8f} | {self.score_name}: {valid_score:.8f}\n", flush=True)            

            if valid_score > self.best_score and valid_score > old_best_score:
                self.best_score = valid_score
                if path != None:
                    self.save_curent_model_state(path)
                    
            if path != None and save_current_graphs:
                self.save_evolution_graphs(path,"current_loss.png","current_score.png")

        print(f"Best score: {self.best_score:.4f}")
        if path != None:
            self.save_evolution_graphs(path,"final_loss.png","final_score.png")
            if save_current_graphs: 
                os.remove (os.path.join(path,"current_loss.png"))
                os.remove (os.path.join(path,"current_score.png"))
        return (self.epoch_loss_evolution, self.valid_loss_evolution, self.valid_score_evolution, self.best_score)
    
    def evaluate(self, valid_dataset, batch_size = 32, collate_fn=None):
        """
        Args:
            valid_dataset (Dataset): the dataset to use to evaluate the model.
            collate_fn (function, optional): the dataloader collate function. Defaults to None.
            batch_size (int, optional): the size of a single batch. Defaults to 32.
        Returns:
            final_valid_loss (list(float)): the average validation loss over valid_dataset.
            score (float): the score computed over the validation dataset.
        """
        valid_loss = 0.0
        # valid_dataset.to(self.model.device)
        dataloader =  torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn ) # num_workers=max(os.cpu_count()//2,1)
        self.model.eval()
        valid_score = 0
        with torch.no_grad():
            with tqdm(dataloader,
                      desc=f"Evaluating",
                      bar_format="{desc}: |{bar}|{percentage:3.0f}% [{elapsed} ({remaining}), {rate_fmt}{postfix}]") as dataloader_bar:
                for batch_idx, dataset_items in enumerate(dataloader_bar):
                    if (type(dataset_items) is list or type(dataset_items) is tuple) and len(dataset_items) == 2:  
                        samples, targets = dataset_items
                    else:
                        samples, targets = dataset_items, dataset_items

                    # Moving tensors to the same device of the model
                    if isinstance(samples,torch.Tensor) and samples.device != self.model.device: samples = samples.to(self.model.device)
                    if isinstance(targets,torch.Tensor) and targets.device != self.model.device: targets = targets.to(self.model.device)

                    if self.process_samples != None: samples, targets = self.process_samples(samples,targets)
                    output = self.model(samples)
                    if self.process_output != None: output = self.process_output(output)

                    loss = self.loss_fn(output, targets)

                    valid_loss += loss.item()
                    valid_score += self.evaluation_fn(output, targets).item()
                    dataloader_bar.set_postfix({'loss': valid_loss/(batch_idx+1), f"{self.score_name}":valid_score/(batch_idx+1)})
        
        return valid_loss/len(dataloader), valid_score/len(dataloader)

    def save_evolution_graphs(self, path, filename1 = "loss.png", filename2 = "score.png"):
        plt.figure(figsize=(8,6),dpi=150)
        plt.plot(self.epoch_loss_evolution, label="train")
        plt.plot(self.valid_loss_evolution, label="val")
        plt.ylabel("loss")
        plt.xticks(range(1,len(self.epoch_loss_evolution)+1, math.ceil(len(self.epoch_loss_evolution)/25)))
        plt.xlabel("epochs")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(path,filename1))
        plt.close()

        plt.figure(figsize=(8,6),dpi=150)
        plt.plot(self.valid_score_evolution)
        plt.ylabel(f"{self.score_name}")
        plt.xlabel("epochs")
        plt.xticks(range(1,len(self.valid_score_evolution)+1, math.ceil(len(self.valid_score_evolution)/25)))
        plt.savefig(os.path.join(path,filename2))
        plt.close()

    
    def save_curent_model_state(self, path):
        # if os.path.exists(path):
        #     pass
        os.makedirs(path,exist_ok=True)
        self.model.save(path)
        with open(os.path.join(path,"evolution.json"), "w") as f:
            evolution_data = {
                "epoch_loss_evolution": self.epoch_loss_evolution,
                "valid_loss_evolution": self.valid_loss_evolution,
                "valid_score_evolution": self.valid_score_evolution,
                "best_score": self.best_score,
                "score_name": self.score_name,
                "saving_time": datetime.fromtimestamp(datetime.now().timestamp()).strftime("%d-%m-%Y %H:%M:%S") 
            }
            json.dump(evolution_data,f,indent="\t")
        # Plotting and saving the loss evolution graph and the score evolution graph
        self.save_evolution_graphs(path,"best_model_loss.png","best_model_score.png")
    
    def load_trainer_evolution(self,path):
        with open(os.path.join(path,"evolution.json"), "r") as f:
            evolution_data = json.load(f)
            self.epoch_loss_evolution = evolution_data["epoch_loss_evolution"]
            self.valid_loss_evolution = evolution_data["valid_loss_evolution"]
            self.valid_score_evolution = evolution_data["valid_score_evolution"]
            self.best_score = evolution_data["best_score"]

#endregion

def saved_model_sorted_list(path,score=None):
    model_list = []
    model_dirs = os.listdir(path)
    for model_dir in model_dirs:
        model_dir_path = os.path.join(path,model_dir)
        with open(f"{model_dir_path}/evolution.json","r") as evol_f:
            evol = json.load(evol_f)
            best_score = evol.get("best_score",0)
            score_name = evol.get("score_name",None)
            if score == None or score_name == None or score==score_name:
                model_list.append({
                    "best_score": best_score,
                    "name": model_dir
                })
    model_list = sorted(model_list, key=lambda x: x["best_score"], reverse=True)
    return model_list




