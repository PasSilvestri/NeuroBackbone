import inspect
import sys
import os
import json
import pickle
import math
import random
from datetime import datetime
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from . import BackboneModule
from neurobackbone.hooks import *
from neurobackbone.functions import *

class BackboneTrainer():
    """Utility class to train and evaluate a model. Each Backbone Trainer instance is uniquly associated with a Backbone model."""

    def __init__(self, model: BackboneModule, optimizer, loss_fn: BackboneFunction, evaluation_fn: BackboneFunction,
                 hooks: List[BackboneHook] = [], **kwargs):
        """
        Args:
            model (BackboneModule): the model we want to train.
            optimizer (torch.optim): the optimizer used to minimize the sum of the loss functions.
            loss_fn (callable): function that computes the loss for the model. Inputs: (output, target). Output: Tensor value
            evaluation_fn (callable): function that computes the score for the model. Inputs: (output, target). Output: Tensor value
            score_name (str): name of the score used. E.g. "F1-score", "Accuracy", ...
            hooks (List[BackboneHook], optional): list of hooks to apply to the model. Defaults to [].
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
        self.evaluation_fn = evaluation_fn
        
        self.epoch_starts_hooks = []
        self.process_samples_hooks = []
        self.process_output_hooks = []
        self.new_loss_hooks = []
        self.new_score_hooks = []
        self.new_best_score_hooks = []
        for h in hooks: self.add_hook(h)
        
        self.epoch_loss_evolution = []
        self.valid_loss_evolution = []
        self.valid_score_evolution = []
        self.best_score = 0
        
        self.stop_flag = False
    
    def add_hook(self, hook: BackboneHook):
        """
        Adds a hook to the trainer hook list. Will be called based on the type of the hook

        Parameters:
            hook (BackboneHook): The hook to be added to the corresponding list.
        """
        if not isinstance(hook,BackboneHook) or type(hook) == BackboneHook:
            raise NotImplementedError(f"Unknown hook type {type(hook)}. It must be a subclass of BackboneHook.")
        
        if isinstance(hook,EpochStartHook): self.epoch_starts_hooks.append(hook)
        elif isinstance(hook,PreprocessSamplesHook): self.process_samples_hooks.append(hook)
        elif isinstance(hook,ProcessOutputHook): self.process_output_hooks.append(hook)
        elif isinstance(hook,NewLossHook): self.new_loss_hooks.append(hook)
        elif isinstance(hook,NewScoreHook): self.new_score_hooks.append(hook)
        elif isinstance(hook,NewBestScoreHook): self.new_best_score_hooks.append(hook)
        hook.__attach__(self)
    
    def remove_hook(self, hook: BackboneHook):
        """
        Removes the specified hook from the trainer hook list

        Parameters:
            hook (BackboneHook): The hook to be removed.
        """
        if hook in self.epoch_starts_hooks: self.epoch_starts_hooks.remove(hook)
        elif hook in self.process_samples_hooks: self.process_samples_hooks.remove(hook)
        elif hook in self.process_output_hooks: self.process_output_hooks.remove(hook)
        elif hook in self.new_loss_hooks: self.new_loss_hooks.remove(hook)
        elif hook in self.new_score_hooks: self.new_score_hooks.remove(hook)
        elif hook in self.new_best_score_hooks: self.new_best_score_hooks.remove(hook)
        hook.__detach__()
        
        

    def train(self, train_dataset: torch.utils.data.Dataset, valid_dataset: torch.utils.data.Dataset,
              epochs: int = 1, epochs_done:int = 0, batch_size: int = 32, shuffle: bool = False,
              save_path:str = None, collate_fn=None, save_current_graphs:bool = False):
        """
        Args:
            train_dataset (Dataset): a Dataset instance containing the training instances.
            valid_dataset (Dataset):a Dataset instance used to evaluate learning progress.
            epochs (int, optional): the number of times to iterate over train_dataset, it includes the already done starting_epoch. Defaults to 1.
            epochs_done (int, optional): the number of already done epochs. Defaults to 0.
            batch_size (int, optional): the size of a single batch. Defaults to 32.
            shuffle (bool, optional): wheter to shuffle or not the batches. Defaults to False.
            save_path (str, optional): The folder path to save the model to. Defaults to None.
            collate_fn (function, optional): the dataloader collate function. Defaults to None.
            save_current_graphs (bool, optional): whether to save the current epoch loss and score to a png file while training to visualize the evolution. They get deleted when the training is over (best and final model graphs are still saved). Defaults to False.

        Returns:
            epoch_loss_evolution (list(float)): The training loss epoch per epoch
            valid_loss_evolution (list(float)): The validation loss epoch per epoch
            valid_score_evolution (list(float)): The score epoch per epoch
            best_score (float): The best score over all epochs
        """
        self.stop_flag = False # First reset the stop flag 
        
        epochs = max(1,int(epochs))
        batch_size = max(1,int(batch_size))

        path = None
        if save_path != None:
            path = f"{save_path}/{self.model.name()}/"

        dataloader =  torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle ) # num_workers = max(os.cpu_count()-2,1)

        for epoch in range(epochs_done, epochs):
            if self.stop_flag: break # If the stop flag is set, we stop the training

            epoch_loss = 0.0
            self.model.train()
            for hook in self.epoch_starts_hooks: hook(epoch=epoch,stage="train")
            # if self.stop_flag: break # If the stop flag is set, we stop the training

            with tqdm(dataloader,
                      desc=f"Training epoch {epoch+1:0>3}/{epochs:0>3}",
                      bar_format="{desc}: |{bar}|{percentage:3.0f}% [{elapsed} ({remaining}), {rate_fmt}{postfix}]") as dataloader_bar:
                for batch_idx, dataset_items in enumerate(dataloader_bar):
                    if self.stop_flag: break # If the stop flag is set, we stop the training
                    
                    if (type(dataset_items) is list or type(dataset_items) is tuple) and len(dataset_items) == 2:  
                        samples, targets = dataset_items
                    else:
                        samples, targets = dataset_items, dataset_items

                    for hook in self.process_samples_hooks: samples, targets = hook(samples,targets,stage="train")                    
                    # if self.stop_flag: break # If the stop flag is set, we stop the training
                    
                    # Moving tensors to the same device of the model
                    if isinstance(samples,torch.Tensor) and samples.device != self.model.device: samples = samples.to(self.model.device)
                    if isinstance(targets,torch.Tensor) and targets.device != self.model.device: targets = targets.to(self.model.device)

                    self.optimizer.zero_grad()
                    
                    output = self.model(samples)
                    for hook in self.process_output_hooks: output = hook(output,stage="train")
                    # if self.stop_flag: break # If the stop flag is set, we stop the training

                    loss = self.loss_fn(output, targets)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    dataloader_bar.set_postfix({'loss': epoch_loss/(batch_idx+1)})
            if self.stop_flag: break # If the stop flag is set, we stop the training (this call is to mirror the batch for loop stop onto the epoch for loop)

            epoch_loss = epoch_loss/len(dataloader)
            self.epoch_loss_evolution.append(epoch_loss)
            for hook in self.new_loss_hooks: hook(epoch_loss,stage="train")
            if self.stop_flag: break # If the stop flag is set, we stop the training

            valid_loss, valid_score = self.evaluate(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)
            self.valid_loss_evolution.append(valid_loss)
            self.valid_score_evolution.append(valid_score)
            #TODO: Maybe move those hook calls to the evaluate function
            for hook in self.new_loss_hooks: hook(valid_loss,stage="valid")
            # if self.stop_flag: break # If the stop flag is set, we stop the training
            for hook in self.new_score_hooks: hook(self.evaluation_fn.name,valid_score,stage="valid")
            # if self.stop_flag: break # If the stop flag is set, we stop the training

            # TODO: add lr scheduler support

            print(f"Epoch: {epoch+1}/{epochs} | Training loss: {epoch_loss:.8f} | Validation loss: {valid_loss:.8f} | {self.evaluation_fn.name}: {valid_score:.8f}\n", flush=True)
    
            if path != None and save_current_graphs:
                self.save_evolution_graphs(path,"current_loss.png","current_score.png")
                
            if valid_score > self.best_score:
                self.best_score = valid_score
                if path != None:
                    self.save_curent_model_state(path)
                for hook in self.new_best_score_hooks: hook(self.evaluation_fn.name,valid_score,stage="train")
                # if self.stop_flag: break # If the stop flag is set, we stop the training
                
            if self.stop_flag: break # If the stop flag is set, we stop the training

        print(f"Best score: {self.best_score:.4f}")
        if path != None:
            self.save_evolution_graphs(path,"final_loss.png","final_score.png")
            if save_current_graphs: 
                os.remove(os.path.join(path,"current_loss.png"))
                os.remove(os.path.join(path,"current_score.png"))
        return (self.epoch_loss_evolution, self.valid_loss_evolution, self.valid_score_evolution, self.best_score)
    
    def evaluate(self, valid_dataset: torch.utils.data.Dataset, batch_size = 32, collate_fn=None):
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

                    for hook in self.process_samples_hooks: samples, targets = hook(samples,targets,stage="valid")
                    # Moving tensors to the same device of the model
                    if isinstance(samples,torch.Tensor) and samples.device != self.model.device: samples = samples.to(self.model.device)
                    if isinstance(targets,torch.Tensor) and targets.device != self.model.device: targets = targets.to(self.model.device)

                    output = self.model(samples)
                    for hook in self.process_output_hooks: output = hook(output, stage="valid")

                    loss = self.loss_fn(output, targets)

                    valid_loss += loss.item()
                    valid_score += self.evaluation_fn(output, targets).item()
                    dataloader_bar.set_postfix({'loss': valid_loss/(batch_idx+1), f"{self.evaluation_fn.name}":valid_score/(batch_idx+1)})
        
        return valid_loss/len(dataloader), valid_score/len(dataloader)
    
    def test(self, test_dataset: torch.utils.data.Dataset, batch_size = 32, collate_fn=None):
        loss_fn = self.loss_fn if self.loss_fn is not None else lambda y_hat,y: torch.zeros(1)
        test_loss = 0.0
        # test_dataset.to(self.model.device)
        dataloader =  torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn ) # num_workers=max(os.cpu_count()//2,1)
        self.model.eval()
        test_score = 0
        with torch.no_grad():
            with tqdm(dataloader,
                      desc=f"Testing",
                      bar_format="{desc}: |{bar}|{percentage:3.0f}% [{elapsed} ({remaining}), {rate_fmt}{postfix}]") as dataloader_bar:
                for batch_idx, dataset_items in enumerate(dataloader_bar):
                    if (type(dataset_items) is list or type(dataset_items) is tuple) and len(dataset_items) == 2:  
                        samples, targets = dataset_items
                    else:
                        samples, targets = dataset_items, dataset_items

                    for hook in self.process_samples_hooks: samples, targets = hook(samples,targets,stage="test")
                    # Moving tensors to the same device of the model
                    if isinstance(samples,torch.Tensor) and samples.device != self.model.device: samples = samples.to(self.model.device)
                    if isinstance(targets,torch.Tensor) and targets.device != self.model.device: targets = targets.to(self.model.device)

                    output = self.model(samples)
                    for hook in self.process_output_hooks: output = hook(output, stage="test")

                    loss = loss_fn(output, targets)

                    test_loss += loss.item()
                    test_score += self.evaluation_fn(output, targets).item()
                    dataloader_bar.set_postfix({'loss': test_loss/(batch_idx+1), f"{self.evaluation_fn.name}":test_score/(batch_idx+1)})
        
        return test_loss/len(dataloader), test_score/len(dataloader)


    def save_evolution_graphs(self, path, filename1 = "loss.png", filename2 = "score.png"):
        """
        Save evolution graphs of loss and score to specified path with specified filenames.

        Args:
            path (str): The directory path where the graphs will be saved.
            filename1 (str): The filename for the loss graph. Default is "loss.png".
            filename2 (str): The filename for the score graph. Default is "score.png".
        """
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
        plt.ylabel(f"{self.evaluation_fn.name}")
        plt.xlabel("epochs")
        plt.xticks(range(1,len(self.valid_score_evolution)+1, math.ceil(len(self.valid_score_evolution)/25)))
        plt.savefig(os.path.join(path,filename2))
        plt.close()

    
    def save_curent_model_state(self, path):
        """
        Save the current model state and training evolution data to the specified path.

        Args:
            path (str): The path to save the model state.
        """
        os.makedirs(path,exist_ok=True)
        self.model.save(path)
        with open(os.path.join(path,"evolution.json"), "w") as f:
            evolution_data = {
                "epoch_loss_evolution": self.epoch_loss_evolution,
                "valid_loss_evolution": self.valid_loss_evolution,
                "valid_score_evolution": self.valid_score_evolution,
                "best_score": self.best_score,
                "score_name": self.evaluation_fn.name,
                "saving_time": datetime.fromtimestamp(datetime.now().timestamp()).strftime("%d-%m-%Y %H:%M:%S") 
            }
            json.dump(evolution_data,f,indent="\t")
        # Plotting and saving the loss evolution graph and the score evolution graph
        self.save_evolution_graphs(path,"best_model_loss.png","best_model_score.png")
    
    def load_trainer_evolution(self,path):
        """
        Load the trainer evolution data from a JSON file located at the specified path.

        Args:
            path (str): The path to the directory containing the 'evolution.json' file.
        """
        with open(os.path.join(path,"evolution.json"), "r") as f:
            evolution_data = json.load(f)
            self.epoch_loss_evolution = evolution_data["epoch_loss_evolution"]
            self.valid_loss_evolution = evolution_data["valid_loss_evolution"]
            self.valid_score_evolution = evolution_data["valid_score_evolution"]
            self.best_score = evolution_data["best_score"]
    
    def stop_training(self):
        """
        Stop the training process. The trainer checks if the training should be stopped at the begging and at the end of each epoch, at the beginning each batch and right after computing the epoch loss (the epoch loss hooks get called).
        """
        self.stop_flag = True