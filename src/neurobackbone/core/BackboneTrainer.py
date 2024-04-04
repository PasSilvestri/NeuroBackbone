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
import neurobackbone.utils as bkb_utils

class BackboneTrainer():
    """Utility class to train and evaluate a model. Each Backbone Trainer instance is uniquly associated with a Backbone model."""

    def __init__(self, model: BackboneModule, optimizer, loss_fn: BackboneFunction, evaluation_fns: List[BackboneFunction], main_metric_name: str = None,
                 hooks: List[BackboneHook] = [], **kwargs):
        """
        Args:
            model (BackboneModule): the model we want to train.
            optimizer (torch.optim): the optimizer used to minimize the sum of the loss functions.
            loss_fn (BackboneFunction): the loss function used to train the model.
            evaluation_fns (List[BackboneFunction]): the evaluation functions used to evaluate the model. The first score is used as metric to evaluate the best model.
            main_metric_name (str, optional): the name of the main metric. Defaults to the first evaluation function.
            hooks (List[BackboneHook], optional): list of hooks to apply to the model. Defaults to [].
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
        self.evaluation_fns = evaluation_fns if isinstance(evaluation_fns,list) else [evaluation_fns]
        
        self.epoch_starts_hooks = []
        self.process_samples_hooks = []
        self.process_output_hooks = []
        self.new_loss_hooks = []
        self.new_score_hooks = []
        self.new_best_score_hooks = []
        for h in hooks: self.add_hook(h)
        
        self.epoch_loss_evolution = []
        self.valid_loss_evolution = []
        self.valid_scores_evolution = {ef.name:[] for ef in self.evaluation_fns}
        self.main_metric = main_metric_name if main_metric_name != None else self.evaluation_fns[0].name
        self.best_scores = {ef.name:0 for ef in self.evaluation_fns}
        
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
            valid_scores_evolution (dict(list(float))): The scores epoch per epoch organized in a dict with the name of the evaluation function as key
            best_scores (dict(float)): The best scores over all epochs organized in a dict with the name of the evaluation function as key
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

            #TODO: Maybe move those hook calls to the evaluate function
            valid_loss, valid_scores = self.evaluate(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)
            self.valid_loss_evolution.append(valid_loss)
            for hook in self.new_loss_hooks: hook(valid_loss,stage="valid")
            
            for score_name, score_value in valid_scores.items():
                self.valid_scores_evolution[score_name].append(score_value)
                for hook in self.new_score_hooks: hook(score_name,score_value,stage="valid")
                if score_value > self.best_scores[score_name]: 
                    self.best_scores[score_name] = score_value
                    for hook in self.new_best_score_hooks: hook(score_name,score_value,stage="valid")
                    if score_name == self.main_metric and path != None:
                        self.save_curent_model_state(path)

            # TODO: add lr scheduler support

            print(f"Epoch: {epoch+1}/{epochs} | Training loss: {epoch_loss:.8f} | Validation loss: {valid_loss:.8f}", flush=True)
            spaces = len(f"Epoch: {epoch+1}/{epochs} ")
            for score_name, score_value in valid_scores.items():
                print(" "*spaces + f"| {score_name}: {score_value:.8f}", flush=True)
    
            if path != None and save_current_graphs:
                bkb_utils.save_losses_graph(path, self.epoch_loss_evolution, self.valid_loss_evolution, filename="current_loss")
                for score_name, score_evolution in self.valid_scores_evolution.items(): bkb_utils.save_score_graph(path,score_evolution,score_name,f"current_{score_name}")
            if self.stop_flag: break # If the stop flag is set, we stop the training

        print(f"Best score ({self.main_metric}): {self.best_scores[self.main_metric]:.4f}")
        if path != None:
            bkb_utils.save_losses_graph(path, self.epoch_loss_evolution, self.valid_loss_evolution, filename="final_loss")
            for score_name, score_evolution in self.valid_scores_evolution.items(): bkb_utils.save_score_graph(path,score_evolution,score_name,f"final_{score_name}")
            if save_current_graphs: 
                os.remove(os.path.join(path,"current_loss.png"))
                for score_name, score_evolution in self.valid_scores_evolution.items(): os.remove(os.path.join(path,f"current_{score_name}.png"))
        return (self.epoch_loss_evolution, self.valid_loss_evolution, self.valid_scores_evolution, self.best_scores)
    
    def evaluate(self, dataset: torch.utils.data.Dataset, batch_size = 32, collate_fn=None, stage="valid"):
        """
        Args:
            valid_dataset (Dataset): the dataset to use to evaluate the model.
            batch_size (int, optional): the size of a single batch. Defaults to 32.
            collate_fn (function, optional): the dataloader collate function. Defaults to None.
            stage (str, optional): the stage of the evaluation. Defaults to "valid". Possible values are "valid" and "test".
        Returns:
            final_loss (float): the average loss over valid_dataset.
            scores (dict(float)): a dictionary of scores computed over the dataset. The keys are the evaluation function names and the values are the scores.
        """
        loss_fn = self.loss_fn if self.loss_fn is not None else lambda y_hat,y: torch.zeros(1)
        total_loss = 0.0
        # valid_dataset.to(self.model.device)
        dataloader =  torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn ) # num_workers=max(os.cpu_count()//2,1)
        self.model.eval()
        valid_scores = {ef.name:0.0 for ef in self.evaluation_fns}
        with torch.no_grad():
            with tqdm(dataloader,
                      desc=f"Evaluating",
                      bar_format="{desc}: |{bar}|{percentage:3.0f}% [{elapsed} ({remaining}), {rate_fmt}{postfix}]") as dataloader_bar:
                for batch_idx, dataset_items in enumerate(dataloader_bar):
                    if self.stop_flag: break # If the stop flag is set, we stop the training
                    
                    if (type(dataset_items) is list or type(dataset_items) is tuple) and len(dataset_items) == 2:  
                        samples, targets = dataset_items
                    else:
                        samples, targets = dataset_items, dataset_items

                    for hook in self.process_samples_hooks: samples, targets = hook(samples,targets,stage=stage)
                    # Moving tensors to the same device of the model
                    if isinstance(samples,torch.Tensor) and samples.device != self.model.device: samples = samples.to(self.model.device)
                    if isinstance(targets,torch.Tensor) and targets.device != self.model.device: targets = targets.to(self.model.device)

                    output = self.model(samples)
                    for hook in self.process_output_hooks: output = hook(output, stage=stage)

                    loss = loss_fn(output, targets)

                    total_loss += loss.item()
                    postfix_dict = {"loss":total_loss/(batch_idx+1)}
                    for evaluation_fn in self.evaluation_fns:
                        valid_scores[evaluation_fn.name] += evaluation_fn(output, targets).item()
                        postfix_dict[evaluation_fn.name] = valid_scores[evaluation_fn.name]/(batch_idx+1)
                    dataloader_bar.set_postfix(postfix_dict)
        
        return total_loss/len(dataloader), {score_name: score_value/len(dataloader) for score_name, score_value in valid_scores.items()}
    
    def test(self, test_dataset: torch.utils.data.Dataset, batch_size = 32, collate_fn=None):
        """
        Args:
            test_dataset (Dataset): the dataset to use to test the model.
            batch_size (int, optional): the size of a single batch. Defaults to 32.
            collate_fn (function, optional): the dataloader collate function. Defaults to None.
        Returns:
            final_loss (float): the average loss over test_dataset.
            scores (dict(float)): a dictionary of scores computed over the dataset. The keys are the evaluation function names and the values are the scores.
        """
        return self.evaluate(test_dataset, batch_size=batch_size, collate_fn=collate_fn, stage="test")
    
    def save_curent_model_state(self, path: str):
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
                "valid_scores_evolution": self.valid_scores_evolution,
                "best_scores": self.best_scores,
                "main_metric": self.main_metric,
                "saving_time": datetime.fromtimestamp(datetime.now().timestamp()).strftime("%d-%m-%Y %H:%M:%S") 
            }
            json.dump(evolution_data,f,indent="\t")
        # Plotting and saving the loss evolution graph and the score evolution graph
        bkb_utils.save_losses_graph(path, self.epoch_loss_evolution, self.valid_loss_evolution, filename="best_model_loss")
        for score_name, score_evolution in self.valid_scores_evolution.items(): bkb_utils.save_score_graph(path,score_evolution,score_name,f"best_model_{score_name}")
    
    def load_trainer_evolution(self,path: str):
        """
        Load the trainer evolution data from a JSON file located at the specified path.

        Args:
            path (str): The path to the directory containing the 'evolution.json' file.
        """
        with open(os.path.join(path,"evolution.json"), "r") as f:
            evolution_data = json.load(f)
            self.epoch_loss_evolution = evolution_data["epoch_loss_evolution"]
            self.valid_loss_evolution = evolution_data["valid_loss_evolution"]
            self.valid_scores_evolution = evolution_data["valid_scores_evolution"]
            self.best_scores = evolution_data["best_scores"]
    
    def stop_training(self):
        """
        Stop the training process. The trainer checks if the training should be stopped at the begging and at the end of each epoch, at the beginning each batch and right after computing the epoch loss (the epoch loss hooks get called).
        """
        self.stop_flag = True
    
    def __graceful_exit(self, save_path: str = None):
        """
        Gracefully exits the training.

        Args:
            save_path (str, optional): The path to save the program state. Defaults to None.
        """
        self.stop_training()
        if save_path != None:
            self.save_curent_model_state(save_path)
            bkb_utils.save_losses_graph(save_path, self.epoch_loss_evolution, self.valid_loss_evolution, filename="final_loss")
            for score_name, score_evolution in self.valid_scores_evolution.items(): bkb_utils.save_score_graph(save_path,score_evolution,score_name,f"final_{score_name}")
        print(f"Best score so far ({self.main_metric}): {self.best_scores[self.main_metric]:.4f}")
            