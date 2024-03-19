from backbone import *
import os
import json
import random
import numpy as np
import math
import random
from typing import List
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchmetrics
from tqdm import tqdm
import transformers
from transformers import AutoModel, AutoTokenizer

transformers.utils.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


SEED = 1749274
seed_everything(SEED)

#region network_blocks
class ContextEmbeddings(BackboneModule):
    def __init__(self, model_name: str = "xlm-roberta-base", local_model_checkpoint:str = None, fine_tune:bool = False, output_embedding_size:int = 0, sentence_embeddings:bool = False, **kwargs):
        super().__init__()

        self.model_name = model_name
        self.local_model_checkpoint = local_model_checkpoint
        model_checkpoint = local_model_checkpoint if local_model_checkpoint != None else model_name
        self.sentence_embeddings = sentence_embeddings

        self.transformer = AutoModel.from_pretrained(model_checkpoint, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        self.embedding_size = self.transformer.config.hidden_size
        self.pad_token_id = self.tokenizer(self.tokenizer.pad_token, add_special_tokens = False)["input_ids"][0]

        self.fine_tune = fine_tune
        if not fine_tune:
            for param in self.transformer.parameters():
                param.requires_grad = False
                self.transformer.eval()
        
        if output_embedding_size > 0 and output_embedding_size != self.embedding_size:
            self.output_embedding_size = output_embedding_size
            self.reducer = torch.nn.Linear(in_features=self.embedding_size, out_features=self.output_embedding_size)
        else:
            self.output_embedding_size = self.embedding_size
            self.reducer = None

    #region old_code
    # def process_sentences(self,sentences) -> torch.Tensor:
    #     """
    #     Args:
    #         sentences (list(str)): The list of sentences, each sentence is a list of words. Has shape (N,variable sentence_lenght)

    #     Returns:
    #         torch.Tensor: The embedding of the sentences. Has shape (N,sentence_length,embedding_dim)
    #     """
    #     tokens = self.tokenizer(sentences, padding=True, truncation=True, is_split_into_words=False, return_tensors="pt")
    #     tokens.to(self.device)
    #     token_embeddings = self.transformer(**tokens)
    #     token_embeddings = torch.mean(torch.stack(token_embeddings.hidden_states[-4:], dim=0),dim=0) # As done in the paper https://aclanthology.org/2021.emnlp-demo.34.pdf to improve performance
        
    #     full_word_embeddings = [] # shape (batch_size, sequence_length, embedding_dim)
    #     zero_tensor = torch.zeros(token_embeddings.shape[-1],device=self.device)
    #     max_sentence_length = 0
        
    #     for sentence_idx in range(token_embeddings.shape[0]):
    #         word_ids = tokens.word_ids(sentence_idx) #e.g. [None, 1, 2 ,3, 3, 4, 5, 6, 6, 6, None, None, None, ...]
    #         sentence_embeddings_dict = {} # a dict with int keys of lists of word tokens embeddings. E.g. {"1":[emb1], "2":[emb2, emb3], "3":[emb4]}
    #         sentence_embeddings_list = [] # a list of word embeddings. Shape (sequence_length, embedding_dim)
    #         for w_id, embedding in zip(word_ids, token_embeddings[sentence_idx]):
    #             if w_id != None:
    #                 sentence_embeddings_dict.setdefault(w_id,[]).append(embedding) # Putting all embeddings with the same word id togethet to average them

    #         for agregated_word_embeddings in sentence_embeddings_dict.values():
    #             if len(agregated_word_embeddings)>1:
    #                 word_emb = torch.mean(torch.stack(agregated_word_embeddings),dim=0)
    #             else:
    #                 word_emb = agregated_word_embeddings[0]
    #             sentence_embeddings_list.append(word_emb)

    #         max_sentence_length = max(len(sentence_embeddings_list),max_sentence_length)
    #         full_word_embeddings.append(sentence_embeddings_list)

    #     final_word_embeddings = [] # shape (batch_size, max_sequence_length, embedding_dim)
    #     final_attention_masks = [] # shape (batch_size, max_sequence_length). E.g. [[1,1,1,1,0,0,0],[1,1,1,1,1,1,1],[1,1,0,0,0,0,0]]
    #     for sentence_embeddings_list in full_word_embeddings:
    #         sentence_length = len(sentence_embeddings_list)
    #         # Fill with zero-tensors as paddings until max_sentence_length
    #         sentence_embeddings_list = sentence_embeddings_list + [zero_tensor]*(max_sentence_length-sentence_length)
    #         final_word_embeddings.append(torch.stack(sentence_embeddings_list))
            
    #         attention_mask = [1] * sentence_length + [0] * (max_sentence_length-sentence_length)
    #         final_attention_masks.append(attention_mask)

            
    #     tensor_embeddings = torch.stack(final_word_embeddings)
    #     final_attention_masks = torch.as_tensor(final_attention_masks, device=self.device)
    #     return tensor_embeddings, final_attention_masks
    #endregion

    def process_sentences(self,sentences) -> torch.Tensor:
        """
        Args:
            sentences (list(str)): The list of sentences, each sentence is a list of words. Has shape (N,variable sentence_lenght)

        Returns:
            torch.Tensor: The embedding of the sentences. Has shape (N,sentence_length,embedding_dim)
        """
        tokens = self.tokenizer(sentences, padding=True, return_tensors="pt")
        tokens.to(self.device)
        token_embeddings = self.transformer(**tokens)
        token_embeddings = torch.mean(torch.stack(token_embeddings.hidden_states[-4:], dim=0),dim=0) # As done in the paper https://aclanthology.org/2021.emnlp-demo.34.pdf to improve performance
        
        if self.sentence_embeddings:
            return token_embeddings[:,0,:], None # Taking the [CLS] embedding as sentence embedding
            # return self.mean_pooling(token_embeddings,tokens["attention_mask"]), None # Computing the mean of the valid tokens of the sentence as sentence embedding
        else:
            return token_embeddings, tokens["attention_mask"]
   
    
    def forward(self, sentences) -> torch.Tensor:
        if not self.fine_tune:
            with torch.no_grad():
                embs, masks = self.process_sentences(sentences)
        else:
            embs, masks = self.process_sentences(sentences)

        if self.reducer != None:
            embs = self.reducer(embs)

        return embs, masks
    
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return F.normalize(sentence_embeddings, p=2, dim=1)
    
    def on_save(self, path:str):

        # if self.local_model_checkpoint == None:
        #     relative_directory_path = self.model_checkpoint + "_local"

        #     save_dir = os.path.join(path,relative_directory_path)
        #     os.makedirs(save_dir,exist_ok=True)

        #     self.transformer.save_pretrained(save_dir)
        #     self.tokenizer.save_pretrained(save_dir)
        #     self.construction_args["current"]["kwargs"]["local_model_checkpoint"] = relative_directory_path

        return
    
class Lstm(BackboneModule):
    def __init__(self, in_features:int, out_features:int = None, num_lstm_layers:int = 2, bidirecional_lstm:bool = False, dropout:float = 0.2, **kwargs) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features if out_features != None else in_features
        self.bidirectional = bidirecional_lstm
        division_factor = 2 if bidirecional_lstm else 1

        self.lstm = torch.nn.LSTM(
            input_size = in_features,
            hidden_size = self.out_features//division_factor,
            num_layers = num_lstm_layers,
            batch_first = True,
            bidirectional = bidirecional_lstm,
            dropout=dropout,
        )
        self.lstm_dropout = nn.Dropout(p=dropout)

    def forward(self,x):
        tok_embs, (h_n,_) = self.lstm(x)
        # if self.bidirectional: h_T = [ h_n[-2], h_n[-1] ]
        # else: h_T = h_n[-1]
        # embs = torch.cat(h_T,dim=1) # shape (N, embedding_size)
        embs = torch.mean(tok_embs, dim=1) # shape (N, embedding_size)
        embs = self.lstm_dropout(embs)
        return embs
    
class Classifier(torch.nn.Module):
    def __init__(self, in_features:int, num_classes:int, num_layers:int = 2, **kwargs):
        super().__init__()

        layers_list = []
        for _ in range(num_layers-1):
            layers_list.append(torch.nn.Linear(in_features,in_features))
        layers_list.append(torch.nn.Linear(in_features,num_classes))
        self.sequential = torch.nn.Sequential(*layers_list)

    def forward(self,x):
        out = self.sequential(x)
        return out
    
class Classifier2(torch.nn.Module):
    def __init__(self, in_features:int, out_features:int, num_layers:int = 2, dropout:float = 0.4, **kwargs):
        super().__init__()

        self.layers_list = torch.nn.ModuleList()
        head_num = max(0,num_layers-1)
        for _ in range(head_num):
            self.layers_list.append(torch.nn.Linear(in_features,in_features))
            self.layers_list.append(torch.nn.Tanh())
            self.layers_list.append(torch.nn.Dropout(p=dropout))
        if num_layers > 0:
            self.layers_list.append(torch.nn.Linear(in_features,out_features))
            # layers_list.append(torch.nn.Tanh())

        # self.sequential = torch.nn.Sequential(*self.layers_list)

    def forward(self,x):
        for module in self.layers_list:
            x = module(x)
        return x
    
class VectorQuantizer(BackboneModule):
    """
    LOSS 1:
        mse loss: the distance between the encoder output for all the definitions of a spcific frame and
        the quantized frame embedding have to be as close as possible
    LOSS 2:
        codebook loss: mse error to move the quantized embedding vectors towards the encoder output
        commitment loss: mse error to force the encoder to commit to its assigned frame embedding 
    LOSS 3:
        cross entropy to the inverse of the distances between the encoder output and the quantized embeddings
    """

    def __init__(self, num_embeddings, embedding_dim, loss_select = 1, commitment_cost = 0.8):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.loss_select = loss_select
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def loss(self, embs, indices):

        if self.loss_select == 1:
            quant_embs = self.embedding(indices)
            loss = F.mse_loss(embs, quant_embs, reduction="mean")
        elif self.loss_select == 2:
            quant_embs = self.embedding(indices)
            commitment_loss = F.mse_loss(embs, quant_embs.detach(),reduction="mean")
            codebook_loss = F.mse_loss(embs.detach(), quant_embs, reduction="mean")
            loss = codebook_loss + self.commitment_cost * commitment_loss
        elif self.loss_select == 3:
            # distances of embs e from the quantized embs q_e: (q_e - e)^2 = q_e^2 + e^2 - 2(q_e*e)
            distances = (torch.sum(self.embedding.weight**2, dim=1)
                        + torch.sum(embs**2, dim=1, keepdim=True) 
                        - 2 * torch.matmul(embs, self.embedding.weight.t())) # shape (batch, num_embeddings/classes)
            probs = 1/(distances+1e-10)
            loss = F.cross_entropy(probs,indices)

        return loss

    def predict(self, embs):

        # distances of embs e from the quantized embs q_e: (q_e - e)^2 = q_e^2 + e^2 - 2(q_e*e)
        distances = (torch.sum(self.embedding.weight**2, dim=1)
                    + torch.sum(embs**2, dim=1, keepdim=True) 
                    - 2 * torch.matmul(embs, self.embedding.weight.t())) # shape (batch, num_embeddings)
        
        # encoding_indices = F.one_hot(torch.argmin(distances, dim=1), num_classes=self.num_embeddings)
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices

#endregion

#region model
class Model(BackboneModule):
    def __init__(self,  num_classes:int, transformer_name:str = "bert-base-uncased", embedding_size:int = 0, num_lstm_layers=2, bidirecional_lstm:bool = False, dropout=0.4, num_linear_layers:int = 2, **kwargs):
        super().__init__()

        self.contextEmbeddings = ContextEmbeddings(model_name=transformer_name, output_embedding_size=embedding_size)
        self.lstm = Lstm(in_features=self.contextEmbeddings.output_embedding_size, num_lstm_layers=num_lstm_layers, bidirecional_lstm=bidirecional_lstm, dropout=dropout)
        self.classifier = Classifier(in_features=self.lstm.out_features, num_classes=num_classes, num_layers=num_linear_layers)

    def forward(self,defs):
        embs, mask = self.contextEmbeddings(defs) # shape (N, sequence_length, embedding_size)
        embs = self.lstm(embs) # shape (N, embedding_size)
        out = self.classifier(embs) # shape (N, num_classes)
        return out
    
    def predict(self,x):
        with torch.no_grad():
            return torch.argmax(self.forward(x),dim=1)
    
class Model2(BackboneModule):
    def __init__(self,  num_classes:int, transformer_name:str = "bert-base-uncased", embedding_size:int = 0, num_linear_layers:int = 2, dropout:float = 0.4, **kwargs):
        super().__init__()

        self.contextEmbeddings = ContextEmbeddings(model_name=transformer_name, output_embedding_size=embedding_size, sentence_embeddings=True)
        self.classifier = Classifier2(in_features=self.contextEmbeddings.output_embedding_size, out_features=num_classes, num_layers=num_linear_layers, dropout=dropout)

    def forward(self,defs):
        embs, _ = self.contextEmbeddings(defs) # shape (N, embedding_size)
        out = self.classifier(embs) # shape (N, num_classes)
        return out
    
    def predict(self,x):
        with torch.no_grad():
            return torch.argmax(self.forward(x),dim=1)

class Model3(BackboneModule):
    def __init__(self,  num_classes:int, transformer_name:str = "bert-base-uncased", embedding_size:int = 0, num_linear_layers:int = 2, dropout:float = 0.4, loss_select = 1, quant_commitment_cost = 0.8 ,**kwargs):
        super().__init__()

        self.contextEmbeddings = ContextEmbeddings(model_name=transformer_name, output_embedding_size=embedding_size, sentence_embeddings=True)
        self.classifier = Classifier2(in_features=self.contextEmbeddings.output_embedding_size, out_features=self.contextEmbeddings.output_embedding_size, num_layers=num_linear_layers, dropout=dropout)
        self.vec_quant = VectorQuantizer(num_classes,self.contextEmbeddings.output_embedding_size, loss_select=loss_select, commitment_cost=quant_commitment_cost)
        self.score_fn = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self,defs):
        embs, _ = self.contextEmbeddings(defs) # shape (N, embedding_size)
        embs = self.classifier(embs) # shape (N, embedding_size)
        return embs
    
    def predict(self,x):
        with torch.no_grad():
            return self.vec_quant.predict(self.forward(x))
    
    def loss_fn(self, output, target):
        return self.vec_quant.loss(output,target)
    def eval_fn(self, output, target):
        output = self.vec_quant.predict(output)
        return self.score_fn(output,target)

class Model4(BackboneModule):
    def __init__(self,  num_classes:int, transformer_name:str = "bert-base-uncased", embedding_size:int = 0,  num_lstm_layers=2, bidirecional_lstm:bool = False, num_linear_layers:int = 2, dropout:float = 0.4, loss_select = 1, quant_commitment_cost = 0.8 ,**kwargs):
        super().__init__()

        self.contextEmbeddings = ContextEmbeddings(model_name=transformer_name, output_embedding_size=embedding_size, sentence_embeddings=False)
        self.lstm = Lstm(in_features=self.contextEmbeddings.output_embedding_size, num_lstm_layers=num_lstm_layers, bidirecional_lstm=bidirecional_lstm, dropout=dropout)
        self.classifier = Classifier2(in_features=self.contextEmbeddings.output_embedding_size, out_features=self.contextEmbeddings.output_embedding_size, num_layers=num_linear_layers, dropout=dropout)
        self.vec_quant = VectorQuantizer(num_classes, self.contextEmbeddings.output_embedding_size, loss_select=loss_select, commitment_cost=quant_commitment_cost)
        self.score_fn = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self,defs):
        embs, mask = self.contextEmbeddings(defs) # shape (N, sequence_length, embedding_size)
        embs = self.lstm(embs) # shape (N, embedding_size)
        embs = self.classifier(embs) # shape (N, embedding_size)
        return embs
    
    def predict(self,x):
        with torch.no_grad():
            return self.vec_quant.predict(self.forward(x))
    
    def loss_fn(self, output, target):
        return self.vec_quant.loss(output,target)
    def eval_fn(self, output, target):
        output = self.vec_quant.predict(output)
        return self.score_fn(output,target)
#endregion

#region dataset
class DefinitionsDataset(torch.utils.data.Dataset):
    def __init__(self,filepath):
        self.definitions = []
        self.frame_indices = []
        with open(filepath, "r") as f:
            for line in f:
                definition,frame_idx = line.strip().split("\t")
                self.definitions.append(definition)
                self.frame_indices.append(int(frame_idx))

    def __len__(self):
        return len(self.definitions)
    
    def __getitem__(self, index):
        return (self.definitions[index],self.frame_indices[index])
#endregion