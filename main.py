from backbone import seed_everything, BackboneTrainer
from model import ContextEmbeddings, Lstm, Classifier, Model, Model2, Model3, Model4
from model import DefinitionsDataset
from vocab import Vocab
import os
import json
import torch
import torchmetrics
DIR = os.path.realpath(os.path.dirname(__file__))
print("Launched...")

SEED = 1749274
seed_everything(SEED)

with open(f"{DIR}/datasets/frame_vocab.json","r") as f:
    frame_vocab = Vocab.from_serializable(json.load(f))
num_classes = len(frame_vocab)
kwargs = {
    "num_classes": num_classes,
    "transformer_name": "bert-base-uncased",
    "num_linear_layers": 1,
    "dropout": 0.4,

    "num_lstm_layers": 3,
    "bidirecional_lstm": False,

    "loss_select": 1,
    "quant_commitment_cost": 1
}

lr = 0.1
batch_size = 32
epochs = 150
save_path = f"{DIR}/models/"

try:
    dd = DefinitionsDataset(f"{DIR}/datasets/event_definitions_dataset.tsv")
    train_dataset, val_dataset = torch.utils.data.random_split(dd, [0.8,0.2], generator=torch.Generator().manual_seed(SEED))

    model = Model(**kwargs)
    print(f"Training {model.name()}...")
    model.to("cuda")

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = model.loss_fn

    # score_fn = torchmetrics.F1Score(task="multiclass", num_classes=num_classes).to(model.device)
    # score_name = "F1-Score"
    # score_fn = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(model.device)
    # score_name = "Accuracy"
    score_fn = model.eval_fn
    score_name = "Accuracy"

    trainer = BackboneTrainer(model=model, optimizer=optimizer, loss_fn=loss_fn, evaluation_fn=score_fn, score_name=score_name)
    trainer.train(
        train_dataset = train_dataset,
        valid_dataset = val_dataset,
        epochs = epochs, 
        batch_size = batch_size,
        shuffle = True,
        save_path = save_path,
        save_current_graphs=True
    )
    
    print(f"Finished training {model.name()}...")
except KeyboardInterrupt as e:
    pass