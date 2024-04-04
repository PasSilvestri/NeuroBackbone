import neurobackbone as bkb
from neurobackbone.core import BackboneModule, BackboneTrainer
import torch
import torchmetrics
import numpy as np
from sklearn.datasets import make_classification

SEED = 1749274
bkb.utils.seed_everything(SEED)

class Model(BackboneModule):
    def __init__(self, input_size, output_classes, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.output_classes = output_classes
        self.linear = torch.nn.Linear(input_size,output_classes)

    def forward(self, input_vec) -> torch.Tensor:
        return self.linear(input_vec)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = list(map(float, line.strip().split('\t')))
                self.data.append((torch.FloatTensor(parts[:-1]), parts[-1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

input_features = 128
num_classes = 2
X, y = make_classification(n_samples=200, n_features=input_features, n_informative=int(input_features/2), n_redundant=0, n_classes = num_classes, n_clusters_per_class=1)

with open('dataset.tsv', 'w') as f:
    for i in range(X.shape[0]):
        f.write('\t'.join(map(str, X[i])) + '\t' + str(y[i]) + '\n')

# Example usage
dd = MyDataset("dataset.tsv")
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dd, [0.8,0.1,0.1])

model = Model(input_size = input_features, output_classes = num_classes)
model.to("cuda")

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fn = bkb.functions.BackboneFunction("Cross-entropy",torch.nn.functional.cross_entropy)
# score_fn = bkb.functions.BackboneFunction(name = "Accuracy", function = lambda preds, targets, accuracy_fn: accuracy_fn(preds, torch.stack((1-targets, targets), dim=-1).int()),
#                                           accuracy_fn = torchmetrics.Accuracy(task="binary", num_classes=num_classes).to(model.device))
score_fn = bkb.functions.BinaryAccuracy(logits=True)

data_preprocessing_hook = bkb.hooks.PreprocessSamplesHook(hook = lambda samples, targets, stage: (samples, targets.type(torch.int64)))

trainer = BackboneTrainer(model=model, optimizer=optimizer, loss_fn=loss_fn, evaluation_fns=score_fn, hooks=[data_preprocessing_hook])

earlyStoppingHook = bkb.hooks.EarlyStoppingValidLossHook(patience=3, margin=0.001)
trainer.add_hook(earlyStoppingHook)

trainer.train(
    train_dataset = train_dataset,
    valid_dataset = val_dataset,
    epochs = 10, 
    batch_size = 32,
    shuffle = True,
    save_path = "./checkpoints",
    save_current_graphs=True
)

print("-------")
final_loss, scores = trainer.test(test_dataset = test_dataset, batch_size = 32)
print(scores)
loaded_model = Model.load(f"./checkpoints/{model.name()}")