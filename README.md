# NeuroBackbone

NeuroBackbone is a lightweight, flexible Python framework based on Pytorch, for building and training neural networks. It is made to speed up development and offers an easy-to-use implementation of basic ML functionalities like training, testing or saving networks. Built around pytorch, with as few dependencies as possible, is meant for students and hobbyists who don't need all the complex functionalities offered by frameworks like pytorch-lightning, and want to go straight to the point.   
<!-- A easy to use library to speed up development using pytorch. It's not meant for production -->

### Installation

To install the package, download it from this repo and place it in your python libraries folder.
TODO: add to pip

### Usage
```python

import neurobackbone as bkb
import torch
import ...

SEED = 1749274
bkb.seed_everything(SEED)

class Model(bkb.BackboneModule):
    def __init__(self, input_size, output_classes, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(input_size,output_classes)

    def forward(self, input_vec) -> torch.Tensor:
        return self.linear(input_vec)
    
    def loss_fn(self, output, target):
        ...
    def eval_fn(self, output, target):
        ...

# Example usage
dd = MyDataset("dataset.tsv")
train_dataset, val_dataset = torch.utils.data.random_split(dd, [0.8,0.2])

model = Model(input_size = 128, output_classes = 2)
model.to("cuda")

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
loss_fn = model.loss_fn
score_name = "Accuracy"
score_fn = model.eval_fn

data_preprocessing_hook = PreprocessSamplesHook(hook = lambda samples, targets, stage: (samples*2, targets))

trainer = bkb.BackboneTrainer(model=model, optimizer=optimizer, loss_fn=loss_fn, evaluation_fn=score_fn, score_name=score_name, hooks=[data_preprocessing_hook])

earlyStoppingHook = EarlyStoppingValidLossHook(patience=3, margin=0.001)
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

...

loaded_model = Model.load("./checkpoints/Model_1710811967248")
```

### Dependences
- python 3.6 or higher
- PyTorch
- tqdm
- matplotlib

### Documentation
For detailed documentation, refer to the docstrings within the code.

### License
This package is yet to be released under any licenses