
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import transformers
import torchmetrics
from datasets import load_dataset
import datasets
import pandas as pd



train = {"sentece": [], "class":[]}
for _, data in pd.read_parquet("./data/inDomainDevFinal.gzip").iterrows():
    train["sentece"].append(data["TargetPT"])
    train["class"].append(data["classe"])


test = {"sentece": [], "class":[]}
for _, data in pd.read_parquet("./data/inDomainTrainFinal.gzip").iterrows():
    test["sentece"].append(data["TargetPT"])
    test["class"].append(data["classe"])
    print(data["classe"])


print(pd.read_parquet("./data/inDomainTrainFinal.gzip").iloc[8000:8010])
input()
class litBertClassifier(pl.LightningModule):
    def __init__(self, vectorsSize, hiddenSize, p=0.5):
        super().__init__()
        self.encoder = nn.Sequential(
        nn.Linear(vectorsSize, hiddenSize),
        nn.Dropout(p),
        nn.ReLU(),
        nn.Linear(hiddenSize, hiddenSize),
        nn.ReLU(),
        nn.Linear(hiddenSize, 2)
        )
    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        z = self.encoder(x)    
        loss = nn.BCELoss(z, y)

        self.log('trainLoss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        z = self.encoder(x)
        print(z)
        
        self.log('valLoss', z)

# data
dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(dataset, [55000, 5000])

train_loader = DataLoader(mnist_train, batch_size=32)
val_loader = DataLoader(mnist_val, batch_size=32)

# model
model = LitAutoEncoder()

# training
trainer = pl.Trainer(gpus=4, num_nodes=8, precision=16, limit_train_batches=0.5)
trainer.fit(model, train_loader, val_loader)
    


#https://huggingface.co/bigscience/bloom  depois
#https://huggingface.co/xlm-roberta-base
#https://huggingface.co/neuralmind/bert-base-portuguese-cased
