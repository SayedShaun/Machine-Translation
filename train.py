import torch
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from dataloader import CombineDataset
from torch.utils.data import DataLoader
from seq2seq_model import Decoder, Encoder, Seq2Seq
from utils import Utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CombineDataset(
    data_path="Sample Data/english to bengali.csv",
    source_column="english_caption",
    target_column="bengali_caption"
)

input_size = len(dataset.source.vocabs)
output_size = len(dataset.target.vocabs)
embed_size = 100 
hidden_size = 128
n_layers = 2
dropout_rate = 0.4
learning_rate = 0.01
batch_size = 32

encoder = Encoder(
    input_size, 
    embed_size, 
    hidden_size, 
    n_layers, 
    dropout_rate).to(device)

decoder = Decoder(
    output_size, 
    embed_size, 
    hidden_size, 
    n_layers, 
    dropout_rate).to(device)

model = Seq2Seq(
    encoder, 
    decoder, 
    output_size,
    device).to(device)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    collate_fn=dataset.collate_fn,
)

pad_idx = dataset.target.vocabs["<PAD>"]
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = Adam(model.parameters(), lr=learning_rate)

utils = Utils()
epochs = 10

for epoch in tqdm(range(epochs)):
    train_loss = utils.train_fn(
        model,
        loss_fn,
        dataloader,
        optimizer,
        device
    )
    val_loss = utils.evaluate_fn(
        model,
        loss_fn,
        dataloader,
        device
    )
    print("Train Loss", train_loss)
    print("Validation Loss", val_loss)