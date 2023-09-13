#!/usr/bin/env python
#coding: utf-8
import os,sys

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from datasets1 import *
from model import *
from attention_model import *
import argparse

class MyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def getargs(args=sys.argv[1:]):
    parser =argparse.ArgumentParser(description='stock trainning',add_help=True,usage=" ")
    parser.add_argument("--csv",default="NULL",help='the csv file for trainning')
    parser.add_argument("--split_date",default="1930-12-31",help='the date before for trainning')
    parser.add_argument("--model_dir",default="NULL",help='the directory for saving models')
    parser.add_argument("--model",default='LSTM',help='opt for a model')
    args1= parser.parse_args()
    return args1

if __name__ == "__main__":
    args = getargs(sys.argv[1:])
    print(args)
    csv = args.csv
    split_date = args.split_date
    model_dir = args.model_dir
    model_name = args.model

    print(f'csv:{csv}')
    
    dataset_train = dataprocessing(csv=csv,split_date=split_date,mode='train')
    dataset_test,y_test = dataprocessing(csv=csv,split_date=split_date,mode='test')
    _,n_features = dataset_test.shape
    X, y = split_sequences(dataset_train, previous_days, predict_day)
    print ("X.shape" , X.shape)
    print ("y.shape" , y.shape)
    train_X,train_y,test_X,test_y = split_dataset(X,y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'n_features:{n_features}')
    if (model_name == 'LSTM'):
        model = modell(n_features).to(device)
    else:
        model = attention_model(n_features).to(device)
    model.train()
    criterion=nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    num_epochs=1000
    batch_size=4
    dataset = MyDataset(train_X, train_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss=0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss =criterion(outputs,targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}",flush=True)
        file_name = f'A_{epoch}.pth'
        torch.save(model.state_dict(), os.path.join(model_dir, file_name))
