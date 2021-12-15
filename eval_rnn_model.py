import random
import numpy as np
import pandas as pd
from rnn_model import TextDataset, preprocess, RNN, load_rnn_model
import torch
import torch.nn as nn
from tqdm import tqdm


def evaluate(model, data_loader, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """
    Summary: Evaluates RNN model performance on test dataset

    Inputs: model - RNN model to be evaluated
            data_loader - DataLoader object used to pass in testing data
            criterion - 
    """
    print('Evaluating model performance on the test dataset')
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_predictions = []
    for texts, labels in tqdm(data_loader):
        texts = texts.to(device)
        labels = labels.to(device)
        
        output = model(texts)
        acc = accuracy(output, labels)
        pred = output.argmax(dim=1)
        all_predictions.append(pred)
        
        loss = criterion(output, labels)
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    full_acc = 100*epoch_acc/len(data_loader)
    full_loss = epoch_loss/len(data_loader)
    print('[TEST]\t Loss: {:.4f}\t Accuracy: {:.2f}%'.format(full_loss, full_acc))
    predictions = torch.cat(all_predictions)
    return full_acc, full_acc, predictions

if __name__=='__main__':
    model = load_rnn_model()
    THRESHOLD = 5 
    MAX_LEN = 100 
    BATCH_SIZE = 32

    data = pd.read_csv("stock_tweet_data.csv")
    data["Sentiment"] = data["Sentiment"].replace(-1,0)
    train_data = data.to_numpy()
    train_data = [(x[1], preprocess(x[0])) for x in train_data]

    train_Ds = TextDataset(train_data, 'train', THRESHOLD, MAX_LEN)
    train_loader = torch.utils.data.DataLoader(train_Ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    
    text = [[1, 'aapl weekly options gamblers lose'], [1, 'aapl always looking improve'], [0, 'aapl worst trading hour summary last 20 days']]
    text = [(x[0], preprocess(x[1])) for x in text]
    test_Ds = TextDataset(text, 'test', THRESHOLD, MAX_LEN, train_Ds.idx_word, train_Ds.word_idx)
    test_loader = torch.utils.data.DataLoader(test_Ds, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    learning_rate = 5e-4 
    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate)

    for text, labels in test_loader:
        optimizer.zero_grad()
        output = model(text)
        pred = output.argmax(dim=1)
        print(pred, labels)