import os
import io
import numpy as np
import torch
from PIL import Image
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

data_path = './data/data'
train_path = './data/train.txt'
test_path = './data/test_without_label.txt'
file_list = os.listdir(data_path)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class MultiDataset(Dataset):
    def __init__(self, data,  is_test=False):
        self.data = data
        self.flag = is_test
        
    def __getitem__(self, index):
        item = (self.data.loc[index]['pic'],
                torch.tensor(self.data.loc[index]['text']['input_ids']),
                torch.tensor(self.data.loc[index]['text']['token_type_ids']),
                torch.tensor(self.data.loc[index]['text']['attention_mask']))
        if not self.flag:
            return *item, self.data.loc[index]['tag']
        else:
            return *item, self.data.loc[index]['guid']

    def __len__(self):
        return len(self.data)

def tag2id(data):
    length = len(data)
    for i in range(length):
        if data['tag'][i] == 'negative':
            data['tag'][i] = 0
        elif data['tag'][i] == 'positive':
            data['tag'][i] = 1
        elif data['tag'][i] == 'neutral':
            data['tag'][i] = 2
    return data

def get_data(data, tokenizer):  
    pic_data = []
    text_data = []
    for item in data['guid']:
        pic_path = os.path.join(data_path, str(item) + ".jpg")
        text_path = os.path.join(data_path, str(item) + ".txt")
        img = Image.open(pic_path)
        text = open(text_path ,'r',encoding='utf-8', errors='ignore').read()
        text2id = tokenizer(text, max_length=128, padding="max_length", truncation=True)

        pic_data.append(np.asarray(img.resize((224,224)), dtype=np.float32).transpose((2, 0 ,1)))
        text_data.append(text2id)

    data['pic'] = pic_data
    data['text'] = text_data
    return data

def get_dataloader(batch_size):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    train_data = tag2id(train_data)

    train_data = get_data(train_data, tokenizer)
    test_data = get_data(test_data, tokenizer)

    train_data, valid_data = train_test_split(train_data, test_size=0.2)

    train_data.reset_index(inplace=True)
    valid_data.reset_index(inplace=True)
    test_data.reset_index(inplace=True)  

    train_dataset = MultiDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    valid_dataset = MultiDataset(valid_data)
    valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle=True)
    test_dataset = MultiDataset(test_data, is_test = True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size)

    return train_dataloader, valid_dataloader, test_dataloader
