import torch
from model.Multimodal import MutilModalClassifier
import torch.optim as optim
import numpy as np


def train_text(model, optimizer, train_dataloader, valid_dataloader, device):  
    cost = torch.nn.CrossEntropyLoss()
    EPOCH = 20  #训练轮数
    train_losses = []
    train_accuracy = []
    valid_losses = []
    valid_accuracy = []
    model.train()  
    for epoch in range(1, EPOCH + 1): 
        train_accuracy_sum, train_loss = 0.0, 0.0
        valid_accuracy_sum, valid_loss = 0.0, 0.0
        train_batch_num = 0
        valid_batch_num = 0
        for batch_pic, *batch_text, batch_y in train_dataloader:
            train_batch_num += 1
            batch_pic = batch_pic.to(device)
            batch_text = [item.to(device) for item in batch_text]
            batch_y = batch_y.to(device)
            
            output = model(batch_pic, *batch_text) 
            loss = cost(output, batch_y)  
            pred = output.data.max(1, keepdim=True)[1] 
            accuracy = pred.eq(batch_y.data.view_as(pred)).cpu().sum()/len(batch_y)
            train_accuracy_sum += accuracy
            train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()  
            optimizer.step()  
            train_losses.append(loss.item())
            train_accuracy.append(accuracy.numpy().tolist())
        
        for batch_pic, *batch_text, batch_y in valid_dataloader:
            valid_batch_num += 1
            batch_pic = batch_pic.to(device)
            batch_text = [item.to(device) for item in batch_text]
            batch_y = batch_y.to(device)
            output = model(batch_pic, *batch_text)  
            loss = cost(output, batch_y)  
            pred = output.data.max(1, keepdim=True)[1] 
            accuracy = pred.eq(batch_y.data.view_as(pred)).cpu().sum()/len(batch_y)
            valid_accuracy_sum += accuracy
            valid_loss += loss.item()
            valid_losses.append(loss.item())
            valid_accuracy.append(accuracy.numpy().tolist())
        
        print('Train Epoch: {}\t train_loss: {}\t train_acc: {}\t valid_loss: {}\t valid_acc: {}'.format(epoch, train_loss/train_batch_num, 
                                                                                                         train_accuracy_sum/train_batch_num, 
                                                                                                         valid_loss/valid_batch_num, 
                                                                                                         valid_accuracy_sum/valid_batch_num))
    
    train_losses = np.array(train_losses)
    np.savetxt('./result/train_text_loss.txt', train_losses)
    train_accuracy = np.array(train_accuracy)
    np.savetxt('./result/train_text_accuracy.txt', train_accuracy)
    valid_losses = np.array(valid_losses)
    np.savetxt('./result/valid_text_loss.txt', valid_losses)
    valid_accuracy = np.array(valid_accuracy)
    np.savetxt('./result/valid_text_accuracy.txt', valid_accuracy)

def run_text(LR, momentum, train_dataloader, valid_dataloader, test_dataloader):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = MutilModalClassifier('Text_model')
        if torch.cuda.is_available():
            model.cuda() 
            print("GPU is using")
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=momentum)
        train_text(model, optimizer, train_dataloader, valid_dataloader, device)