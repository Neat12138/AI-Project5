import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from make_dataloader import get_dataloader
import torch.optim as optim
import numpy as np
import argparse

from run_multimodal import run_multimodal
from run_picture import run_picture
from run_text import run_text

# 训练参数
def set_arg():
    parser = argparse.ArgumentParser(description="Multi arg")
    parser.add_argument('--model_type', default='Multimodal')
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--momentum', default=0.90, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    args = parser.parse_args()
    model_type = args.model_type
    LR = args.lr
    momentum = args.momentum
    batch_size = args.batch_size
    return model_type, LR, momentum, batch_size

if __name__ == '__main__':
    model_type, LR, momentum, batch_size = set_arg()
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(batch_size)
    
    if model_type == 'Multimodal':
        run_multimodal(LR, momentum, train_dataloader, valid_dataloader, test_dataloader)
        
    elif model_type == 'Picture_model':
        run_picture(LR, momentum, train_dataloader, valid_dataloader, test_dataloader)
        
    elif model_type == 'Text_model':
        run_text(LR, momentum, train_dataloader, valid_dataloader, test_dataloader)