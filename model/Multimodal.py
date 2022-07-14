import torch
from torch import nn
from transformers import BertModel
from model.VGG16 import *
from model.bert import *


class MutilModalClassifier(nn.Module):
    def __init__(self, model_type = 'Multimodal', device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(MutilModalClassifier, self).__init__()
        self.model_type = model_type
        self.device = device
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.vgg = VGG16feature()
        self.fc = nn.Sequential(
            nn.Linear(768 + 4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 3),
        )
        self.freeze_bert_param()

    def forward(self, pic, input_ids, token_type_ids, attention_mask):
        if self.model_type == 'Multimodal':
            bert_out = self.bert(input_ids, token_type_ids, attention_mask)
            vgg_feature = self.vgg(pic)
            return self.fc(torch.cat((bert_out.last_hidden_state[:, 0], vgg_feature), dim=1))
        elif self.model_type == 'Picture_model':
            vgg_feature = self.vgg(pic)
            return self.fc(torch.cat((vgg_feature, torch.zeros(size=(pic.shape[0], 768)).to(self.device)), dim=1))
        elif self.model_type == 'Text_model':
            bert_out = self.bert(input_ids, token_type_ids, attention_mask)
            return self.fc(torch.cat((bert_out.last_hidden_state[:, 0], torch.zeros(size=(pic.shape[0], 4096)).to(self.device)), dim=1))
        
    def freeze_bert_param(self):
        for param in self.bert.parameters():
            param.requires_grad_(False)
