from transformers import BertModel
from torch import nn

class BertBiLstm(nn.Module):
    def __init__(self, hidden_dim, num_classes=3):
        super(BertBiLstm, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.lstm = nn.LSTM(768, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, token_type_ids, attention_mask):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask)
        out, state = self.lstm(sequence_output[0])
        classifier = self.classifier(out)

        return classifier