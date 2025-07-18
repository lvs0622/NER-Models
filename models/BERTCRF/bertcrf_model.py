import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF

class BERTCRF(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(BERTCRF, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden2tag = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        emissions = self.hidden2tag(outputs.last_hidden_state)

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
            return loss
        else:
            prediction = self.crf.decode(emissions, mask=attention_mask.byte())
            return prediction
