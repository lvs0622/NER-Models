import torch
import torch.nn as nn
from transformers import BertModel


class FLATModel(nn.Module):
    def __init__(self, bert_model='bert-base-chinese', hidden_size=768, num_labels=10, dropout_rate=0.1):
        super(FLATModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss, logits
        else:
            return logits
