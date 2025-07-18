# CANNER_model.py

import torch
import torch.nn as nn


class CANNERModel(nn.Module):
    def __init__(self, bert, num_labels, hidden_size=768):
        super().__init__()
        self.bert = bert
        self.char_lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size // 2,
                                 num_layers=1, bidirectional=True, batch_first=True)

        self.word_lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size // 2,
                                 num_layers=1, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)  # char_lstm + word_lstm 拼接

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, char_embeds, word_embeds, attention_mask, labels=None):
        char_out, _ = self.char_lstm(char_embeds)
        word_out, _ = self.word_lstm(word_embeds)

        combined = torch.cat([char_out, word_out], dim=-1)  # [B, L, 2*H]
        logits = self.classifier(self.dropout(combined))     # [B, L, num_labels]

        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss, logits
        else:
            return logits
