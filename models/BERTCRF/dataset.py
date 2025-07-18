# bertcrf_dataset.py

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
import json


class BERTCRFDataset(Dataset):
    def __init__(self, data_path, tokenizer, label2id, max_len=128):
        self.sentences = []
        self.labels = []
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

        with open(data_path, encoding='utf-8') as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if not line:
                    if words:
                        self.sentences.append(words)
                        self.labels.append(tags)
                        words, tags = [], []
                else:
                    splits = line.split()
                    words.append(splits[0])
                    tags.append(splits[-1])
            if words:
                self.sentences.append(words)
                self.labels.append(tags)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx]
        tags = self.labels[idx]

        encoding = self.tokenizer(words,
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len,
                                  return_tensors='pt')

        word_ids = encoding.word_ids(batch_index=0)
        labels = [-100] * self.max_len
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                labels[i] = -100
            elif word_idx < len(tags):
                labels[i] = self.label2id.get(tags[word_idx], self.label2id.get("O", 0))

        return {
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'labels': torch.tensor(labels, dtype=torch.long)
        }


def bertcrf_collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
