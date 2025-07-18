# CANNER_dataset.py

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import json


class CANNERDataset(Dataset):
    def __init__(self, file_path, tokenizer, label_map_path, lexicon_path):
        self.tokenizer = tokenizer

        # 加载 label2id
        with open(label_map_path, 'r', encoding='utf-8') as f:
            self.label2id = json.load(f)
        self.num_labels = len(self.label2id)

        # 加载词典
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            self.lexicon = set(w.strip() for w in f if w.strip())

        # 读取数据
        self.samples = self.read_bio_file(file_path)

    def read_bio_file(self, file_path):
        samples = []
        with open(file_path, encoding='utf-8') as f:
            words, labels = [], []
            for line in f:
                line = line.strip()
                if not line:
                    if words:
                        samples.append((words, labels))
                        words, labels = [], []
                else:
                    try:
                        w, l = line.split()
                        words.append(w)
                        labels.append(l)
                    except:
                        continue
            if words:
                samples.append((words, labels))
        return samples

    def find_lexicon_mask(self, tokens):
        word_mask = [0] * len(tokens)
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                word = ''.join(tokens[i:j + 1])
                if word in self.lexicon:
                    for k in range(i, j + 1):
                        word_mask[k] = 1
        return word_mask

    def __getitem__(self, idx):
        tokens, labels = self.samples[idx]
        encoding = self.tokenizer(tokens, is_split_into_words=True, return_tensors=None, add_special_tokens=False)

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        label_ids = [self.label2id.get(l, self.label2id.get("O", 0)) for l in labels]
        word_mask = self.find_lexicon_mask(tokens)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long),
            'word_mask': torch.tensor(word_mask, dtype=torch.float)
        }

    def __len__(self):
        return len(self.samples)


# ⚙️ 自定义 collate_fn：pad input_ids, attention_mask, labels, word_mask
from torch.nn.utils.rnn import pad_sequence


def canner_collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    word_mask = [item['word_mask'] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    word_mask = pad_sequence(word_mask, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'word_mask': word_mask,
    }
