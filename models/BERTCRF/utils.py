import torch
from torch.utils.data import Dataset
from collections import defaultdict

# 标签映射
def build_label_map(path):
    label_set = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith("-DOCSTART-"):
                parts = line.strip().split()
                if len(parts) == 2:
                    label_set.add(parts[1])
    label_list = ['O'] + sorted(label_set - {'O'})
    return {label: idx for idx, label in enumerate(label_list)}, label_list

class NERDataset(Dataset):
    def __init__(self, filepath, tokenizer):
        self.data = []
        self.tokenizer = tokenizer
        self.label2id, self.id2label = build_label_map(filepath)
        self.num_labels = len(self.label2id)

        with open(filepath, encoding='utf-8') as f:
            tokens, labels = [], []
            for line in f:
                if line.strip() == "":
                    if tokens:
                        self.data.append((tokens, labels))
                        tokens, labels = [], []
                else:
                    word, label = line.strip().split()
                    tokens.append(word)
                    labels.append(label)
            if tokens:
                self.data.append((tokens, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, labels = self.data[idx]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        labels = ['O'] + labels + ['O']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attn_mask = [1] * len(input_ids)
        label_ids = [self.label2id.get(l, self.label2id['O']) for l in labels]
        return torch.tensor(input_ids), torch.tensor(attn_mask), torch.tensor(label_ids)

def collate_fn(batch):
    input_ids, attention_mask, labels = zip(*batch)
    max_len = max(len(x) for x in input_ids)
    def pad(x): return torch.nn.functional.pad(x, (0, max_len - len(x)), value=0)
    input_ids = torch.stack([pad(x) for x in input_ids])
    attention_mask = torch.stack([pad(x) for x in attention_mask])
    labels = torch.stack([pad(x) for x in labels])
    return input_ids, attention_mask, labels

class EarlyStopping:
    def __init__(self, patience=3, verbose=False):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, score, model):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
