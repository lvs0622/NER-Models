import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class CANNERDataset(Dataset):
    def __init__(self, path, tokenizer, label2id, lexicon, max_len=128):
        self.samples = self.load_bio(path)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.lexicon = lexicon
        self.max_len = max_len

    def load_bio(self, path):
        data = []
        with open(path, encoding='utf-8') as f:
            words, labels = [], []
            for line in f:
                line = line.strip()
                if not line:
                    if words:
                        data.append((words, labels))
                        words, labels = [], []
                else:
                    word, label = line.split()
                    words.append(word)
                    labels.append(label)
            if words:
                data.append((words, labels))
        return data

    def find_lexicon_match(self, tokens):
        word_mask = [0] * len(tokens)
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                word = ''.join(tokens[i:j+1])
                if word in self.lexicon:
                    for k in range(i, j+1):
                        word_mask[k] = 1
        return word_mask

    def __getitem__(self, idx):
        words, labels = self.samples[idx]
        tokens = []
        label_ids = []

        for word, label in zip(words, labels):
            word_tokens = self.tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            label_ids.extend([self.label2id[label]] + [-100] * (len(word_tokens) - 1))

        tokens = tokens[:self.max_len - 2]
        label_ids = label_ids[:self.max_len - 2]

        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        label_ids = [-100] + label_ids + [-100]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        pad_len = self.max_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        label_ids += [-100] * pad_len

        # 模拟 word-level 表征：这里简化为字 embedding 乘以 lexicon mask
        char_embeds = torch.tensor(input_ids).float()  # Placeholder
        word_mask = self.find_lexicon_match(words)
        word_embeds = char_embeds.clone()


        for i, m in enumerate(word_mask[:len(words)]):
            if m == 0:
                word_embeds[i] = 0

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(label_ids),
            'word_mask': torch.tensor(word_mask, dtype=torch.float)  # 用来构造 word_embeds
        }

    def __len__(self):
        return len(self.samples)
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
