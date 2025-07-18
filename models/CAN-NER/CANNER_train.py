# main_canner_train.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel, AdamW
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score

from CANNER_model import CANNERModel


class BioDataset(Dataset):
    def __init__(self, data_path, tokenizer, label2id, max_len=128):
        self.sentences = []
        self.labels = []
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

        # 读取BIO数据，格式：每行“word label”，句子间空行分隔
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
            # 加最后一个句子
            if words:
                self.sentences.append(words)
                self.labels.append(tags)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx]
        tags = self.labels[idx]

        # tokenizer编码，返回input_ids, attention_mask等
        encoding = self.tokenizer(words,
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len,
                                  return_tensors='pt')

        # 获取token对应的word索引
        offset_mapping = encoding.offset_mapping[0]  # shape: [max_len, 2]
        word_ids = encoding.word_ids(batch_index=0)  # list of word idx per token

        labels = [-100] * self.max_len  # 初始都ignore
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                labels[i] = -100
            else:
                # word对应的标签id
                if word_idx < len(tags):
                    labels[i] = self.label2id.get(tags[word_idx], 0)

        # 返回字嵌入输入为BERT的input_ids，词嵌入（这里先用char_embed模拟，word_embed后面用外部词典可改）
        return {
            'input_ids': encoding.input_ids[0],
            'attention_mask': encoding.attention_mask[0],
            'labels': torch.tensor(labels),
            'words': words,
            'tags': tags,
        }


def collate_fn(batch):
    input_ids = torch.stack([x['input_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])
    labels = torch.stack([x['labels'] for x in batch])
    return input_ids, attention_mask, labels


def get_label_map(labels):
    unique_labels = set()
    for label_list in labels:
        unique_labels.update(label_list)
    unique_labels = sorted(list(unique_labels))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in tqdm(dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # BERT生成char_embeds
        with torch.no_grad():
            outputs = model.bert(input_ids, attention_mask=attention_mask)
            char_embeds = outputs.last_hidden_state  # [B, L, H]

        # 这里word_embeds暂时用char_embeds替代演示，可按需求改进
        word_embeds = char_embeds

        optimizer.zero_grad()
        loss, logits = model(char_embeds, word_embeds, attention_mask, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, id2label):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model.bert(input_ids, attention_mask=attention_mask)
            char_embeds = outputs.last_hidden_state
            word_embeds = char_embeds

            logits = model(char_embeds, word_embeds, attention_mask)
            preds = torch.argmax(logits, dim=-1)

            # 转换为标签字符串，忽略-100
            for pred_seq, label_seq in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                pred_labels = []
                true_labels = []
                for p, l in zip(pred_seq, label_seq):
                    if l == -100:
                        continue
                    pred_labels.append(id2label.get(p, 'O'))
                    true_labels.append(id2label.get(l, 'O'))
                all_preds.append(pred_labels)
                all_labels.append(true_labels)

    f1 = f1_score(all_labels, all_preds)
    print(classification_report(all_labels, all_preds))
    return f1


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="/root/autodl-tmp/data/train.txt")
    parser.add_argument("--dev_path", type=str, default="/root/autodl-tmp/data/dev.txt")
    parser.add_argument("--test_path", type=str, default="/root/autodl-tmp/data/test.txt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    bert_model = BertModel.from_pretrained('bert-base-chinese')

    # 读取标签
    def read_labels(file):
        labels = [[]]  # 预置第一个句子标签列表
        with open(file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    labels.append([])  # 新句子开始
                else:
                    labels[-1].append(line.split()[-1])
        # 可能最后多了个空的列表，去掉
        if len(labels[-1]) == 0:
            labels.pop()
        return labels

    train_labels = read_labels(args.train_path)
    label2id, id2label = get_label_map(train_labels)
    print("标签映射:", label2id)

    train_dataset = BioDataset(args.train_path, tokenizer, label2id, max_len=args.max_len)
    dev_dataset = BioDataset(args.dev_path, tokenizer, label2id, max_len=args.max_len)
    test_dataset = BioDataset(args.test_path, tokenizer, label2id, max_len=args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = CANNERModel(bert_model, num_labels=len(label2id)).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    best_f1 = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train(model, train_loader, optimizer, device)
        print(f"训练损失: {train_loss:.4f}")

        f1 = evaluate(model, dev_loader, device, id2label)
        print(f"验证集 F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_canner_model.pth")
            print("模型已保存")

    print("在测试集上评估:")
    model.load_state_dict(torch.load("best_canner_model.pth"))
    f1 = evaluate(model, test_loader, device, id2label)
    print(f"测试集 F1: {f1:.4f}")


if __name__ == "__main__":
    main()
