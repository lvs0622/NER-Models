import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from FLAT_model import FLATModel
from utils.ner_utils import NERDataset
from sklearn.metrics import classification_report
import os

# 配置参数
bert_model = 'bert-base-chinese'
batch_size = 16
num_epochs = 20
max_len = 128
learning_rate = 3e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 标签
labels = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']  # 示例
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

# 加载 tokenizer 和数据集
tokenizer = BertTokenizer.from_pretrained(bert_model)
train_dataset = NERDataset('data/train.txt', tokenizer, label2id, max_len)
dev_dataset = NERDataset('data/dev.txt', tokenizer, label2id, max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

# 模型
model = FLATModel(bert_model=bert_model, num_labels=len(label2id)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 训练
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        loss, _ = model(input_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"[Epoch {epoch + 1}] Loss: {total_loss:.4f}")

    # 评估
    model.eval()
    true_labels, pred_labels = [], []
    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)

            predictions = torch.argmax(logits, dim=-1)

            for pred, gold in zip(predictions, labels):
                pred = pred.cpu().numpy()
                gold = gold.cpu().numpy()
                for p, g in zip(pred, gold):
                    if g != -100:
                        pred_labels.append(id2label[p])
                        true_labels.append(id2label[g])
    print(classification_report(true_labels, pred_labels, digits=4))
