import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup, AdamW
from sklearn.metrics import precision_recall_fscore_support
from bertcrf_model import BERTCRF
from utils import NERDataset, collate_fn, EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数
EPOCHS = 20
LR = 3e-5
BATCH_SIZE = 32
MODEL_NAME = 'bert-base-chinese'
PATIENCE = 3

# 数据准备
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
train_dataset = NERDataset("/root/autodl-tmp/data/train.txt", tokenizer)
dev_dataset = NERDataset("/root/autodl-tmp/data/dev.txt", tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

model = BERTCRF(MODEL_NAME, num_labels=train_dataset.num_labels).to(device)
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
early_stopper = EarlyStopping(patience=PATIENCE, verbose=True)

def evaluate():
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dev_loader:
            input_ids, attn_mask, labels = [b.to(device) for b in batch]
            preds = model(input_ids, attn_mask)
            for p, l, m in zip(preds, labels.cpu().numpy(), attn_mask.cpu().numpy()):
                true_len = sum(m)
                all_preds.extend(p[:true_len])
                all_labels.extend(l[:true_len])
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro', zero_division=0)
    return p, r, f1

# 训练
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, attn_mask, labels = [b.to(device) for b in batch]
        loss = model(input_ids, attn_mask, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    p, r, f1 = evaluate()
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | P: {p:.4f} R: {r:.4f} F1: {f1:.4f}")
    early_stopper(f1, model)
    if early_stopper.early_stop:
        print("Early stopping.")
        break
