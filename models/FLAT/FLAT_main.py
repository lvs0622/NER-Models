import argparse
import os
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from model import FLATModel
from utils.ner_utils import NERDataset, extract_labels_from_data
from sklearn.metrics import classification_report

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='/root/autodl-tmp/data/train.txt', type=str)
    parser.add_argument('--dev_path', default='/root/autodl-tmp/data/dev.txt', type=str)
    parser.add_argument('--test_path', default='/root/autodl-tmp/data/test.txt', type=str)
    parser.add_argument('--bert_model', default='bert-base-chinese', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--max_len', default=128, type=int)
    parser.add_argument('--lr', default=3e-5, type=float)
    return parser.parse_args()

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        loss, _ = model(input_ids, attention_mask, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, id2label):
    model.eval()
    true_labels, pred_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
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
    report = classification_report(true_labels, pred_labels, digits=4)
    return report

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义标签
    labels = extract_labels_from_data([args.train_path, args.dev_path, args.test_path])
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    train_data = NERDataset(args.train_path, tokenizer, label2id, args.max_len)
    dev_data = NERDataset(args.dev_path, tokenizer, label2id, args.max_len)
    test_data = NERDataset(args.test_path, tokenizer, label2id, args.max_len)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    model = FLATModel(bert_model=args.bert_model, num_labels=len(label2id)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("🔧 Start training...")
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizer, device)
        print(f"✅ Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}")

        print("📊 Evaluating on dev set...")
        dev_report = evaluate(model, dev_loader, device, id2label)
        print(dev_report)

    print("🧪 Evaluating on test set...")
    test_report = evaluate(model, test_loader, device, id2label)
    print(test_report)

if __name__ == '__main__':
    main()
