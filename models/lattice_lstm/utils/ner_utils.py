import os
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

######################################
#           数据加载函数             #
######################################
def load_ner_data(file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """从文件中加载NER数据"""
    sents = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        current_sent = []
        current_labels = []
        for line in f:
            line = line.strip()
            if not line:
                if current_sent:
                    sents.append(current_sent)
                    labels.append(current_labels)
                    current_sent = []
                    current_labels = []
                continue
            parts = line.split()
            if len(parts) != 2:
                print(f"格式错误的行: {line}")
                continue
            word, label = parts
            current_sent.append(word)
            current_labels.append(label)
        if current_sent:
            sents.append(current_sent)
            labels.append(current_labels)
    return sents, labels


######################################
#           词典构建函数             #
######################################
def build_char_vocab(sents: List[List[str]]) -> Tuple[dict, int]:
    char2id = {"<PAD>": 0, "<UNK>": 1}
    for sent in sents:
        for char in sent:
            if char not in char2id:
                char2id[char] = len(char2id)
    return char2id, len(char2id)

def build_word_vocab(sents: List[List[str]], max_len: int = 4) -> Tuple[dict, int]:
    word2id = {"<PAD>": 0, "<UNK>": 1}
    for sent in sents:
        for i in range(len(sent)):
            for j in range(i + 1, min(i + max_len + 1, len(sent) + 1)):
                word = "".join(sent[i:j])
                if word not in word2id:
                    word2id[word] = len(word2id)
    return word2id, len(word2id)

def build_label_vocab(labels: List[List[str]]) -> Tuple[dict, dict]:
    unique_labels = sorted(set(label for seq in labels for label in seq))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


######################################
#            输入处理逻辑            #
######################################
def generate_skip_input(sent: List[str], word2id: dict, max_len: int = 4):
    seq_len = len(sent)
    skip_input = [[] for _ in range(seq_len)]
    for i in range(seq_len):
        word_ids = []
        lengths = []
        for j in range(i + 1, min(i + max_len + 1, seq_len + 1)):
            word = ''.join(sent[i:j])
            if word in word2id:
                word_ids.append(word2id[word])
                lengths.append(j - i)
        if word_ids:
            skip_input[i] = [word_ids, lengths]
    return skip_input

def preprocess_data(sents, labels, char2id, label2id, word2id, max_word_len=4):
    pad_id = len(label2id)
    data = []
    for sent, label_seq in zip(sents, labels):
        char_ids = [char2id.get(char, char2id['<UNK>']) for char in sent]
        label_ids = [label2id.get(label, pad_id) for label in label_seq]
        skip_input = generate_skip_input(sent, word2id, max_word_len)
        data.append((char_ids, label_ids, skip_input))
    return data, pad_id


######################################
#            批次生成器              #
######################################
def batch_generator(data, batch_size: int, pad_id=0):
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        char_ids_list, label_ids_list, skip_inputs_list = zip(*batch)
        max_len = max(len(seq) for seq in char_ids_list)

        char_ids_padded = []
        label_ids_padded = []
        for chars, labels in zip(char_ids_list, label_ids_list):
            pad_len = max_len - len(chars)
            char_ids_padded.append(chars + [0] * pad_len)
            label_ids_padded.append(labels + [pad_id] * pad_len)

        char_tensor = torch.LongTensor(char_ids_padded)
        label_tensor = torch.LongTensor(label_ids_padded)

        yield char_tensor, label_tensor, skip_inputs_list


######################################
#        通用评估函数（可复用）       #
######################################
def flatten_lists(true_labels, pred_labels):
    y_true = [label for seq in true_labels for label in seq]
    y_pred = [label for seq in pred_labels for label in seq]
    return y_true, y_pred

def evaluate(model, data, classifier, config, criterion, label2id):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    pad_id = len(label2id)

    with torch.no_grad():
        for char_ids, labels, skip_inputs in batch_generator(data, config.batch_size, pad_id):
            char_ids = char_ids.to(config.device)
            labels = labels.to(config.device)
            batch_size = char_ids.size(0)
            hidden_outs = []

            for i in range(batch_size):
                seq_len = (char_ids[i] != 0).sum().item()
                if seq_len == 0:
                    continue
                char_input = char_ids[i:i+1, :seq_len]
                skip_input = [skip_inputs[i], True]
                hx = torch.zeros(1, config.hidden_dim).to(config.device)
                cx = torch.zeros(1, config.hidden_dim).to(config.device)
                output = model(char_input, skip_input, (hx, cx))
                hidden_outs.append(output.squeeze(0))

            if hidden_outs:
                max_len = max(h.size(0) for h in hidden_outs)
                padded = []
                for h in hidden_outs:
                    pad_len = max_len - h.size(0)
                    zero_padding = torch.zeros(pad_len, h.size(1), h.size(2)).to(config.device)  # 保持维度一致
                    padded.append(torch.cat([h, zero_padding], dim=0))

                hidden_out_batch = torch.stack(padded)
                logits = classifier(hidden_out_batch)
                logits_flat = logits.view(-1, logits.shape[-1])
                labels_flat = labels.view(-1)
                mask = (labels_flat != pad_id)

                loss = criterion(logits_flat[mask], labels_flat[mask])
                total_loss += loss.item()

                preds = torch.argmax(logits_flat, dim=-1).cpu().numpy()
                all_preds.extend(preds[mask.cpu().numpy()])
                all_labels.extend(labels_flat[mask].cpu().numpy())

    return all_preds, all_labels, total_loss / len(data)


######################################
#       打印分类报告（选配）          #
######################################
def print_classification_report(y_true, y_pred, label2id):
    labels = list(label2id.values())
    target_names = list(label2id.keys())
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0
    )
    print(report)


def clean_ner_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(file_path, 'w', encoding='utf-8') as f:
        for line in lines:
            line = line.strip()
            if not line:
                f.write('\n')
                continue
            parts = line.split()
            if len(parts) != 2:
                print(f"❌ 格式错误的行: {line}")  # 打印错误行，方便定位
                continue
            f.write(line + '\n')
