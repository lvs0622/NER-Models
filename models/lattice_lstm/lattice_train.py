import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.ner_utils import load_ner_data, build_char_vocab, build_word_vocab, build_label_vocab, preprocess_data, batch_generator, evaluate, print_classification_report
from lattice_model import LatticeLSTM  # 你的模型代码文件名
from sklearn.metrics import f1_score
import time
from tqdm import tqdm

class Config:
    epochs = 20
    lr = 3e-5
    batch_size = 32
    char_emb_dim = 100
    word_emb_dim = 100
    word_drop = 0.3
    hidden_dim = 9  # 你模型的hidden_dim保持9
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# ----------- 数据加载 -----------
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
                print(f"❌ 格式错误的行: {line}")  # 输出错误信息
                continue
            f.write(line + '\n')


# ======= 清洗数据（加在数据加载前）=======
clean_ner_file('/root/autodl-tmp/data/train.txt')
clean_ner_file('/root/autodl-tmp/data/dev.txt')


train_sents, train_labels = load_ner_data('/root/autodl-tmp/data/train.txt')
dev_sents, dev_labels = load_ner_data('/root/autodl-tmp/data/dev.txt')

from collections import Counter

all_labels = [label for seq in train_labels for label in seq]
print("标签分布：", Counter(all_labels))


char2id, _ = build_char_vocab(train_sents)
word2id, _ = build_word_vocab(train_sents)
label2id, id2label = build_label_vocab(train_labels)

train_data, pad_id = preprocess_data(train_sents, train_labels, char2id, label2id, word2id)
dev_data, _ = preprocess_data(dev_sents, dev_labels, char2id, label2id, word2id)

num_labels = len(label2id)

# ----------- 模型初始化 -----------
model = LatticeLSTM(
    char_vocab_size=len(char2id),
    char_emb_dim=config.char_emb_dim,
    hidden_dim=config.hidden_dim,
    word_drop=config.word_drop,
    word_alphabet_size=len(word2id),
    word_emb_dim=config.word_emb_dim,
    num_labels=num_labels,
    pretrain_word_emb=None,
    fix_word_emb=False,
    gpu=torch.cuda.is_available()
)

classifier = nn.Linear(config.hidden_dim, num_labels)

model.to(config.device)
classifier.to(config.device)

optimizer = AdamW(list(model.parameters()) + list(classifier.parameters()), lr=config.lr)
num_training_steps = (len(train_data) // config.batch_size + 1) * config.epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
criterion = nn.CrossEntropyLoss()

# ----------- 训练主循环 -----------
best_f1 = 0.0
patience = 3
patience_counter = 0

for epoch in range(config.epochs):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    print(f"\n🟢 Epoch {epoch + 1}/{config.epochs}")
    batch_iter = tqdm(batch_generator(train_data, config.batch_size, pad_id), desc="Training", total=(len(train_data) // config.batch_size + 1))

    for batch_idx, (char_ids, labels, skip_inputs) in enumerate(batch_iter):
        char_ids = char_ids.to(config.device)
        labels = labels.to(config.device)
        batch_size = char_ids.size(0)

        optimizer.zero_grad()
        hidden_outs = []

        for i in range(batch_size):
            seq_len = (char_ids[i] != 0).sum().item()
            if seq_len == 0:
                continue
            char_input = char_ids[i:i + 1, :seq_len]
            skip_input = [skip_inputs[i], True]
            hx = torch.zeros(1, config.hidden_dim).to(config.device)
            cx = torch.zeros(1, config.hidden_dim).to(config.device)

            output_hidden = model(char_input, skip_input, (hx, cx))  # (1, seq_len, hidden_dim)
            hidden_outs.append(output_hidden.squeeze(0))  # (seq_len, hidden_dim)

        if not hidden_outs:
            continue

        max_len = max(h.size(0) for h in hidden_outs)
        padded = []
        for h in hidden_outs:
            pad_len = max_len - h.size(0)
            padded.append(torch.cat([h, torch.zeros(pad_len, 1, config.hidden_dim).to(config.device)], dim=0))
        hidden_out_batch = torch.stack(padded)  # (batch_size, max_len, hidden_dim)

        logits = classifier(hidden_out_batch)  # (batch_size, max_len, num_labels)
        logits_flat = logits.view(-1, num_labels)
        labels_flat = labels[:, :max_len].contiguous().view(-1)
        mask = (labels_flat != pad_id)

        loss = criterion(logits_flat[mask], labels_flat[mask])
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # 更新进度条上的信息
        batch_iter.set_postfix(loss=loss.item())

    avg_loss = total_loss / (len(train_data) // config.batch_size + 1)
    elapsed = time.time() - start_time
    print(f"🟡 Epoch {epoch + 1} completed in {elapsed:.2f}s - Avg Loss: {avg_loss:.4f}")

    # ----------- 验证 ----------- #
    model.eval()
    all_preds, all_labels, _ = evaluate(model, dev_data, classifier, config, criterion, label2id)

    f1 = f1_score(all_labels, all_preds, average='micro')
    print(f"🔵 Epoch {epoch + 1} Dev F1: {f1:.4f}")
    print_classification_report(all_labels, all_preds, label2id)

    if f1 > best_f1:
        best_f1 = f1
        patience_counter = 0
        torch.save({'model': model.state_dict(), 'classifier': classifier.state_dict()}, 'best_model.pth')
        print("✅ Best model saved.")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⛔ Early stopping triggered.")
            break