import json


class Config:
    def __init__(self):
        self.train_path = "/root/autodl-tmp/data/train.txt"
        self.dev_path = "/root/autodl-tmp/data/dev.txt"
        self.test_path = "/root/autodl-tmp/data/test.txt"
        self.label_map_path = "label_map.json"

        self.pretrained_model = "bert-base-chinese"
        self.max_len = 128
        self.batch_size = 32
        self.lr = 3e-5
        self.num_epochs = 20

def build_label_map(data_path, save_path):
    label_set = set()
    with open(data_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            splits = line.split()
            if len(splits) > 1:
                label_set.add(splits[-1])

    label_list = sorted(label_set)
    label2id = {label: i for i, label in enumerate(label_list)}
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    print(f"Label map saved to {save_path} with {len(label2id)} labels.")

build_label_map("/root/autodl-tmp/data/train.txt", "label_map.json")
