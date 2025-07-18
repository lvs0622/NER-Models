import json

def bio_to_span(input_file, output_file):
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sentence = []
    labels = []

    for line in lines:
        line = line.strip()
        if not line:
            if sentence:
                entities = []
                i = 0
                while i < len(labels):
                    if labels[i].startswith("B-"):
                        start = i
                        ent_type = labels[i][2:]
                        i += 1
                        while i < len(labels) and labels[i] == f"I-{ent_type}":
                            i += 1
                        end = i - 1
                        entities.append({"index": [start, end], "type": ent_type})
                    else:
                        i += 1
                data.append({"sentence": sentence, "ner": entities})
                sentence = []
                labels = []
        else:
            if ' ' in line:
                word, label = line.split()
            else:
                continue  # 忽略非法行
            sentence.append(word)
            labels.append(label)

    # 处理最后一个句子
    if sentence:
        entities = []
        i = 0
        while i < len(labels):
            if labels[i].startswith("B-"):
                start = i
                ent_type = labels[i][2:]
                i += 1
                while i < len(labels) and labels[i] == f"I-{ent_type}":
                    i += 1
                end = i - 1
                entities.append({"index": [start, end], "type": ent_type})
            else:
                i += 1
        data.append({"sentence": sentence, "ner": entities})

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"转换完成，已保存为 {output_file}")


input_path = "train.txt"  # 替换为你的BIO输入文件路径
output_path = "train.json"  # 你希望保存的输出路径
bio_to_span(input_path, output_path)
