# utils/build_lexicon_from_bio.py

from collections import Counter

def extract_lexicon_from_bio(file_list, output_path, min_freq=1):
    entity_counter = Counter()

    for file in file_list:
        with open(file, 'r', encoding='utf-8') as f:
            words = []
            labels = []

            for line in f:
                line = line.strip()
                if not line:
                    # process a sentence
                    i = 0
                    while i < len(labels):
                        if labels[i].startswith('B-'):
                            entity_type = labels[i][2:]
                            j = i + 1
                            while j < len(labels) and labels[j] == f'I-{entity_type}':
                                j += 1
                            entity = ''.join(words[i:j])
                            entity_counter[entity] += 1
                            i = j
                        else:
                            i += 1
                    words = []
                    labels = []
                else:
                    try:
                        word, label = line.split()
                        words.append(word)
                        labels.append(label)
                    except:
                        continue

    # 写入词典文件
    with open(output_path, 'w', encoding='utf-8') as out:
        for entity, count in entity_counter.items():
            if count >= min_freq and len(entity) > 1:
                out.write(entity + '\n')

    print(f"✅ 构建完成，共写入 {len(entity_counter)} 个实体词到 {output_path}")


if __name__ == '__main__':
    extract_lexicon_from_bio(
        file_list=[
            '/root/autodl-tmp/data/train.txt',
            '/root/autodl-tmp/data/dev.txt',
            '/root/autodl-tmp/data/test.txt',
        ],
        output_path='/root/autodl-tmp/data/lexicon.txt',
        min_freq=1  # 可调成 2 或 3，过滤低频实体
    )
