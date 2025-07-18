from sklearn.metrics import precision_recall_fscore_support

def calculate_f1(preds, labels):
    # 忽略O标签计算F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', labels=list(range(1, len(tag_alphabet))))
    return f1