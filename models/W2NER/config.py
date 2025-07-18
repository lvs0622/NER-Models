import json
import os


class Config:
    def __init__(self, args):
        if not hasattr(args, 'config'):
            raise ValueError("args must have 'config' attribute specifying the config file path")

        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found at: {args.config}")

        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 数据文件路径
        self.train_file = config.get("train_file", "/root/autodl-tmp/data/train.json")
        self.dev_file = config.get("dev_file", "/root/autodl-tmp/data/dev.json")
        self.test_file = config.get("test_file", "/root/autodl-tmp/data/test.json")

        # 配置字段设定（有默认值的写法）
        self.dataset = config.get("dataset", "default_dataset")
        self.save_path = config.get("save_path", "model.pt")
        self.predict_path = config.get("predict_path", "output.json")

        self.dist_emb_size = config.get("dist_emb_size", 50)
        self.type_emb_size = config.get("type_emb_size", 50)
        self.lstm_hid_size = config.get("lstm_hid_size", 200)
        self.conv_hid_size = config.get("conv_hid_size", 100)
        self.bert_hid_size = config.get("bert_hid_size", 768)
        self.biaffine_size = config.get("biaffine_size", 128)
        self.ffnn_hid_size = config.get("ffnn_hid_size", 150)

        self.dilation = config.get("dilation", [1, 2, 3])

        self.emb_dropout = config.get("emb_dropout", 0.1)
        self.conv_dropout = config.get("conv_dropout", 0.1)
        self.out_dropout = config.get("out_dropout", 0.1)

        self.epochs = config.get("epochs", 20)
        self.batch_size = config.get("batch_size", 32)

        self.learning_rate = config.get("learning_rate", 3e-5)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.clip_grad_norm = config.get("clip_grad_norm", 1.0)
        self.bert_name = config.get("bert_name", "bert-base-uncased")
        self.bert_learning_rate = config.get("bert_learning_rate", 2e-5)
        self.warm_factor = config.get("warm_factor", 0.1)

        self.use_bert_last_4_layers = config.get("use_bert_last_4_layers", 1)
        self.seed = config.get("seed", 42)

        # 用命令行参数覆盖配置（如果有）
        # 只在 args 中字段值不为 None 且不为 False 的情况下，才覆盖 config 中已有的配置
        for k, v in vars(args).items():
            if v is not None and v != "" and k != "config":
                setattr(self, k, v)

    def __repr__(self):
        return "\n".join(f"{k}: {v}" for k, v in sorted(self.__dict__.items()))
