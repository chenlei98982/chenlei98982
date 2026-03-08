"""
需求：把项目常用的变量进行统一管理
    企业中的常规做法，形式可能不一样，可能是放在外部文件中或者shell脚本中，但是目标都是为了方便管理
主要参数：
1. 训练设备
2. 数据目录
3. 模型超参数
    3.1 输入层大小
    3.2 迭代轮数
    3.3 批次大小
    3.4 隐层大小
    3.5 学习率
    3.6 dropout
"""

import os
import torch
import json

root_module = os.path.dirname(os.path.abspath(__file__))
from transformers import BertTokenizer, BertModel


class Config(object):
    def __init__(self):
        # 如果是windows或者linux电脑（使用GPU）
        # M1芯片及其以上的电脑（使用GPU）
        self.device = 'mps'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_path = os.path.join(root_module, 'data/train.txt')
        self.train_bertpath = os.path.join(root_module, 'data/train_bert.txt')
        self.vocab_path = os.path.join(root_module, 'data/vocab.txt')
        self.embedding_dim = 300
        self.epochs = 100
        # self.batch_size = 8
        self.batch_size = 32
        self.hidden_dim = 256

        # lstm : e-3
        # bert(只微调输出层) : e-5
        # bert(全量微调)    : e-8
        self.lr = 2e-3  # crf的时候，lr可以小点，比如1e-3
        self.dropout = 0.2
        # self.model = 'BiLSTM'
        # self.model = "BiLSTM_CRF"  # 可以只用"BiLSTM"
        self.model = "BERT_BiLSTM_CRF"
        self.tag2id = json.load(open(os.path.join(root_module, 'data/tag2id.json')))

        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(root_module, 'RoBERTa_zh_L12_PyTorch'),
                                                       ignore_mismatched_sizes=True)
        self.bert_model = BertModel.from_pretrained(os.path.join(root_module, 'RoBERTa_zh_L12_PyTorch'),
                                                    ignore_mismatched_sizes=True)
        self.ber_lr = 2e-8
        self.bert_embedding_dim = 768


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
    print(torch.cuda.is_available())
    conf = Config()
    print(conf.tokenizer)
    print(conf.bert_model)
