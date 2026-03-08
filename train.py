import torch.optim as optim
from model.ner_lstm import *
from model.ner_lstm_crf import *
from utils.data_loader import *
from tqdm import tqdm
from model.ner_bert_lstm_crf import BERT_NERLSTM_CRF
# classification_report可以导出字典格式，修改参数：output_dict=True，可以将字典在保存为csv格式输出
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from config import *

conf = Config()


def model2train():
    # 获取数据集
    train_dataloader, dev_dataloader = get_data()
    """"
        # self.model = 'BiLSTM'
        self.model = "BiLSTM_CRF"  
    """
    model_dict = {
        'BiLSTM': NERLSTM,
        'BiLSTM_CRF': NERLSTM_CRF,
        'BERT_BiLSTM_CRF':BERT_NERLSTM_CRF
    }
    embedding_dim=conf.embedding_dim
    if 'BERT_BiLSTM_CRF' == conf.model:
        embedding_dim=conf.bert_embedding_dim

    model = model_dict[conf.model](
        embedding_dim
        , conf.hidden_dim
        , conf.dropout
        , word2id
        , conf.tag2id
    )
    model.to(conf.device)
    criterion = nn.CrossEntropyLoss()
    optimer = optim.AdamW(model.parameters(), lr=conf.lr)

    f1_score = -1000
    if 'BiLSTM' == model.MODEL_NAME:
        for epoch in range(conf.epochs):
            model.train()
            loss_total,total=0.0,0
            for i,(input_ids_padded, labels_padded, attention_mask)  in enumerate(tqdm(train_dataloader)):
                input_ids_padded=input_ids_padded.to(conf.device)
                labels_padded=labels_padded.to(conf.device)
                attention_mask=attention_mask.to(conf.device)
                y_pred= model(input_ids_padded,attention_mask)
                y_pred=y_pred.view(-1, len(conf.tag2id))
                y_true=labels_padded.view(-1)

                loss=criterion(y_pred,y_true)
                optimer.zero_grad()
                loss.backward()
                optimer.step()
                loss_total+=loss.item()
                total+=1
            precision, recall, f1, report = model2dev(dev_dataloader, model)
            if f1 > f1_score:
                # f1_score = f1
                torch.save(model.state_dict(), 'save_model/bilstm_best.pth')
                print(report)

    elif 'BiLSTM_CRF' == model.MODEL_NAME:
        for epoch in range(conf.epochs):
            model.train()
            for i,(input_ids_padded, labels_padded, attention_mask)  in enumerate(tqdm(train_dataloader)):
                input_ids_padded = input_ids_padded.to(conf.device)
                labels_padded = labels_padded.to(conf.device)
                attention_mask = attention_mask.to(conf.device)
                attention_mask=attention_mask.to(bool)
                loss = model.log_likelihood(input_ids_padded,labels_padded,attention_mask).mean()
                optimer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),10)
                optimer.step()
            precision, recall, f1, report = model2dev(dev_dataloader, model)
            if f1 > f1_score:
                # f1_score = f1
                torch.save(model.state_dict(), 'save_model/bilstm_crf_best.pth')
                print(report)
    elif 'BERT_BiLSTM_CRF' == model.MODEL_NAME:
        for param in model.word_embeds.parameters():
            param.requires_grad = False
        for layer in model.word_embeds.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        optimer = optim.AdamW(
            [
                {'params': [p for n, p in model.word_embeds.named_parameters() if 'layer.10' in n or 'layer.11' in n],
                 'lr': 2e-8},

                # {'params': model.word_embeds.parameters(), 'lr': 2e-8},  # BERT学习率小
                {'params': model.lstm.parameters(), 'lr': 5e-4},  # BiLSTM中等
                {'params': model.hidden2tag.parameters(), 'lr': 5e-4},  # 全连接层中等
                {'params': model.crf.parameters(), 'lr': 5e-4}  # CRF学习率大一点
            ],  # <--- 注意这里的逗号和闭合括号
            weight_decay=0.01  # AdamW的其他参数写在这里
        )
        for epoch in range(conf.epochs):
            model.train()
            for i, (input_ids_padded, labels_padded, attention_mask) in enumerate(tqdm(train_dataloader)):
                input_ids_padded = input_ids_padded.to(conf.device)
                labels_padded = labels_padded.to(conf.device)
                attention_mask = attention_mask.to(conf.device)
                attention_mask = attention_mask.to(bool)
                loss = model.log_likelihood(input_ids_padded, labels_padded, attention_mask).mean()
                optimer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimer.step()
            precision, recall, f1, report = model2dev(dev_dataloader, model)
            if f1 > f1_score:
                # f1_score = f1
                torch.save(model.state_dict(), 'save_model/bert_bilstm_crf_best.pth')
                print(report)
    else:
        print(f"模型:{conf.model}不存在! ")
        return

def model2dev(dev_iter, model):
    preds, golds = [], []
    model.eval()
    for index, (inputs, labels, mask) in enumerate(tqdm(dev_iter, desc="测试集验证")):
        val_x = inputs.to(conf.device)
        mask = mask.to(conf.device)
        val_y = labels.to(conf.device)
        predict = []
        if model.MODEL_NAME == "BiLSTM":
            pred = model(val_x, mask)
            predict = torch.argmax(pred, dim=-1).tolist()
        elif model.MODEL_NAME == "BiLSTM_CRF":
            mask = mask.to(torch.bool)
            predict = model(val_x, mask)
        else:
            mask = mask.to(torch.bool)
            predict = model(val_x, mask)
        # 统计非0的，也就是真实标签的长度
        # 4. TODO 计算评估指标 ： 获取真实长度预测值和真实值
        leng = []
        # TODO val_y因为我们进行padding的时候补的是0，非实体O也是0，在这里我们无法判断它是补充的还是本身就是非实体
        for i in mask.cpu():
            tmp = []
            for j in i:
                if j.item() > 0:
                    tmp.append(j.item())
            leng.append(len(tmp))
        # 提取真实长度的预测标签
        print(f'predict---->{predict}')
        for index, i in enumerate(predict):
            preds.extend(i[:leng[index]])

        # 提取真实长度的真实标签
        for index, i in enumerate(val_y.cpu().tolist()):
            golds.extend(i[:leng[index]])
    """
    TODO average的不同策略
    ``'binary'``:
        仅报告由 ``pos_label`` 指定的类别的结果。
        这仅适用于目标（``y_{true,pred}``）是二分类的情况。
    ``'micro'``:
        通过计算总真正例、假反例和假正例来全局计算指标。
    ``'macro'``:
        计算每个标签的指标，并求其未加权的平均值。
        这不会考虑标签不平衡。
    ``'weighted'``:
        计算每个标签的指标，并找到其按支持度（每个标签的真实实例数）加权的平均值。
        这会调整 'macro' 以考虑标签不平衡；可能导致 F-score 不在精确度和召回率之间。
    ``'samples'``:
        计算每个实例的指标，并求其平均值（仅适用于多标签分类，且此处与 :func:`accuracy_score` 不同的情况）。
    """
    precision = precision_score(golds, preds, average='macro')
    recall = recall_score(golds, preds, average='macro')
    f1 = f1_score(golds, preds, average='macro')
    report = classification_report(golds, preds)

    return precision, recall, f1, report
if __name__ == '__main__':
    model2train()