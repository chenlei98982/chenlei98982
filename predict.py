import torch.nn as nn
import torch.optim as optim
from model.ner_lstm import *
from model.ner_lstm_crf import *
from utils.data_loader import *
from tqdm import tqdm

# 实例化模型
models = {'BiLSTM': NERLSTM,
          'BiLSTM_CRF': NERLSTM_CRF}

model = models[conf.model](conf.embedding_dim, conf.hidden_dim, conf.dropout, word2id, conf.tag2id)

if conf.model == 'BiLSTM':
    model.load_state_dict(torch.load('save_model/bilstm_best.pth'))
else:
    model.load_state_dict(torch.load('save_model/bilstm_crf_best.pth'))

id2tag = {value: key for key, value in conf.tag2id.items()}


def model2test(sample):
    x = []
    for char in sample:
        if char not in word2id:
            char = "UNK"
        x.append(word2id[char])

    x_train = torch.tensor([x])
    mask = (x_train != 0).long()
    model.eval()
    with torch.no_grad():
        if model.MODEL_NAME =="BiLSTM":
            outputs = model(x_train, mask)
            preds_ids = torch.argmax(outputs,dim=-1)[0]
            tags = [id2tag[i.item()] for i in preds_ids]
        else:
            preds_ids = model(x_train, mask)
            tags = [id2tag[i] for i in preds_ids[0]]
        chars = [i for i in sample]
        assert len(chars) == len(tags)
        result = extract_entities(chars, tags)
        return result


def extract_entities(tokens, labels):
    entities = []
    entity = []
    entity_type = None

    for token, label in zip(tokens, labels):
        if label.startswith("B-"):  # 实体的开始
            if entity:  # 如果已经有实体，先保存
                entities.append((entity_type, ''.join(entity)))
                entity = []
            entity_type = label.split('-')[1]
            entity.append(token)
        elif label.startswith("I-") and entity:  # 实体的中间或结尾
            entity.append(token)
        else:
            if entity:  # 保存上一个实体
                entities.append((entity_type, ''.join(entity)))
                entity = []
                entity_type = None

    # 如果最后一个实体没有保存，手动保存
    if entity:
        entities.append((entity_type, ''.join(entity)))

    return {entity: entity_type for entity_type, entity in entities}


if __name__ == '__main__':
    result = model2test(sample='入院后完善各项检查，给予右下肢持续皮牵引，应用健骨药物治疗，患者略发热，查血常规：白细胞数12.18*10^9/L，中性粒细胞百分比92.00%。给予应用抗生素预防感染。复查血常规：白细胞数6.55*10^9/L，中性粒细胞百分比74.70%，红细胞数2.92*10^12/L，血红蛋白94.0g/L。考虑贫血，指示加强营养。建议患者手术治疗,患者拒绝手术治疗。继续右下肢牵引，患者家属要求今日出院。')
    print(result)
