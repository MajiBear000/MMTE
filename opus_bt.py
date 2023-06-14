# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import csv
import os
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import shap

class RawData(object):
    def __init__(self, name):
        self.name=name
        self.data={}

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value
           
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(
        self,
        guid,
        sentence,
        index=None,
        target=None,
        label=None,
        POS=None,
        FGPOS=None,
        text_a_2=None,
        text_b_2=None,
    ):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.sentence = sentence
        self.index = index
        self.target = target
        self.label = label
        self.POS = POS
        self.FGPOS = FGPOS
        self.sen_2 = text_a_2
        self.ids_2 = text_b_2
    
def _read_mohx(data_dir, set_type):
    dataset = []
    for k in tqdm(range(10), desc='K-fold'):
        file_path = data_dir+str(k)+'.tsv'
        with open(file_path, encoding='utf8') as f:
            lines = csv.reader(f, delimiter='\t')
            next(lines)
            w_index = 0
            flag = True
            for line in lines:                
                sen_id = line[0]
                sentence = line[2]
                label = line[1]
                POS = line[3]
                FGPOS = line[4]
                ind = line[-1]

                index = int(ind)
                word = sentence.split()[index]
                guid = "%s-%s-%s" % (set_type, str(k), sen_id)

                dataset.append(
                    InputExample(guid=guid, sentence=sentence, index=index, target=word, label=label, POS=POS, FGPOS=FGPOS)
                    )
        break
        print(file_path, len(dataset))
    return dataset

def load_mohx():
    dataset_name = 'mohx'
    data_dir = 'data/MOH-X/CLS'
    dataset = RawData(dataset_name)
    train_path = os.path.join(data_dir, 'train')
    test_path = os.path.join(data_dir, 'test')
    if dataset_name == 'trofi' or 'mohx':
        dataset['test'] = _read_mohx(test_path, 'test')
        dataset['train'] = _read_mohx(train_path, 'train')
    return dataset

def tokenize_by_index(tokenizer, seq, index=None, no_flat=False):
    seq = seq.split(' ')   # seq already being splited
    tokens_ids = [[tokenizer.bos_token_id]]
    for i,ele in enumerate(seq):
        if i:    tokens_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' '+ele)))
        else:    tokens_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ele)))
    tokens_ids.append([tokenizer.eos_token_id])

    if not index==None:
        i_s = 0     #start index of target word
        for i, ele in enumerate(tokens_ids):
            i_e = i_s+len(ele)    #end index of target word
            if i == index+1:
                if not no_flat:
                    tokens_ids = sum(tokens_ids, [])  # return a flat ids list
                return tokens_ids, [i_s, i_e]
            i_s += len(ele)
    
    if not no_flat:
        tokens_ids = sum(tokens_ids, [])  # return a flat ids list
    return tokens_ids

def translate(data):
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    trans_data = RawData(data.name)

    explainer = shap.Explainer(model, tokenizer)

    cn_data_meta = []
    cn_data_liter = []
    for key in data.data.keys():
        en_data = data[key]
        for sample in tqdm(en_data):
            tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
            model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
            tokenized = tokenizer(sample.sentence,padding=True,truncation=True,max_length=512,return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**tokenized)
            cn_sam = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            shap_values = explainer(sample.sentence, fixed_context=1)
            _, idx = tokenize_by_index(tokenizer, sample.sentence, index=sample.index)
            idx = idx[0]-1
            shap_atten = shap_values.values[0]
            cor_idx = shap_atten[idx].argmax()
            cor_target = tokenizer.decode(outputs[0][cor_idx+1], skip_special_tokens=True)
            
            if sample.label == '1':
                cn_data_meta.append([sample.target, sample.index, sample.sentence, cor_idx, cor_target, cn_sam[0]])
            else:
                cn_data_liter.append([sample.target, sample.index, sample.sentence, cor_idx, cor_target, cn_sam[0]])
    trans_data['meta'] = cn_data_meta
    trans_data['liter'] = cn_data_liter
    return trans_data

def back_translate(data):
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    bt_data = RawData(data.name)

    explainer = shap.Explainer(model, tokenizer)

    en_data_meta = []
    en_data_liter = []
    for key in data.data.keys():
        cn_data = data[key]
        for sample in tqdm(cn_data):
            zh = sample[5]
            idx = sample[3]
            tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
            model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
            tokenized = tokenizer(zh,padding=True,truncation=True,max_length=512,return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**tokenized)
            en_sam = tokenizer.batch_decode(outputs)

            shap_values = explainer(zh, fixed_context=1)
            shap_atten = shap_values.values[0]
            cor_idx = shap_atten[idx].argmax()
            cor_target = tokenizer.decode(outputs[0][cor_idx+1], skip_special_tokens=True)

            if cor_target == sample[0]:
                pass_bt == '1'
            else:
                pass_bt == '0'
    
            if key == 'meta':
                en_data_meta.append([sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], cor_idx, cor_target, en_sam[0]])
            else:
                en_data_liter.append([sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], cor_idx, cor_target, en_sam[0]])
    bt_data['bt_meta'] = en_data_meta
    bt_data['bt_liter'] = en_data_liter
    return bt_data

def save_tsv(data, path, headline=None):
    print(f'{path} len: {len(data)}')
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        if headline:
            writer.writerow(headline)
        writer.writerows(data)

def save_data(data, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for key in data.data.keys():
        path = os.path.join(save_path, key+'.tsv')
        save_tsv(data[key], path, ['target','idx','en','cn_target','cn_idx','cn','bt_target','bt_idx','bt','pass_bt'])

def main():
    data = load_mohx()
    cn_data = translate(data)
    bt_data = back_translate(cn_data)
    
    save_path = 'data/MOH-X/opus'
    save_data(bt_data, save_path)    

if __name__ == '__main__':
    main()
