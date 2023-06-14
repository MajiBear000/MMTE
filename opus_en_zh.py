# -*- coding: utf-8 -*-
import csv
import os
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
    
def _read_vua(data_dir, set_type):
    dataset = []
    with open(data_dir, encoding='utf8') as f:
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
            if not POS in ['NOUN', 'VERB']:
            	continue

            index = int(ind)
            word = sentence.split()[index]
            guid = "%s-%s" % (set_type, sen_id)
	
            dataset.append(
                InputExample(guid=guid, sentence=sentence, index=index, target=word, label=label, POS=POS, FGPOS=FGPOS)
                )
    print(data_dir, len(dataset))
    return dataset

def load_vua():
    dataset_name = 'vua'
    data_dir = 'data/VUA20'
    dataset = RawData(dataset_name)
    train_path = os.path.join(data_dir, 'train.tsv')
    test_path = os.path.join(data_dir, 'test.tsv')
    
    dataset['train'] = _read_vua(train_path, 'train')
    dataset['test'] = _read_vua(test_path, 'test')
    return dataset

def translate(data):
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    trans_data = RawData(data.name)
    
    sen_lib={}

    cn_data_meta = []
    cn_data_liter = []
    for key in data.data.keys():
        en_data = data[key]
        for sample in tqdm(en_data):
            if sample.guid in sen_lib.keys():
            	cn_sam = sen_lib[sample.guid]
            else:
                tokenized = tokenizer(sample.sentence,padding=True,truncation=True,max_length=512,return_tensors="pt")
                with torch.no_grad():
                    outputs = model.generate(**tokenized)
                cn_sam = tokenizer.batch_decode(outputs)
                sen_lib[sample.guid] = cn_sam
            if sample.label == '1':
                cn_data_meta.append([sample.target, sample.index, sample.sentence, cn_sam[0], sample.POS])
            else:
                cn_data_liter.append([sample.target, sample.index, sample.sentence, cn_sam[0], sample.POS])
    trans_data['meta'] = cn_data_meta
    trans_data['liter'] = cn_data_liter
    return trans_data

def back_translate(data):
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    bt_data = RawData(data.name)

    en_data_meta = []
    en_data_liter = []
    for key in data.data.keys():
        cn_data = data[key]
        for sample in tqdm(cn_data):
            tokenized = tokenizer(sample[3],padding=True,truncation=True,max_length=512,return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**tokenized)
            en_sam = tokenizer.batch_decode(outputs)
            if key == 'meta':
                en_data_meta.append([sample[0], sample[1], sample[2], sample[3], en_sam[0]])
            else:
                en_data_liter.append([sample[0], sample[1], sample[2], sample[3], en_sam[0]])
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
        path = os.path.join(save_path, key)
        save_tsv(data[key], path, ['target','index','en','cn', 'POS'])

def main():
    #data = load_mohx()
    #cn_data = translate(data)
    #bt_data = back_translate(cn_data)
    
    #save_path = 'data/MOH-X/opus'
    #save_data(bt_data, save_path)    
    
    vua = load_vua()
    zh_vua = translate(vua)
    #bt_data = back_translate(cn_data)
    
    save_path = 'data/VUA20/opus'
    save_data(zh_vua, save_path)    

if __name__ == '__main__':
    main()
