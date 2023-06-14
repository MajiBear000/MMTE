# -*- coding: utf-8 -*-
import csv
import os
import torch
import sys
import uuid
import requests
import hashlib
import time
from imp import reload

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

reload(sys)

YOUDAO_URL = 'https://openapi.youdao.com/api'
APP_KEY = '0878e59daea994d0'
APP_SECRET = 'hdfC8DHqMYUiLFnX45HWjUdCvk8JDEEJ'

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
        print(file_path, len(dataset))
    return dataset

def load_mohx():
    dataset_name = 'mohx'
    data_dir = 'data/MOH-X/CLS'
    dataset = RawData(dataset_name)
    train_path = os.path.join(data_dir, 'train')
    test_path = os.path.join(data_dir, 'test')
    if dataset_name == 'trofi' or 'mohx':
        dataset['train'] = _read_mohx(train_path, 'train')
        #dataset['test'] = _read_mohx(test_path, 'test')
    return dataset

def encrypt(signStr):
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(signStr.encode('utf-8'))
    return hash_algorithm.hexdigest()

def truncate(q):
    if q is None:
        return None
    size = len(q)
    return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]

def connect(sentence):
    q = sentence
    data = {}
    data['from'] = 'en'
    data['to'] = 'zh-CHS'
    data['signType'] = 'v3'
    curtime = str(int(time.time()))
    data['curtime'] = curtime
    salt = str(uuid.uuid1())
    signStr = APP_KEY + truncate(q) + salt + curtime + APP_SECRET
    sign = encrypt(signStr)
    data['appKey'] = APP_KEY
    data['q'] = q
    data['salt'] = salt
    data['sign'] = sign
    data['vocabId'] = '196F1D5844F5479DB21F07BD2042E9C6'

    output = requests.get(YOUDAO_URL, params=data).json()
    print(output['translation'][0])
    return output['translation']

def translate(data):
    trans_data = RawData(data.name)
    
    for key in data.data.keys():
        cn_data_meta = []
        cn_data_liter = []
        en_data = data[key]
        for sample in tqdm(en_data):
            time.sleep(1)
            cn_sam = connect(sample.sentence)
            if sample.label == '1':
                cn_data_meta.append([sample.target, sample.index, sample.sentence, cn_sam[0]])
            else:
                cn_data_liter.append([sample.target, sample.index, sample.sentence, cn_sam[0]])
        trans_data[key+'_meta'] = cn_data_meta
        trans_data[key+'_liter'] = cn_data_liter
    return trans_data

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
        save_tsv(data[key], path, ['target','index','en','cn'])

def main():
    data = load_mohx()
    cn_data = translate(data)
    
    save_path = 'data/MOH-X/youdao'
    save_data(cn_data, save_path)    

if __name__ == '__main__':
    main()
