import pandas as pd

def func_csv2id(file_name):
    CSV_FILE_PATH = './'+file_name+'.csv'
    data = pd.read_csv(CSV_FILE_PATH,sep = ',',error_bad_lines=False,header = 0)
    data = data.values.tolist()
    return data

def preprocess(data):
    new_data = []
    print(data[0])
    for drug1,drug2,score in data:
        new_data.append([drug1[1:-1],drug2[1:-1],score])
    return new_data

def read_entity2id_file(file_path: str, drug_vocab: dict, entity_vocab: dict):
    print(f'Logging Info - Reading entity2id file: {file_path}' )
    assert len(drug_vocab) == 0 and len(entity_vocab) == 0
    entity2id = {}
    id2entity = {}
    with open(file_path, encoding='utf8') as reader:
        count=0
        for line in reader:
            if(count==0):
                count+=1
                continue
            drug, entity = line.strip().split(' ')
            drug_vocab[entity]=len(drug_vocab)
            entity_vocab[entity] = len(entity_vocab)
            #entity2id[drug] = drug_vocab[entity]
            id2entity[drug_vocab[entity]] = drug
    return id2entity

def read_pair(data,id2entity):
    DDI_pair = []
    for drug1,drug2,score in data:
        DDI_pair.append([id2entity[int(drug1)][-7:],id2entity[int(drug2)][-7:],score])
    df = pd.DataFrame(DDI_pair, columns=['drug1','drug2','score'])
    df.to_csv("./data/pdd/new_DDI_pdd_DB_score.csv", index=False, encoding='utf-8')


import os
RAW_DATA_DIR = os.getcwd()+'/raw_data'
file_path = os.path.join(RAW_DATA_DIR,'pdd','entity2id.txt')
file_name = 'data/pdd/new_DDI_pdd_score'
drug_vocab = {}
entity_vocab = {}

data = func_csv2id(file_name)
new_data = preprocess(data)
id2entity = read_entity2id_file(file_path,drug_vocab,entity_vocab)
read_pair(new_data,id2entity)

