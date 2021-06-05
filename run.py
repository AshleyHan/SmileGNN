# -*- coding: utf-8 -*-
import sys
import random
import os
import numpy as np
import re
from collections import defaultdict


sys.path.append('/home/.../SmileGNN/')  # add the env path
from sklearn.model_selection import train_test_split, StratifiedKFold
from SmileGNN.main import train
from SmileGNN.layers.feature import *

from SmileGNN.config import DRUG_EXAMPLE, RESULT_LOG, PROCESSED_DATA_DIR, LOG_DIR, MODEL_SAVED_DIR, ENTITY2ID_FILE, \
    KG_FILE, \
    EXAMPLE_FILE, DRUG_VOCAB_TEMPLATE, ENTITY_VOCAB_TEMPLATE, \
    RELATION_VOCAB_TEMPLATE, SEPARATOR, THRESHOLD, TRAIN_DATA_TEMPLATE, DEV_DATA_TEMPLATE, DRUG_FEATURE_TEMPLATE, \
    DRUG_SIM_TEMPLATE, \
    TEST_DATA_TEMPLATE, ADJ_ENTITY_TEMPLATE, ADJ_RELATION_TEMPLATE, ModelConfig, NEIGHBOR_SIZE
from SmileGNN.utils import pickle_dump, format_filename, write_log, pickle_load


def read_entity2id_file(file_path: str, drug_vocab: dict, entity_vocab: dict,dataset:str):
    # read_entity2id_file
    # 输入：entity2id 文件路径；空字典；空字典
    # 输出：字典entity2id：知识图谱路径-新编号
    print(f'Logging Info - Reading entity2id file: {file_path}')
    assert len(drug_vocab) == 0 and len(entity_vocab) == 0
    entity2id = {}
    with open(file_path, encoding='utf8') as reader:
        count = 0
        for line in reader:
            if (count == 0):
                count += 1
                continue
            if dataset == 'kegg':
                drug, entity = line.strip().split('\t')
            elif dataset == 'pdd':
                drug, entity = line.strip().split(' ')
            drug_vocab[entity] = len(drug_vocab)
            entity_vocab[entity] = len(entity_vocab)
            entity2id[drug] = drug_vocab[entity]
    return entity2id


def read_example_file(file_path: str, separator: str, drug_vocab: dict):
    #read approved_example
    #输入：approved_example.txt;分隔符；entity2id字典
    #输出：np array 药物相互作用矩阵 [drug1_id, drug2_id, 1/0]
    print(f'Logging Info - Reading example file: {file_path}')
    assert len(drug_vocab) > 0
    examples = []
    with open(file_path, encoding='utf8') as reader:
        for idx, line in enumerate(reader):
            d1, d2, flag = line.strip().split(separator)[:3]
            if d1 not in drug_vocab or d2 not in drug_vocab:
                continue
            if d1 in drug_vocab and d2 in drug_vocab:
                examples.append([drug_vocab[d1], drug_vocab[d2], int(flag)])

    examples_matrix = np.array(examples)
    print(f'size of example: {examples_matrix.shape}')
    X = examples_matrix[:, :2]
    y = examples_matrix[:, 2:3]
    train_data_X, valid_data_X, train_y, val_y = train_test_split(X, y, test_size=0.2, stratify=y)
    train_data = np.c_[train_data_X, train_y]
    valid_data_X, test_data_X, val_y, test_y = train_test_split(valid_data_X, val_y, test_size=0.5)
    valid_data = np.c_[valid_data_X, val_y]
    test_data = np.c_[test_data_X, test_y]
    return examples_matrix


def read_kg(file_path: str, entity_vocab: dict, relation_vocab: dict, neighbor_sample_size: int):
    #read train2id(kg)
    #输入：train2id.txt；entity2id字典；relation2id字典；neighbor sample size(5)
    #输出：adj_entity,每个节点的五个邻居节点id；adj_relation，每个节点与五个邻居节点关联类型
    print(f'Logging Info - Reading kg file: {file_path}')

    kg = defaultdict(list)
    with open(file_path, encoding='utf8') as reader:
        count = 0
        for line in reader:
            if count == 0:
                count += 1
                continue
            head, tail, relation = line.strip().split(' ')

            if head not in entity_vocab:
                entity_vocab[head] = len(entity_vocab)
            if tail not in entity_vocab:
                entity_vocab[tail] = len(entity_vocab)
            if relation not in relation_vocab:
                relation_vocab[relation] = len(relation_vocab)

            # undirected graph
            kg[entity_vocab[head]].append((entity_vocab[tail], relation_vocab[relation]))
            kg[entity_vocab[tail]].append((entity_vocab[head], relation_vocab[relation]))
    print(f'Logging Info - num of entities: {len(entity_vocab)}, '
          f'num of relations: {len(relation_vocab)}')

    print('Logging Info - Constructing adjacency matrix...')
    n_entity = len(entity_vocab)
    adj_entity = np.zeros(shape=(n_entity, neighbor_sample_size), dtype=np.int64)
    adj_relation = np.zeros(shape=(n_entity, neighbor_sample_size), dtype=np.int64)

    #如果邻居节点>5,随机选取5个
    for entity_id in range(n_entity):
        all_neighbors = kg[entity_id]
        n_neighbor = len(all_neighbors)
        if n_neighbor == 0:
            print(entity_id)
            continue
        sample_indices = np.random.choice(
            n_neighbor,
            neighbor_sample_size,
            replace=False if n_neighbor >= neighbor_sample_size else True
        )

        adj_entity[entity_id] = np.array([all_neighbors[i][0] for i in sample_indices])
        adj_relation[entity_id] = np.array([all_neighbors[i][1] for i in sample_indices])

    return adj_entity, adj_relation


import pandas as pd


def read_feature(file_path: str, entity2id: dict, drug_vocab: dict,dataset:str):
    #读取药物结构特征向量
    #输入：药物结构特征.csv；entity2id字典；drug_vocab字典
    #输出：np array 药物结构特征向量
    drug_smiles = np.zeros(shape=(len(drug_vocab), 64), dtype=float)
    data = pd.read_csv(file_path, sep=',', error_bad_lines=False)
    data = data.values.tolist()
    i = 0
    j = 0
    for drug in data:
        drug_name = ''
        if dataset == 'kegg':
            id = drug[-1]       #kegg -1 pdd -2
            drug_name = '<http://bio2rdf.org/kegg:' + id + '>'
        elif dataset == 'pdd':
            id = drug[-2]
            drug_name = 'http://bio2rdf.org/drugbank:' + id
        if drug_name not in entity2id:
            j += 1
        else:
            i += 1
            drug_feature = drug[0:64]
            drug_smiles[entity2id[drug_name]] = np.array(drug_feature)
    print(i, j)
    return drug_smiles


def read_sim(file_path: str, entity2id: dict, drug_vocab: dict):
    # 从文件中读取到dic中
    # 将drug名称转换为id编号，新dic
    # 生成np.matrix，返回
    # 后续处理在model中完成
    drug_sim = np.zeros(shape=(len(entity2id), len(entity2id)), dtype='float32')  # float64占用内存太大，改为32
    data = pd.read_csv(file_path, header=0, index_col=0)
    data = data.to_dict()
    i = 0
    for drug1 in data:
        drug_name1 = '<http://bio2rdf.org/kegg:' + drug1 + '>'
        if drug_name1 in entity2id:
            for drug2 in data[drug1]:
                drug_name2 = '<http://bio2rdf.org/kegg:' + drug2 + '>'
                if drug_name2 in entity2id:
                    i += 1
                    drug_sim[entity2id[drug_name1]][entity2id[drug_name2]] = data[drug1][drug2]
    print(i, len(drug_vocab) * len(drug_vocab))
    return drug_sim


def process_data(dataset: str, neighbor_sample_size: int, K: int):
    drug_vocab = {}
    entity_vocab = {}
    relation_vocab = {}

    entity2id = read_entity2id_file(ENTITY2ID_FILE[dataset], drug_vocab, entity_vocab,dataset)

    pickle_dump(format_filename(PROCESSED_DATA_DIR, DRUG_VOCAB_TEMPLATE, dataset=dataset), drug_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, ENTITY_VOCAB_TEMPLATE, dataset=dataset), entity_vocab)

    examples_file = format_filename(PROCESSED_DATA_DIR, DRUG_EXAMPLE, dataset=dataset)
    examples = read_example_file(EXAMPLE_FILE[dataset], SEPARATOR[dataset], drug_vocab)
    np.save(examples_file, examples)

    drug_feature = read_feature(os.path.join(os.getcwd() + '/raw_data' + '/pca_smiles_kegg_64.csv'), entity2id, drug_vocab,dataset)
    drug_feature_file = format_filename(PROCESSED_DATA_DIR, DRUG_FEATURE_TEMPLATE, dataset=dataset)
    np.save(drug_feature_file, drug_feature, allow_pickle=True)
    print('Logging Info - Saved:', drug_feature_file)

    '''
    #注释部分：第一次运行，还没有np save过，需要跑一遍；后续就不用了
    adj_entity_file = format_filename(PROCESSED_DATA_DIR, ADJ_ENTITY_TEMPLATE, dataset=dataset)
    adj_relation_file = format_filename(PROCESSED_DATA_DIR, ADJ_RELATION_TEMPLATE, dataset=dataset)

    adj_entity, adj_relation = read_kg(KG_FILE[dataset], entity_vocab, relation_vocab,
                                       neighbor_sample_size)

    pickle_dump(format_filename(PROCESSED_DATA_DIR, DRUG_VOCAB_TEMPLATE, dataset=dataset),
                drug_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, ENTITY_VOCAB_TEMPLATE, dataset=dataset),
                entity_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, RELATION_VOCAB_TEMPLATE, dataset=dataset),
                relation_vocab)
    adj_entity_file = format_filename(PROCESSED_DATA_DIR, ADJ_ENTITY_TEMPLATE, dataset=dataset)
    np.save(adj_entity_file, adj_entity)
    print('Logging Info - Saved:', adj_entity_file)

    adj_relation_file = format_filename(PROCESSED_DATA_DIR, ADJ_RELATION_TEMPLATE, dataset=dataset)
    np.save(adj_relation_file, adj_relation)
    print('Logging Info - Saved:', adj_entity_file)
    
    drug_feature = read_feature(os.path.join(os.getcwd()+'/raw_data'+'/pca_smiles_kegg.csv'),entity2id,drug_vocab)
    drug_feature_file = format_filename(PROCESSED_DATA_DIR, DRUG_FEATURE_TEMPLATE, dataset=dataset)
    np.save(drug_feature_file, drug_feature,allow_pickle=True)
    print('Logging Info - Saved:', drug_feature_file)


    drug_sim = read_sim(os.path.join(os.getcwd()+'/raw_data'+'/kegg/kegg_sim.csv'),entity2id,drug_vocab)
    drug_sim_file = format_filename(PROCESSED_DATA_DIR, DRUG_SIM_TEMPLATE, dataset=dataset)
    #np.save(drug_sim_file, drug_sim,allow_pickle=True)
    #print('Logging Info - Saved:', drug_sim_file)


    adj = drug_sim
    adj = sp.csr_matrix(adj)
    sp.save_npz(drug_sim_file, adj)  # 保存'''


    cross_validation(K, examples, dataset, neighbor_sample_size)


def cross_validation(K_fold, examples, dataset, neighbor_sample_size):
    #K折交叉验证
    subsets = dict()
    n_subsets = int(len(examples) / K_fold)
    remain = set(range(0, len(examples) - 1))

    for i in reversed(range(0, K_fold - 1)):
        subsets[i] = random.sample(remain, n_subsets)
        remain = remain.difference(subsets[i])
    subsets[K_fold - 1] = remain
    aggregator_types = ['sum', 'concat', 'neigh']
    #aggregator_types = ['concat']

    for t in aggregator_types:
        count = 1
        temp = {'dataset': dataset, 'aggregator_type': t, 'avg_auc': 0.0, 'avg_acc': 0.0, 'avg_f1': 0.0,
                'avg_aupr': 0.0}

        for i in reversed(range(0, K_fold)):
            test_d = examples[list(subsets[i])]
            val_d, test_data = train_test_split(test_d, test_size=0.5)
            train_d = []
            for j in range(0, K_fold):
                if i != j:
                    train_d.extend(examples[list(subsets[j])])
            train_data = np.array(train_d)
            if dataset == 'kegg':
                train_log = train(
                    kfold=count,
                    dataset=dataset,
                    train_d=train_data,
                    dev_d=val_d,
                    test_d=test_data,
                    neighbor_sample_size=neighbor_sample_size,
                    embed_dim=32,
                    n_depth=2,
                    l2_weight=1e-7,
                    lr=2e-2,
                    optimizer_type='adam',
                    batch_size=2048,
                    aggregator_type=t,
                    n_epoch=50,
                    callbacks_to_add=['modelcheckpoint', 'earlystopping'],
                )
            elif dataset == 'pdd':
                train_log = train(
                    kfold=count,
                    dataset=dataset,
                    train_d=train_data,
                    dev_d=val_d,
                    test_d=test_data,
                    neighbor_sample_size=neighbor_sample_size,
                    embed_dim=64,
                    n_depth=2,
                    l2_weight=1e-7,
                    lr=1e-2,
                    optimizer_type='adam',
                    batch_size=1024,
                    aggregator_type=t,
                    n_epoch=50,
                    callbacks_to_add=['modelcheckpoint', 'earlystopping'],
                )
            count += 1
            temp['avg_auc'] = temp['avg_auc'] + train_log['test_auc']
            temp['avg_acc'] = temp['avg_acc'] + train_log['test_acc']
            temp['avg_f1'] = temp['avg_f1'] + train_log['test_f1']
            temp['avg_aupr'] = temp['avg_aupr'] + train_log['test_aupr']
        for key in temp:
            if key == 'aggregator_type' or key == 'dataset':
                continue
            temp[key] = temp[key] / K_fold
        write_log(format_filename(LOG_DIR, RESULT_LOG[dataset]), temp, 'a')
        print(f'Logging Info - {K_fold} fold result: avg_auc: {temp["avg_auc"]}, avg_acc: {temp["avg_acc"]}, avg_f1: {temp["avg_f1"]}, avg_aupr: {temp["avg_aupr"]}')


if __name__ == '__main__':
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(MODEL_SAVED_DIR):
        os.makedirs(MODEL_SAVED_DIR)
    model_config = ModelConfig()
    #process_data('kegg', NEIGHBOR_SIZE['kegg'], 5)
    process_data('pdd', NEIGHBOR_SIZE['pdd'], 5)



