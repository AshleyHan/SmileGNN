# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score,precision_recall_curve
import sklearn.metrics as m
from SmileGNN.utils import write_log

#添加指标：ACC, AUPR, AUC-ROC, F1 +std

class KGCNMetric(Callback):
    def __init__(self, x_train, y_train, x_valid, y_valid,aggregator_type,dataset,K_fold,batch_size):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.aggregator_type=aggregator_type
        self.dataset=dataset
        self.k=K_fold
        self.threshold=0.5
        self.batch_size = batch_size
        # self.user_list, self.train_record, self.valid_record, \
        #     self.item_set, self.k_list = self.topk_settings()

        super(KGCNMetric, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        #print(np.size(self.x_valid,axis=1))
        y_pred = self.model.predict(self.x_valid,batch_size = self.batch_size).flatten()
        y_true = self.y_valid.flatten()
        try:
            auc = roc_auc_score(y_true=y_true, y_score=y_pred)# roc曲线的auc
        except ValueError:
            auc = 1
            pass
        precision, recall, _thresholds = precision_recall_curve(y_true=y_true, probas_pred=y_pred)
        aupr=m.auc(recall,precision)
        y_pred = [1 if prob >= self.threshold else 0 for prob in y_pred]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)
        
        print(type(aupr))
        logs['val_aupr']=float(aupr)
        logs['val_auc'] = float(auc)
        logs['val_acc'] = float(acc)
        logs['val_f1'] = float(f1)
        
        logs['dataset']=self.dataset
        logs['aggregator_type']=self.aggregator_type
        logs['kfold']=self.k
        logs['epoch_count']=epoch+1
        print(f'Logging Info - epoch: {epoch+1}, val_auc: {auc}, val_aupr: {aupr}, val_acc: {acc}, val_f1: {f1}')
        log_list =str(np.array(logs).tolist())
        write_log('log/train_history.txt',log_list,mode='a')

    @staticmethod
    def get_user_record(data, is_train):
        user_history_dict = defaultdict(set)
        for interaction in data:
            user = interaction[0]
            item = interaction[1]
            label = interaction[2]
            if is_train or label == 1:
                user_history_dict[user].add(item)
        return user_history_dict

