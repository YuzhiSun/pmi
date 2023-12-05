import os

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
import torch

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from PMI.utils2 import concat_and_proprecess_features
from PMI_tidy.code.util_stitch import make_train_data


def get_index(label_matrix, event_num, seed, CV):
    index_all_class = np.zeros(len(label_matrix))
    for j in range(event_num):
        index = np.where(label_matrix == j)
        kf = KFold(n_splits=CV, shuffle=True, random_state=seed)
        k_num = 0
        for train_index, test_index in kf.split(range(len(index[0]))):
            index_all_class[index[0][test_index]] = k_num
            k_num += 1
    return index_all_class
def evaluate(pred_type, pred_score, y_test, event_num):
    y_one_hot = label_binarize(y_test, classes=np.arange(event_num + 1))
    y_one_hot = y_one_hot[:, [0, 1]]

    result_auc_micro = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_auc_macro = roc_auc_score(y_one_hot, pred_score, average='macro')
    return result_auc_micro, result_auc_macro
def cross_validation(feature_matrix,label_matrix,event_num,seed,
                     CV,kind,model_name):
    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.zeros((0, event_num), dtype=float)
    label_matrix = np.array(label_matrix)
    feature_matrix = np.array(feature_matrix)
    feature_matrix = feature_matrix[:,-20:]
    index_all_class = get_index(label_matrix, event_num, seed, CV)

    for k in range(CV):
        train_index = np.where(index_all_class != k)
        test_index = np.where(index_all_class == k)
        pred = np.zeros((len(test_index[0]), event_num), dtype=float)
        x_train = feature_matrix[train_index]
        x_test = feature_matrix[test_index]
        y_train = label_matrix[train_index]
        y_test = label_matrix[test_index]
        eval_set = [(x_test, y_test)]
        # model = XGBClassifier(
        #     n_estimators = 100,
        #     max_depth = 5,
        #     learning_rate = 0.1
        # )
        #
        # model.fit(x_train, y_train,
        #           early_stopping_rounds = 20,
        #           eval_metric = 'logloss',
        #           eval_set = eval_set,
        #           verbose=True )
        # RF
        model = RandomForestClassifier()
        model.fit(x_train, y_train)

        # 决策树
        # model = DecisionTreeClassifier()
        # model.fit(x_train, y_train)



        prob = model.predict_proba(x_test)

        # 给出预测的分类概率
        pred += prob
        pred_score = pred / 1
        pred_type = np.argmax(pred_score, axis=1)
        y_true = np.hstack((y_true, y_test))
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.row_stack((y_score, pred_score))
        os.makedirs(f'../data/stitch/{kind}/predictRes/{model_name}', exist_ok=True)
        pred_score_path = f'../data/stitch/{kind}/predictRes/{model_name}/{str(k)}.txt'
        wfp = open(pred_score_path, 'w')
        for i in range(len(y_test)):
            res = str(pred_score[i][0]) + ' ' + str(pred_score[i][1]) + ' ' + str(y_test[i]) + '\n'
            wfp.write(res)
        wfp.close()
        auc_ = roc_auc_score(y_test, pred_type)
        precision, recall, thresholds = precision_recall_curve(y_test, pred_type)
        aupr = auc(recall, precision)
        result_micro, result_macro = evaluate(pred_type, pred_score, y_test, event_num)
        print(f"idx:{k}, auc_micro:{result_micro:.4f}, auc_macro:{result_macro:.4f}, "
              f"auc_all:{auc_:.4f}, aupr_all:{aupr:.4f} ")
    auc_ = roc_auc_score(y_true, y_pred)
    acc_ = accuracy_score(y_true, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    aupr = auc(recall, precision)
    result_all_micro, result_all_macro = evaluate(y_pred, y_score, y_true, event_num)
    print(f"auc_micro_all:{result_all_micro:.4f}, auc_macro_all:{result_all_macro:.4f}, "
          f"acc_all:{acc_:.4f}, auc_all:{auc_:.4f}, aupr_all:{aupr:.4f} ")

if __name__ == '__main__':
    model_name = 'RF'
    event_num = 2
    seed = 6
    CV = 10
    # pca_for_data()
    # D:\project2023\DGL_Torch\data\k_fold_data\matrix_pca_res.csv
    # data_for_train = pd.read_csv('/home/yuzhi/project2023/data/feas/kw_meta_gcn_snap.csv')
    # interaction_file = '../data/snap/origin_info/snap_interaction_with_neg.csv'

    kind='Mus'
    data_for_train = make_train_data(kind, True)
    # data_for_train = pd.read_csv('../data/self/train/self_train_valid_Mus.csv')
    feature_matrix, label_matrix = data_for_train.drop(columns=['cid', 'Entry', 'label']).to_numpy(), data_for_train['label'].to_numpy()
    feature_matrix = feature_matrix[:,:-40]
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    cross_validation(feature_matrix=feature_matrix,label_matrix=label_matrix,
                     event_num=event_num,seed=seed, CV=CV,kind=kind,model_name=model_name)