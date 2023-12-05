import pandas as pd
import torch
from torch.utils import data
from PMI_tidy.code.k_fold_snap import MGAT_GCN_DNN, ProteinMetaDataset
from PMI_tidy.code.util_snap import standard_numerical_and_one_hot_cols, pca_deal


# snap for case study
# find the PMI which is not in snap
def make_interaction():
    already_pmi = pd.read_csv('/home/yuzhi/project2023/PMI_tidy/data/snap/origin_info/snap_interaction_with_neg.csv')
    cid = already_pmi[['cid']].drop_duplicates(ignore_index=True)
    entry = already_pmi[['Entry']].drop_duplicates(ignore_index=True)
    already_pmi['pmi'] = already_pmi['cid'].astype('str') + '-' + already_pmi['Entry']
    all_pmis = cid.assign(temp=1).merge(entry.assign(temp=1)).drop(columns=['temp'])
    all_pmis['pmi'] = all_pmis['cid'].astype('str') + '-' + all_pmis['Entry']
    select_pmi = all_pmis[~all_pmis['pmi'].isin(list(already_pmi['pmi']))].sample(100000)

    select_pmi[['cid','Entry']].to_csv('/home/yuzhi/project2023/PMI_tidy/data/snap/case_study/case_to_predict100k.csv', index=False)
def make_interaction_case2():
    already_pmi = pd.read_csv('/home/yuzhi/project2023/PMI_tidy/data/snap/origin_info/snap_interaction_with_neg.csv')
    cid = pd.read_csv('/home/yuzhi/project2023/PMI_tidy/data/stitch/Arabidopsis/features/meta_cid_feas.csv')
    cid = cid[['cid']].drop_duplicates(ignore_index=True)
    entry = already_pmi[['Entry']].drop_duplicates(ignore_index=True)
    already_pmi['pmi'] = already_pmi['cid'].astype('str') + '-' + already_pmi['Entry']
    all_pmis = cid.assign(temp=1).merge(entry.assign(temp=1)).drop(columns=['temp'])
    all_pmis['pmi'] = all_pmis['cid'].astype('str') + '-' + all_pmis['Entry']
    select_pmi = all_pmis[~all_pmis['pmi'].isin(list(already_pmi['pmi']))].sample(50000)
    select_pmi[['cid','Entry']].to_csv('/home/yuzhi/project2023/PMI_tidy/data/snap/case_study/case_to_predict2_100k.csv', index=False)
def make_train_data(interaction_csv_path='../data/snap/origin_info/snap_interaction_with_neg.csv',
                                   standard=False, pca = False, sample=0.3):
    mainDF = pd.read_csv(interaction_csv_path)
    if sample :
        mainDF = mainDF.sample(frac=sample)
    try:
        mainDF = mainDF.drop(columns=['probe'])
    except:
        print('no probe to deal')


    kwDF = pd.read_csv('../data/snap/features/snap_protein_kw_feas.csv')
    metaDF = pd.read_csv('../data/snap/features/meta_feas.csv')

    # metaGcnDF = pd.read_csv('../data/snap/features/meta_snap_gcn_feas.csv')
    # proteinGcnDF = pd.read_csv('../data/snap/features/protein_snap_gcn_feas.csv')
    # metaDF = pd.read_csv('../data/snap/features/meta_feas_for_case.csv')
    metaGcnDF = pd.read_csv('../data/snap/features/meta_snap_gcn_feas_for_case.csv')
    proteinGcnDF = pd.read_csv('../data/snap/features/protein_snap_gcn_feas_for_case.csv')


    kwDF.rename(columns={'id': 'Entry'}, inplace=True)
    kwDF = standard_numerical_and_one_hot_cols(idList=['Entry'], df=kwDF, standard=standard)
    metaDF.rename(columns={'KEGG': 'cid'}, inplace=True)
    metaDF = standard_numerical_and_one_hot_cols(idList=['cid'],df=metaDF, standard=standard)
    metaGcnDF.drop(columns=['index'],inplace=True)
    metaGcnDF = standard_numerical_and_one_hot_cols(idList=['cid'], df=metaGcnDF, standard=standard)
    proteinGcnDF.drop(columns=['index'], inplace=True)
    proteinGcnDF = standard_numerical_and_one_hot_cols(idList=['Entry'], df=proteinGcnDF, standard=standard)


    mainDF = pd.merge(mainDF, metaDF, on='cid',how='left')
    mainDF = pd.merge(mainDF, kwDF, on='Entry', how='left')
    mainDF = pd.merge(mainDF, metaGcnDF, on='cid', how='left')
    mainDF = pd.merge(mainDF, proteinGcnDF, on='Entry', how='left')

    mainDF.drop_duplicates(inplace=True, ignore_index=True)
    dfForModel = mainDF.fillna(value=0)

    id = dfForModel[['cid', 'Entry']]
    feas = dfForModel.drop(columns=['cid', 'Entry'])
    if pca == True:
        feas = pca_deal(feas, 0.99)
    feas = feas.fillna(value=0)
    res = pd.concat([id,feas], axis=1)

    res.head(10).to_csv('../data/snap/train/case_demo.csv', index=False)
    del mainDF, kwDF, metaDF, proteinGcnDF, metaGcnDF, dfForModel, feas
    return res
import numpy as np
def valid_auc(dataloader, model, device):
    model.eval()
    lst = []
    with torch.no_grad():
        for X in dataloader:
            X = X.to(device)
            pred = model(X)
            prob = pred.cpu().numpy()
            lst.append(prob)
    res = np.vstack(lst)
    return res
def predict(feature_matrix, label_matrix, model_name, event_num,
                      device,  batch_size, drop_rate, vector_size_input,model_path):

    label_matrix = np.array(label_matrix)
    feature_matrix = np.array(feature_matrix)

    x_test = feature_matrix
    y_test = label_matrix
    vector_size = vector_size_input

    test_dataset = ProteinMetaDataset(x_test, y_test,predict=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4,shuffle=False)
    print(f"Using {device} device")
    model = model_name(drop_rate, vector_size)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    pred = valid_auc(test_loader,model,device)
    pred_score = pred / 1
    pred_type = np.argmax(pred_score, axis=1)
    pred_score_path = f'../data/snap/predictRes/case_study2_1000k.csv'
    wfp = open(pred_score_path, 'w')
    for i in range(len(y_test)):
        res = str(pred_score[i][0]) + ',' + str(pred_score[i][1]) + ',' +  str(pred_type[i]) + ',' + str(y_test[i][0])+','+ str(y_test[i][1]) + '\n'
        wfp.write(res)
    wfp.close()

def pipline():
    model_name = MGAT_GCN_DNN
    model_path = '/home/yuzhi/project2023/PMI_tidy/data/snap/model/MGAT_GCN_DNN/2.pt'
    event_num = 2
    droprate = 0.3
    batch_size = 32
    # interaction_path = '/home/yuzhi/project2023/PMI_tidy/data/snap/case_study/case_to_predict1000k.csv'
    interaction_path = '/home/yuzhi/project2023/PMI_tidy/data/snap/case_study/case_to_predict2_100k.csv'

    data_for_train = make_train_data(interaction_path, standard=True, pca=False, sample=False)
    feature_matrix, label_matrix = data_for_train.drop(columns=['cid', 'Entry']).to_numpy(), data_for_train[
        ['cid', 'Entry']].to_numpy()
    feature_matrix[np.isinf(feature_matrix)] = 0.0
    vector_size = feature_matrix.shape[1]
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    predict(feature_matrix=feature_matrix, label_matrix=label_matrix,
            model_name=model_name, event_num=event_num,
            batch_size=batch_size, device=device, drop_rate=droprate,
            vector_size_input=vector_size, model_path=model_path)


def find_meta_not_in_snap():
    target_metas = pd.read_csv('/home/yuzhi/project2023/PMI_tidy/data/stitch/Homo/origin_info/target_metabolite.csv',
                               names=['id','cid','a','b','c','d'])
    exist_metas = pd.read_csv('/home/yuzhi/project2023/PMI_tidy/data/snap/origin_info/snap_cid.csv',names=['cid'])
    exist_metas_lst = list(exist_metas['cid'].values)
    not_exist_metas = target_metas[~target_metas['cid'].isin(exist_metas_lst)]
    return not_exist_metas[['cid']]
    # not_exist_metas[['id','cid']].to_csv('/home/yuzhi/project2023/PMI_tidy/data/snap/case_study/not_exist_cid.csv', index=False)

def make_id_for_protein_and_meta():
    interaction = pd.read_csv(r'../data/snap/origin_info/snap_interaction_filtered.csv')
    other_meta = find_meta_not_in_snap()
    meta_id = interaction[['cid']].drop_duplicates()
    meta_id_concat = pd.concat([meta_id,other_meta],ignore_index=True)
    meta_id_concat = meta_id_concat.sort_values(by=['cid'], ignore_index=True).reset_index()
    protein_id = interaction[['Entry']].drop_duplicates()
    protein_id = protein_id.sort_values(by=['Entry'], ignore_index=True).reset_index()
    meta_id_concat[['index', 'cid']].to_csv('../data/snap/net/meta_node_id_for_case.csv', index=False)
    protein_id[['index', 'Entry']].to_csv('../data/snap/net/protein_node_id_for_case.csv', index=False)
    print()
def concat_feas():
    snap_metas = pd.read_csv('/home/yuzhi/project2023/PMI_tidy/data/snap/features/meta_feas.csv')
    pmidb_metas = pd.read_csv('/home/yuzhi/project2023/PMI_tidy/data/stitch/Arabidopsis/features/meta_cid_feas.csv')
    pmidb_metas.rename(columns={'cid':'KEGG'},inplace=True)
    concat_metas = pd.concat([snap_metas,pmidb_metas],ignore_index=True)
    concat_metas.to_csv('/home/yuzhi/project2023/PMI_tidy/data/snap/features/meta_feas_for_case.csv',index=False)
    print()
def temp():
    dt = pd.read_csv('/home/yuzhi/project2023/PMI_tidy/data/stitch/Saccharomyces/origin_info/Saccharomyces_interaction.csv')
    print(len(set(list(dt['Entry'].values))))
    print()
if __name__ == '__main__':
    # make_interaction()
    """make data for case study"""
    # make_train_data('/home/yuzhi/project2023/PMI_tidy/data/snap/case_study/case_to_predict.csv', standard=True,
    #                 pca=False, sample=False)
    # case_snap_pmi_not_exist()
    # find_meta_not_in_snap()
    # make_id_for_protein_and_meta()
    # concat_feas()
    # make_interaction_case2()
    pipline()
    # temp()