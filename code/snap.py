import pandas as pd

def extract_target_meta():
    meta = pd.read_csv('D:\project2023\DGL_Torch\data\pretion_metabolin\metabolite_info_txt.csv', header=None)
    meta = meta[1].astype('int').tolist()

    snap_meta = pd.read_csv('D:\project2023\DGL_Torch\data\Snap\ChG-InterDecagon_targets.csv', skiprows=1,names=['cid','gene'])
    snap_meta['id'] = snap_meta['cid'].apply(lambda x: int(x[3:]))
    snap_meta = snap_meta[snap_meta['id'].isin(meta)]
    print()
def make_snap_meta_and_protein_list():
    snap_meta = pd.read_csv('D:\project2023\DGL_Torch\data\Snap\ChG-InterDecagon_targets.csv', skiprows=1,names=['cid','gene'])

    target_meta = set(snap_meta['cid'].apply(lambda x: int(x[3:])).tolist())
    target_protein = set(snap_meta['gene'].tolist())
    # pd.DataFrame(list(target_protein)).to_csv('D:\project2023\DGL_Torch\data\Snap\snap_protein_id.csv',index=False,header=None)
    # pd.DataFrame(list(target_meta)).to_csv('D:\project2023\DGL_Torch\data\Snap\snap_cid.csv',index=False)
    snap_info = snap_meta
    snap_info['meta_id'] = snap_meta['cid'].apply(lambda x: int(x[3:]))
    uniprot = pd.read_csv('D:\project2023\DGL_Torch\data\Snap\snap_protein_id_uniportkb.tsv', sep='\t')
    snap_info = pd.merge(snap_info, uniprot, left_on='gene', right_on='From', how='left')
    snap_info.drop(columns=['From'],inplace=True)
    snap_info.to_csv('D:\project2023\DGL_Torch\data\Snap\snap_interaction.csv', index=False)
    print()

def deal_fail():
    fail_meta = pd.read_csv(r'D:\project2023\DGL_Torch\data\Snap\failed_meta.csv',index_col=0,skiprows=1, names=['meta'])
    fail_protein = pd.read_csv('D:\project2023\DGL_Torch\data\Snap\snap_protein_kw_failed.csv', index_col=0, skiprows=1, names=['protein'])
    fail_meta = fail_meta['meta'].tolist()
    fail_protein = fail_protein['protein'].tolist()
    interaction = pd.read_csv('D:\project2023\DGL_Torch\data\Snap\snap_interaction.csv')
    interaction_target = interaction[~(interaction['meta_id'].isin(fail_meta) | interaction['Entry'].isin(fail_protein))]
    interaction_target.to_csv('D:\project2023\DGL_Torch\data\Snap\snap_interaction_filtered.csv', index=False)

    print()
def sample_neg():
    import random
    interaction = pd.read_csv('D:\project2023\DGL_Torch\data\Snap\snap_interaction_filtered.csv')
    interaction_simple = interaction[['cid','Entry']]
    # interaction = interaction[interaction_simple.duplicated(keep=False)]  # 检查重复行
    interaction_simple.drop_duplicates(inplace=True)
    metas = list(set(interaction_simple['cid'].tolist()))
    proteins = set(interaction_simple['Entry'].tolist())
    interaction_simple['label'] = 1
    print(f'all meta nums is {len(metas)}')
    meta_neg_lst, protein_neg_lst = [], []
    for idx, meta in enumerate(metas):
        if idx % 50 == 0: print(f'this is {idx}th meta')
        positive_num = len(set(interaction_simple[interaction_simple['cid'] == meta]['Entry'].tolist()))
        proteins_neg = list(set(interaction_simple[~interaction_simple.isin(set(interaction_simple[interaction_simple['cid'] == meta]['Entry'].tolist()))]['Entry'].tolist()))
        sample_num = min(len(proteins_neg), positive_num)
        sample_neg_protein = random.sample(proteins_neg, sample_num)
        for protein in sample_neg_protein:
            meta_neg_lst.append(meta)
            protein_neg_lst.append(protein)
    neg_sample_df = pd.DataFrame(data=zip(meta_neg_lst, protein_neg_lst),columns=['cid', 'Entry'])
    neg_sample_df['label'] = 0
    all_sample_df = pd.concat([interaction_simple[['cid','Entry','label']], neg_sample_df],ignore_index=True)
    test = all_sample_df[['cid', 'Entry']] # 测试看是否生成的负样本和正样本有交集
    all_sample_df.to_csv('D:\project2023\DGL_Torch\data\Snap\snap_all_interaction.csv', index=False)
    print()

def protein_id_mapping():
    protein_ids = pd.read_csv('D:\project2023\DGL_Torch\data\Snap\snap_protein_id_uniportkb.tsv', sep='\t')
    protein_ids['Entry'].to_csv('D:\project2023\DGL_Torch\data\Snap\snap_protein_entry_ids.csv', index=False)
    # 在uniprot中进行id映射得到映射结果


    print()
if __name__ == '__main__':
    # extract_target_meta()
    # make_snap_meta_and_protein_list()
    # sample_neg()
    protein_id_mapping()