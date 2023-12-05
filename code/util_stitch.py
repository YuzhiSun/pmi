"""该文件主要用来获取和处理并解析各种数据库数据"""
import re

import pandas as pd
from bs4 import BeautifulSoup
import queue
from Bio import Entrez, PDB,ExPASy
import uniprot as uni
import json
import numpy as np
import requests
import threading
import warnings
import os
import sys
from collections import defaultdict

from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
def extractSubcellularLocation(path):
    subcellLocationDF = pd.DataFrame(columns=['id','location'])

    with open(path, 'r', encoding='gbk') as infile:
        id = ''
        location = ''
        for line in infile:
            if line[0] == '(':
                subcellInfo = line.split(' ')
                loctmp = subcellInfo[2:-1]
                locStr = ' '.join(loctmp)
                location = locStr

            elif line[0] == '>':
                id = line[1:].strip()
                subcellLocationDF = subcellLocationDF.append({'id':id, 'location':location}, ignore_index=True)
            else:
                continue
            # data_line = line.strip("\n").split('\t')
            # # print(data_line[0])
            # data.append(data_line[0])
    return subcellLocationDF

"""获取uniprot信息"""
def obtainUniprot(id,fields):
    rootUrl = f'https://rest.uniprot.org/uniprotkb/search?query={id}&fields={fields}'
    response = requests.get(rootUrl)
    if response.status_code == 200:
        res = response.text
        result = json.loads(res)
        return result
    else:
        print(f'{id} has url error and status code is {response.status_code}!')
        return None

def obtainFromUrl():
    url = 'https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=tsv&query=%28Human%29'
    response = requests.get(url)
    if response.status_code == 200:
        res = response.text

        return res
    else:
        print(f'{id} has url error and status code is {response.status_code}!')
        return None

"""获取alphafold信息"""
def obtainAlphaFold(id):
    rootUrl = f'https://alphafold.ebi.ac.uk/files/AF-{id}-F1-model_v4.pdb'
    response = requests.get(rootUrl)
    if response.status_code == 200:
        res = response.text
        return res
    else:
        print(f'{id} has url error and status code is {response.status_code}!')
        return None
"""获取protparam信息"""
def obtainProtparam(id):
    rootUrl = f'https://web.expasy.org/cgi-bin/protparam/protparam1?{id}'
    response = requests.get(rootUrl)
    if response.status_code == 200:
        res = response.text
        return res
    else:
        print(f'{id} has url error and status code is {response.status_code}!')
        return None
"""处理keywords信息"""
def dealKeywords(proteinId ,jsonStr):
    tupleList = []
    try:
        curKeywords = jsonStr['results'][0]['keywords']
        for oneDict in curKeywords:
            key = oneDict['id'] +'>' + oneDict['category']
            value = oneDict['name']
            tupleList.append((key + '|' + value))
        return tupleList
    except:
        print(f'{proteinId} has error at dealKeywords')
        return None
"""处理gene信息"""
def dealGeneInfo(proteinId, jsonStr):
    try:
        geneInfo = jsonStr.get('results')[0].get('genes')[0]
        geneName = geneInfo.get('geneName').get('value')
        try:
            geneOLNmaes =[x.get('value') for x in geneInfo.get('orderedLocusNames')]
        except:
            geneOLNmaes=None
        resStr = f'geneName>{geneName}|orderedLocusNames>{geneOLNmaes}'
        return resStr
    except:
        print(f'{proteinId} has error at dealGeneInfo')
        return None
def dealOrganismInfo(proteinId, jsonStr):
    try:
        organismName = jsonStr.get('results')[0].get('organism').get('scientificName')
        resStr = f'{organismName}'
        return resStr
    except:
        print(f'{proteinId} has error at dealOrganismInfo')
        return None
def dealPDBInfo(proteinId, fieldsReturn):
    pdbInfoList = []
    try:
        res = fieldsReturn.get('results')[0].get('uniProtKBCrossReferences')
        for info in res:
            pdbId = info.get('id')
            method = info.get('properties')[0].get('value')
            resolution = info.get('properties')[1].get('value')
            chains = info.get('properties')[2].get('value')
            pdbInfoList.append((pdbId, method, resolution, chains))
        def sort_by_resolution(x):
            return x[2]
        res = sorted(pdbInfoList, key=sort_by_resolution, reverse=False)[0]
    except:
        print(f'{proteinId} has error at deal pdb info')
        res = None
    return res
def dealAlphaFold(proteinId, fieldsReturn):
    if fieldsReturn:
        res = fieldsReturn
    else:
        print(f'{proteinId} has error at deal alpha info')
        res = None
    return res
"""构建蛋白质列表"""
def make_protein_list(path):
    with open(path, 'r', encoding='gbk') as infile:
        data = []
        for line in infile:
            data_line = line.strip("\n").split('\t')
            data.append(data_line[0])
    return data
def make_protein_list_yeast(path):
    df = pd.read_csv(path, sep='\t')
    protein_list = df['Entry'].to_list()
    return protein_list
"""查找alpha失败的里面pdb是否成功获取到了"""
def findDiffBtwnPDBandAlpha():
    pdb = pd.read_csv('D:\project2023\DGL_Torch\data\\tmp\pdbInfoFailed.csv')
    alpha = pd.read_csv('D:\project2023\DGL_Torch\data\\tmp\\alphaFoldPdbFailed.csv')
    res1 = pd.merge(pdb, alpha, on='0')
    list1 = list(set(alpha['0'].values.tolist()) - set(res1['0'].values.tolist()))
    pd.DataFrame(list1).to_csv('D:\project2023\DGL_Torch\data\\tmp\\pdbInPDBnotInAlpha.csv',index=False)
    print()
"""批量获取keywords信息"""
def writePDB():
    pdbDF = pd.read_csv('D:\project2023\DGL_Torch\data\\tmp\\alphaFoldPdbInfo.csv')
    for idx, info in pdbDF.iterrows():
        id = info['id']
        pdb = info['features']
        with open(f'D:\project2023\DGL_Torch\data\\alphaFoldPDB\\{id}.ent','w') as fileread:
            fileread.write(pdb)
        if idx % 100 == 0:
            print(f'{idx + 1} is ok')
multiResDF = pd.DataFrame(columns=['id', 'features'])
class multiTreadDeal(threading.Thread):
    def __init__(self, idQueue, failedQueue, dealFC,field=None,obtainFC=None,  *args, **kwargs):
        super(multiTreadDeal, self).__init__( *args, **kwargs)
        self.idQueue = idQueue
        self.failedQueue = failedQueue
        self.field = field
        self.dealFC = dealFC
        self.obtainFC = obtainFC
    def run(self) -> None:
        while not self.idQueue.empty():
            global multiResDF
            global x
            proteinId = self.idQueue.get()
            fields = self.field
            try:
                if self.obtainFC is not None:
                    fieldsReturn = self.obtainFC(id=proteinId)
                else:
                    fieldsReturn = obtainUniprot(id=proteinId, fields=fields)
                res = self.dealFC(proteinId, fieldsReturn)
                if res == None:  self.failedQueue.put(proteinId)
                else:
                    glock.acquire()
                    multiResDF = multiResDF.append({'id': proteinId, 'features': res}, ignore_index=True)
                    x += 1
                    if x % 500 == 0:print(f'finished {x}')
                    glock.release()
            except:
                print(f'{proteinId} has error at parse data')
                self.failedQueue.put(proteinId)
keywordsDF = pd.DataFrame(columns=['id', 'features'])
class multiTreadKeywords(threading.Thread):
    def __init__(self, idQueue, failedQueue, *args, **kwargs):
        super(multiTreadKeywords, self).__init__(*args, **kwargs)
        self.idQueue = idQueue
        self.failedQueue = failedQueue
    def run(self) -> None:
        while not self.idQueue.empty():
            global keywordsDF
            global x
            proteinId = self.idQueue.get()
            fields = 'keyword'
            try:
                fieldsReturn = obtainUniprot(id=proteinId, fields=fields)
                res = dealKeywords(proteinId, fieldsReturn)
                if res == None:  self.failedQueue.put(proteinId)
                else:
                    glock.acquire()
                    keywordsDF = keywordsDF.append({'id': proteinId, 'features': res}, ignore_index=True)
                    x += 1
                    if x % 500 == 0:print(f'finished {x}')
                    glock.release()
            except:
                print(f'{proteinId} has error at parse data')
                self.failedQueue.put(proteinId)
"""构造keyword特征"""
def make_features_of_keywords(kind):
    originDF = pd.read_csv(f'../data/stitch/{kind}/features/proteinFeas.csv')
    target_path = f'../data/stitch/{kind}/features/protein_kw_feas.csv'
    keyList = []
    for col in originDF['features']:
        curList =[x.split('|')[0] +':'+ x.split('|')[1] for x in eval(col)]
        keyList += curList
    colums = list(set(keyList))
    kwFeaDF = pd.DataFrame(columns=['id'] + colums)
    kwFeaDF.to_csv(target_path, index=False)
    for index, line in originDF[['id','features']].iterrows():
        proteinId = line['id']
        feaDic = {'id':proteinId}
        for x in eval(line['features']):
            feaDic[x.split('|')[0] +':'+ x.split('|')[1]] = 1
        kwFeaDF = kwFeaDF.append(feaDic, ignore_index=True)
        if index % 51 == 1:
            kwFeaDF.to_csv(target_path,mode='a+',header=False, index=False)
            kwFeaDF = pd.DataFrame(columns=['id'] + colums)
            print(f'deal {index}')
    kwFeaDF.to_csv(target_path, mode='a+', header=False, index=False)
"""读取代谢物列表"""
def obtain_meta_df():
    df = pd.read_csv('../data/self/origin_info/metabolite_info.txt',
                     sep='\t',header=None)
    df.rename(columns={0:'KEGG', 1:'HMDB', 2:'Name', 3:'url',4:'SMILES'}, inplace=True)
    return df[['KEGG','SMILES']]

def obtain_from_PaDEL():
     from PaDEL_pywrapper import PaDEL
     from PaDEL_pywrapper.descriptor import ALOGP, Crippen, FMF
     from rdkit import Chem
     smiles_list = [
         # erlotinib
         "n1cnc(c2cc(c(cc12)OCCOC)OCCOC)Nc1cc(ccc1)C#C",
         # midecamycin
         "CCC(=O)O[C@@H]1CC(=O)O[C@@H](C/C=C/C=C/[C@@H]([C@@H](C[C@@H]([C@@H]([C@H]1OC)O[C@H]2[C@@H]([C@H]([C@@H]([C@H](O2)C)O[C@H]3C[C@@]([C@H]([C@@H](O3)C)OC(=O)CC)(C)O)N(C)C)O)CC=O)C)O)C",
         # selenofolate
         "C1=CC(=CC=C1C(=O)NC(CCC(=O)OCC[Se]C#N)C(=O)O)NCC2=CN=C3C(=N2)C(=O)NC(=N3)N"
     ]
     mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
     descriptors = [ALOGP, Crippen, FMF]
     padel = PaDEL(descriptors)
     res = padel.calculate(mols)
     print(res)
def obtain_from_padelpy(df):
    from padelpy import from_smiles
    dflist = []
    for _, line in df.iterrows():
        KEGG, smiles = line['KEGG'], line['SMILES']
        if smiles == 'NA':continue
        try:
            descriptors = from_smiles(smiles, fingerprints=True)
            descriptors['KEGG'] = KEGG
        except:
            # print(f'{KEGG} has error')
            continue
        dflist.append(pd.DataFrame(pd.Series(descriptors)).T)
        print(f'{KEGG} is ok')
    df = pd.concat(dflist,ignore_index=True)
    cols = list(df.columns.array)
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    return df
def getCurrentFile(rootDir):
    idList = os.listdir(rootDir)  # 列出文件夹下所有的目录与文件
    allFiles, ids = [], []
    for i in range(0, len(idList)):
        path = os.path.join(rootDir, idList[i])
        allFiles.append(path)
        ids.append(idList[i][:-4])
    return ids, allFiles
"""解析pdb文件内容并存储信息"""
def parsePDBinfo():
    pathList = getCurrentFile('D:\project2023\DGL_Torch\data\PDBalphaFold')
    parser = PDB.PDBParser()

    resDF = pd.DataFrame(columns=['proteinId','atom_cnt','atom_chain','atom_res_types','res_types', 'res_per_chain'])
    resDF.to_csv('D:\project2023\DGL_Torch\data\\tmp\\alphaFoldFeas.csv', index=False)
    for i, idx_path_i in enumerate(zip(pathList[0], pathList[1])):
        idx = idx_path_i[0]
        path_i = idx_path_i[1]
        curInfo = parser.get_structure(f'idx',path_i)
        atom_cnt = defaultdict(int)
        atom_chain = defaultdict(int)
        atom_res_types = defaultdict(int)
        for atom in curInfo.get_atoms():
            my_residue = atom.parent
            my_chain = my_residue.parent
            atom_chain[my_chain.id] += 1
            if my_residue.resname != 'HOH':
                atom_cnt[atom.element] += 1
            atom_res_types[my_residue.resname] += 1
        res_types = defaultdict(int)
        res_per_chain = defaultdict(int)
        for residue in curInfo.get_residues():
            res_types[residue.resname] += 1
            res_per_chain[residue.parent.id] += 1

        resDF = resDF.append({'proteinId':str(idx), 'atom_cnt':str(atom_cnt), 'atom_chain':str(atom_chain),'atom_res_types':str(atom_res_types),
                                             'res_types':str(res_types), 'res_per_chain':str(res_per_chain)}, ignore_index=True)
        if i % 100 == 0:
            resDF.to_csv('D:\project2023\DGL_Torch\data\\tmp\\alphaFoldFeas.csv',header=False,mode='a+', index=False)
            resDF = pd.DataFrame(columns=['proteinId', 'atom_cnt', 'atom_chain', 'atom_res_types', 'res_types', 'res_per_chain'])
            print(f'has done {i}')
    resDF.to_csv('D:\project2023\DGL_Torch\data\\tmp\\alphaFoldFeas.csv', header=False, mode='a+', index=False)
"""构造PDB数据特征"""
def make_PDB_feas():
    df = pd.read_csv('D:\project2023\DGL_Torch\data\\tmp\\alphaFoldFeas.csv')
    atom_chain_keys, atom_res_types_keys, res_types_keys, res_per_chain_keys = [], [], [], []
    for idx, line in df[['proteinId','atom_chain','atom_res_types', 'res_types','res_per_chain']].iterrows():
        atom_chain_keys.extend(eval(line['atom_chain'].split('>,')[1][:-1]).keys())
        atom_res_types_keys.extend(eval(line['atom_res_types'].split('>,')[1][:-1]).keys())
        res_types_keys.extend(eval(line['res_types'].split('>,')[1][:-1]).keys())
        res_per_chain_keys.extend(eval(line['res_per_chain'].split('>,')[1][:-1]).keys())
    atom_chain_keys, atom_res_types_keys, res_types_keys, res_per_chain_keys \
        = set(atom_chain_keys), set(atom_res_types_keys), set(res_types_keys), set(res_per_chain_keys)
    atom_df = pd.DataFrame(columns=['proteinId'] + list(atom_chain_keys) + list(atom_res_types_keys))
    res_df = pd.DataFrame(columns=['proteinId'] + list(res_types_keys) + list(res_per_chain_keys))
    atom_df.to_csv('D:\project2023\DGL_Torch\data\\tmp\\alphaFoldAtomFeas.csv', index=False)
    atom_df.to_csv('D:\project2023\DGL_Torch\data\\tmp\\alphaFoldResFeas.csv', index=False)
    for idx, line in df[['proteinId', 'atom_chain', 'atom_res_types', 'res_types', 'res_per_chain']].iterrows():
        proteinId = line['proteinId']
        atom_chain = eval(line['atom_chain'].split('>,')[1][:-1])
        atom_res = eval(line['atom_res_types'].split('>,')[1][:-1])
        res_type = eval(line['res_types'].split('>,')[1][:-1])
        res_chain = eval(line['res_per_chain'].split('>,')[1][:-1])
        atom = {'proteinId':proteinId,**atom_chain, **atom_res}
        res = {'proteinId':proteinId,**res_type, **res_chain}
        atom_df = atom_df.append(atom, ignore_index=True)
        res_df = res_df.append(res, ignore_index=True)
        if idx % 100 == 0:
            atom_df.to_csv('D:\project2023\DGL_Torch\data\\tmp\\alphaFoldAtomFeas.csv',header=False,mode='a+', index=False)
            res_df.to_csv('D:\project2023\DGL_Torch\data\\tmp\\alphaFoldResFeas.csv',header=False,mode='a+', index=False)
            atom_df = pd.DataFrame(columns=['proteinId'] + list(atom_chain_keys) + list(atom_res_types_keys))
            res_df = pd.DataFrame(columns=['proteinId'] + list(res_types_keys) + list(res_per_chain_keys))
            print(f'has done {idx}')
    atom_df.to_csv('D:\project2023\DGL_Torch\data\\tmp\\alphaFoldAtomFeas.csv', header=False, mode='a+', index=False)
    res_df.to_csv('D:\project2023\DGL_Torch\data\\tmp\\alphaFoldResFeas.csv', header=False, mode='a+', index=False)
"""处理ProtParam数据"""
def dealProtPara(id,res):
    try:
        soup = BeautifulSoup(res)
        curInfo = soup.find(name='div', attrs={'id': 'sib_body'}).find(name='pre').find(name='pre').text
        curInfo = curInfo.split('\n')
        feaDict = {}
        flag = 0
        for idx, substr in enumerate(curInfo):
            if flag > 0:
                flag -= 1
                continue
            if substr == '': continue
            if substr.find('\t') != -1 and substr.find(':') != -1:
                key = substr.split()[3]
                try:
                    value1 = substr.split()[5]
                    value2 = substr.split()[6][:-1]
                except:
                    value1 = int(re.findall("\d+", substr.split()[4])[0])
                    value2 = substr.split()[5][:-1]
                feaDict[key + '-nums'] = int(value1)
                feaDict[key + '-percent'] = float(value2) * 0.01
                for line in curInfo[idx + 1: idx + 22]:
                    if line == '': continue
                    try:
                        key, value1, value2 = line.split()[0], int(line.split()[2]), round(float(line.split()[3][:-1]) * 0.01,
                                                                                       2)
                    except:
                        key, value1, value2 = line.split()[0], int(re.findall("\d+", line.split()[1])[0]), round(float(line.split()[2][:-1]) * 0.01,
                                                                                       2)
                    feaDict[key + '-nums'] = value1
                    feaDict[key + '-percent'] = value2
                for line in curInfo[idx + 23: idx + 26]:
                    if line == '': continue
                    key, value1, value2 = line.split()[0], int(line.split()[1]), round(float(line.split()[2][:-1]) * 0.01,
                                                                                       2)
                    feaDict[key + '-nums'] = value1
                    feaDict[key + '-percent'] = value2
                flag = 25

            elif substr.find('Atomic') != -1:
                statis = 0
                for line in curInfo[idx + 2:]:
                    statis += 1
                    if line == '' and statis > 4:
                        break
                    if line == '':continue
                    key, value1 = line.split()[0], int(line.split()[2].strip())
                    feaDict[key] = value1
                flag = statis
            elif substr.find('Ext. coefficient') != -1:
                key, value = substr.split()[1], substr.split()[2]
                if feaDict.get(key) == None:
                    feaDict[key] = value
                else:
                    feaDict[key + '1'] = value
            elif substr.find('Abs') != -1:
                key, value = substr.split()[0], substr.split()[4]
                if feaDict.get(key) == None:
                    feaDict[key] = value[:-1]
                else:
                    feaDict[key + '1'] = value[:-1]
            elif substr.find('N-terminal') != -1:
                key, value = substr.split()[1], substr.split()[7]
                feaDict[key] = value
            elif substr.find('The estimated half-life') != -1:
                m1str, m2str, m3str = substr, curInfo[idx + 1], curInfo[idx + 2]
                m1 = int(re.findall("\d+", substr)[0])
                m2 = int(re.findall("\d+", m2str)[0])
                m3 = int(re.findall("\d+", m3str)[0])
                if m1str.find('hour') != -1:
                    feaDict['half-life1'] = m1 * 60
                elif m1str.find('min') != -1:
                    feaDict['half-life1'] = m1
                else:
                    feaDict['half-life1'] = round(m1 / 60, 2)

                if m2str.find('hour') != -1:
                    feaDict['half-life2'] = m2 * 60
                elif m2str.find('min') != -1:
                    feaDict['half-life2'] = m2
                else:
                    feaDict['half-life2'] = round(m2 / 60, 2)

                if m3str.find('hour') != -1:
                    feaDict['half-life3'] = m3 * 60
                elif m3str.find('min') != -1:
                    feaDict['half-life3'] = m3
                else:
                    feaDict['half-life3'] = round(m3 / 60, 2)

            elif substr.find('instability') != -1:
                info = substr.split()
                key, value = info[1], float(info[8])
                feaDict[key] = value

            elif substr.find('classifies the protein') != -1:
                info = substr.split()
                key, value = info[1], info[5]
                feaDict[key] = value
            elif substr.find(':') != -1:
                key, value = substr.split(':')[0], substr.split(':')[1].strip()
                if value == '':
                    continue
                else:
                    try:
                        feaDict[key] = float(value)
                    except:
                        continue
        return feaDict
    except:
        print(f'deal {id} has error ')
        return None
def make_prot_param_feas():
    df = pd.read_csv('D:\project2023\DGL_Torch\data\\tmp\protParaInfo.csv')
    keys = []
    for idx, line in df.iterrows():
        keys.extend(eval(line['features']).keys())
    keys = ['proteinId'] + list(set(keys))
    protDF = pd.DataFrame(columns=keys)
    protDF.to_csv('D:\project2023\DGL_Torch\data\\tmp\\protParaFeas.csv',index=False)
    for idx, line in df.iterrows():
        proteinId = line['id']
        info = eval(line['features'])
        info['proteinId'] = proteinId
        protDF = protDF.append(info,ignore_index=True)
        if idx % 100 == 0:
            protDF.to_csv('D:\project2023\DGL_Torch\data\\tmp\\protParaFeas.csv',header=False, mode='a+', index=False)
            protDF = pd.DataFrame(columns=keys)
            print(f'has done {idx}')
    protDF.to_csv('D:\project2023\DGL_Torch\data\\tmp\\protParaFeas.csv', header=False, mode='a+', index=False)
"""获取蛋白质uniprot id 和 string对应id"""

def filter_protein_type():
    df = pd.read_csv('../data/self/origin_info/organism.csv')
    mainDf = pd.read_csv('../data/self/origin_info/interaction.txt',
                         sep='\t')[['KEGG', 'Uniprot_KB_id', 'interaction', 'probe']]
    mainDf.rename(columns={'Uniprot_KB_id': 'id'}, inplace=True)
    types = set(df['features'].values)
    for type_str in types:
        df_human = df[df['features'] == type_str]
        human_interactions = pd.merge(mainDf,df_human,on='id',how='inner')
        res = human_interactions.drop(columns=['features'])
        res.to_csv(f'../data/self/origin_info/interaction_{type_str.strip().split()[0]}.csv',index=False)
def make_id_for_protein_and_meta(kind):
    interaction = pd.read_csv(f'../data/stitch/{kind}/origin_info/{kind}_interaction_filtered.csv')
    meta_id = interaction[['cid']].drop_duplicates()
    meta_id = meta_id.sort_values(by=['cid'], ignore_index=True).reset_index()
    protein_id = interaction[['Entry']].drop_duplicates()
    protein_id = protein_id.sort_values(by=['Entry'], ignore_index=True).reset_index()
    meta_id[['index', 'cid']].to_csv(f'../data/stitch/{kind}/net/meta_node_id.csv', index=False)
    protein_id[['index', 'Entry']].to_csv(f'../data/stitch/{kind}/net/protein_node_id.csv', index=False)
    print()

def change_id():
    string_api_url = 'https://version-11-5.string-db.org/api'
    output_format = "tsv-no-header"
    method = 'get_string_ids'
    uniprot_id_list = pd.read_csv(r'../data/snap/origin_info/snap_interaction_filtered.csv')
    uniprot_id_list = list(set(uniprot_id_list['Entry'].to_list()))
    demo_list = uniprot_id_list
    params = {
        "identifiers": "\r".join(demo_list),  # your protein list
        "species": 9606,  # species NCBI identifier
        "limit": 1,  # only one (best) identifier per input protein
        "echo_query": 1,  # see your input identifiers in the output
        "caller_identity": "www.awesome_app.org"  # your app name
    }
    request_url = "/".join([string_api_url, output_format, method])
    results = requests.post(request_url, data=params)
    resDF = pd.DataFrame(columns=['uniprot_id', 'string_id'])
    for idx, line in enumerate(results.text.strip().split("\n")):
        l = line.split("\t")
        try:
            input_identifier, string_identifier = l[0], l[2]
            resDF = resDF.append({'uniprot_id':input_identifier, 'string_id':string_identifier},ignore_index=True)
        except:
            print(f'{l[0]} has error')
            continue
        if idx % 100 == 0:
            print(f'{idx} is ok!')
    resDF.to_csv('../data/snap/origin_info/protein_id_mapping.csv',index=False)
def make_network(kind):
    protein_net = pd.read_csv(f'../data/stitch/{kind}/origin_info/protein_interaction.txt',sep=' ')
    id_mapping = pd.read_csv(f'../data/stitch/{kind}/origin_info/protein_id_mapping.tsv', sep='\t')
    protein_net = pd.merge(id_mapping,protein_net,how='inner',left_on='From',right_on='protein1')
    protein_net = pd.merge(protein_net,id_mapping,how='inner',left_on='protein2',right_on='From')
    protein_net = protein_net[['Entry_x','Entry_y','combined_score']]
    protein_net.to_csv(f'../data/stitch/{kind}/net/protein_interaction_net.csv',index=False)
def standard_numerical_and_one_hot_cols(idList,df,standard=False):
    dfFeas = df.drop(columns=idList)
    numericFeasIdx = dfFeas.dtypes[dfFeas.dtypes != 'object'].index
    if standard:
        dfFeas[numericFeasIdx] = dfFeas[numericFeasIdx].apply(lambda x: (x - x.min()) / (x.max()-x.min()))
    dfFeas[numericFeasIdx] = dfFeas[numericFeasIdx].fillna(0)
    dfFeas = pd.get_dummies(dfFeas, dummy_na=True)
    dfFeas[idList] = df[idList]
    return dfFeas
def pca_deal(df, dim_percent):
    feaNP = np.array(df)
    n_sample = feaNP.shape[0]
    lowFeas = PCA(n_components=dim_percent)
    newFeas = lowFeas.fit_transform(feaNP)
    columns = lowFeas.get_feature_names_out()
    feas = pd.DataFrame(newFeas, columns=columns)
    return feas
def make_train_data(kind,standard=True, pca = False, sample=False):
    interaction_csv_path = f'../data/stitch/{kind}/origin_info/{kind}_interaction_with_neg.csv'
    mainDF = pd.read_csv(interaction_csv_path)
    if sample :
        mainDF = mainDF.sample(frac=sample)
    try:
        mainDF = mainDF.drop(columns=['probe'])
    except:
        print('no probe to deal')

    kwDF = pd.read_csv(f'../data/stitch/{kind}/features/protein_kw_feas.csv')
    metaDF = pd.read_csv(f'../data/stitch/{kind}/features/meta_cid_feas.csv')
    metaGcnDF = pd.read_csv(f'../data/stitch/{kind}/features/meta_gcn_feas.csv')
    proteinGcnDF = pd.read_csv(f'../data/stitch/{kind}/features/protein_gcn_feas.csv')


    kwDF.rename(columns={'id': 'Entry'}, inplace=True)
    kwDF = standard_numerical_and_one_hot_cols(idList=['Entry'], df=kwDF, standard=standard)
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
    label = dfForModel[['label']]
    id = dfForModel[['cid', 'Entry']]
    feas = dfForModel.drop(columns=['cid', 'Entry', 'label'])
    if pca == True:
        feas = pca_deal(feas, 0.99)
    feas = feas.fillna(value=0)
    res = pd.concat([id,label,feas], axis=1)

    res.head(10).to_csv(f'../data/stitch/{kind}/features/train_valid_demo.csv', index=False)
    del mainDF, kwDF, metaDF, proteinGcnDF, metaGcnDF, dfForModel, feas
    return res
def make_train_transfer_data(kind,source,standard=True, pca = False, sample=False):
    interaction_csv_path = f'../data/stitch/{kind}/origin_info/{kind}_interaction_with_neg.csv'
    mainDF = pd.read_csv(interaction_csv_path)
    if sample :
        mainDF = mainDF.sample(frac=sample)
    try:
        mainDF = mainDF.drop(columns=['probe'])
    except:
        print('no probe to deal')

    kwDF = pd.read_csv(f'../data/stitch/{kind}/features/protein_kw_feas.csv')
    metaDF = pd.read_csv(f'../data/stitch/{kind}/features/meta_cid_feas.csv')
    metaGcnDF = pd.read_csv(f'../data/stitch/{source}/features/meta_gcn_feas.csv')
    proteinGcnDF = pd.read_csv(f'../data/stitch/{source}/features/protein_gcn_feas.csv')


    kwDF.rename(columns={'id': 'Entry'}, inplace=True)
    kwDF = standard_numerical_and_one_hot_cols(idList=['Entry'], df=kwDF, standard=standard)
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
    label = dfForModel[['label']]
    id = dfForModel[['cid', 'Entry']]
    feas = dfForModel.drop(columns=['cid', 'Entry', 'label'])
    if pca == True:
        feas = pca_deal(feas, 0.99)
    feas = feas.fillna(value=0)
    res = pd.concat([id,label,feas], axis=1)

    res.head(10).to_csv(f'../data/stitch/{kind}/features/train_valid_transfer_demo.csv', index=False)
    del mainDF, kwDF, metaDF, proteinGcnDF, metaGcnDF, dfForModel, feas
    return res
"""抽取snap数据"""
def meta_filter():
    interactions = pd.read_csv('../data/snap/origin_info/snap_interaction_with_neg.csv')
    meta_inter_cnt = interactions[['cid', 'label']].groupby(by='cid').count()
    sort_res = meta_inter_cnt.sort_values(by='label', ascending=False)
    meta_nums = sort_res.shape[0]
    top_10 = sort_res.iloc[:10, :]
    top_50 = sort_res.iloc[:50,:]
    median_50 = sort_res.iloc[meta_nums//2:meta_nums//2+50,:]
    last_50 = sort_res.iloc[-50:, :]
    rnd_50 = sort_res.sample(50)
    def make_info(target, interactions, type_name='top50'):
        target = target.index.to_list()
        interactions = interactions[interactions['cid'].isin(target)]
        interactions.to_csv(f'../data/snap/origin_info/snap_interaction_{type_name}.csv', index=False)
        meta_id = list(set(interactions['cid'].to_list()))
        meta_id = pd.DataFrame(meta_id, columns=['cid'])
        meta_id.reset_index(inplace=True)
        meta_id.to_csv(f'../data/snap/net/snap_interaction_{type_name}_meta_id.csv',index=False)
        protein_id = list(set(interactions['Entry'].to_list()))
        protein_id = pd.DataFrame(protein_id, columns=['Entry'])
        protein_id.reset_index(inplace=True)
        protein_id.to_csv(f'../data/snap/net/snap_interaction_{type_name}_protein_id.csv', index=False)
    # make_info(top_50, interactions, 'top50')
    # make_info(median_50, interactions, 'median50')
    # make_info(last_50, interactions, 'last50')
    # make_info(rnd_50, interactions, 'rnd50')
    make_info(top_10, interactions, 'top10')
def process_stitch(kind='Arabidopsis'):
    from txdpy import get_num
    """----------change file-----------"""
    origin_interaction = pd.read_csv(f'../data/stitch/{kind}/origin_info/protein_chemical.tsv', sep='\t')
    origin_interaction.rename(columns={'chemical':'cid', 'protein':'stringID'}, inplace=True)
    target_chem = pd.read_csv('../data/stitch/Homo/origin_info/target_metabolite.csv', names=['kegg','cid','hmdb','name','url','SMILES'])
    target_chem = target_chem[['cid','SMILES']]
    # 筛选得分大于600的
    origin_interaction = origin_interaction[origin_interaction['combined_score'] > 600]
    origin_interaction['cid'] = origin_interaction['cid'].apply(lambda x: int(get_num(x)[0])).tolist()
    # 只保留PMIDB中出现的小分子代谢物
    origin_interaction = pd.merge(target_chem, origin_interaction, how='inner',on='cid')
    target_protein = set(origin_interaction['stringID'].tolist())

    pd.DataFrame(list(target_protein)).to_csv(f'../data/stitch/{kind}/origin_info/protein_id.csv',index=False,header=None)
    # 通过uniprot网站在线转换id
    """--------这里暂停一下， 去线上获取unirpot的id--------"""
    protein_id = pd.read_csv(f'../data/stitch/{kind}/origin_info/protein_id_mapping.tsv', sep='\t')
    target_interaction = pd.merge(origin_interaction, protein_id, left_on='stringID', right_on='From', how='left')
    target_interaction = target_interaction[['cid', 'Entry']]
    target_interaction.dropna(inplace=True)
    target_protein = set(target_interaction['Entry'].tolist())
    pd.DataFrame(list(target_protein)).to_csv(f'../data/stitch/{kind}/origin_info/protein_entry.csv',index=False,header=None)

    target_interaction.to_csv(f'../data/stitch/{kind}/origin_info/{kind}_interaction.csv', index=False)
    print()
def interaction_filter(kind):
    all_interaction = pd.read_csv(f'../data/stitch/{kind}/origin_info/{kind}_interaction.csv')
    all_interaction = all_interaction[['cid','Entry']]
    protein_feas_id = pd.read_csv(f'../data/stitch/{kind}/features/protein_kw_feas.csv')
    Entrys = set(protein_feas_id['id'].tolist())
    meta_feas_id = pd.read_csv(f'../data/stitch/Homo/features/metaFea.csv')
    meta_id_mapping = pd.read_csv(f'../data/stitch/Homo/origin_info/target_metabolite.csv', usecols=[0,1], names=['KEGG','cid'])
    meta_features = pd.merge(meta_feas_id, meta_id_mapping, how='inner', left_on='KEGG', right_on='KEGG')
    meta_features = pd.concat([meta_features.iloc[:,-1:], meta_features.iloc[:,1:-1]], axis=1)
    meta_features.to_csv(f'../data/stitch/{kind}/features/meta_cid_feas.csv', index=False)
    Cids = set(meta_features['cid'].tolist())
    target_interaction = all_interaction[all_interaction['cid'].isin(Cids)]
    target_interaction = target_interaction[target_interaction['Entry'].isin(Entrys)]
    target_interaction.drop_duplicates(inplace=True)
    target_interaction.to_csv(f'../data/stitch/{kind}/origin_info/{kind}_interaction_filtered.csv', index=False)
"""构建负集"""
def make_neg(kind):
    import random
    interaction = pd.read_csv(f'../data/stitch/{kind}/origin_info/{kind}_interaction_filtered.csv')
    interaction_simple = interaction[['cid', 'Entry']]
    interaction_simple.drop_duplicates(inplace=True)
    metas = list(set(interaction_simple['cid'].tolist()))
    interaction_simple['label'] = 1
    print(f'all meta nums is {len(metas)}')
    meta_neg_lst, protein_neg_lst = [], []
    for idx, meta in enumerate(metas):
        if idx % 50 == 0: print(f'this is {idx}th meta')
        positive_num = len(set(interaction_simple[interaction_simple['cid'] == meta]['Entry'].tolist()))
        proteins_neg = list(set(interaction_simple[~interaction_simple['Entry'].isin(
            set(interaction_simple[interaction_simple['cid'] == meta]['Entry'].tolist()))]['Entry'].tolist()))
        sample_num = min(len(proteins_neg), positive_num)
        sample_neg_protein = random.sample(proteins_neg, sample_num)
        for protein in sample_neg_protein:
            meta_neg_lst.append(meta)
            protein_neg_lst.append(protein)
    neg_sample_df = pd.DataFrame(data=zip(meta_neg_lst, protein_neg_lst), columns=['cid', 'Entry'])
    neg_sample_df['label'] = 0
    all_sample_df = pd.concat([interaction_simple[['cid', 'Entry', 'label']], neg_sample_df], ignore_index=True)
    test = all_sample_df[['cid', 'Entry']]  # 测试看是否生成的负样本和正样本有交集
    all_sample_df.to_csv(f'../data/stitch/{kind}/origin_info/{kind}_interaction_with_neg.csv', index=False)


if __name__ == '__main__':
    glock = threading.Lock()
    x = 0
    """提取snap互作数据，完成id映射"""
    kind='Mus'
    # process_stitch(kind)
    """获取snap中蛋白质和代谢物的特征，方法同self数据集，代谢物id对应的SMILES在pubchem网站可以在线获取"""
    """获取蛋白质keywords信息"""
    # path = f'../data/stitch/{kind}/origin_info/protein_entry.csv'
    # idList = make_protein_list(path)
    # proteinIdQueue = queue.Queue()
    # failedQueue = queue.Queue()
    # failedList = []
    # threadList = []
    # for proteinId in idList:
    #     proteinIdQueue.put(proteinId)
    # for i in range(50):
    #     th = multiTreadKeywords(idQueue=proteinIdQueue, failedQueue=failedQueue)
    #     threadList.append(th)
    #     th.start()
    # for curthread in threadList:
    #     curthread.join()
    # while not failedQueue.empty():
    #     failedList.append(failedQueue.get())
    # print('writing the result to csv ...')
    # keywordsDF.to_csv(f'../data/stitch/{kind}/features/proteinFeas.csv', index=False)

    """2. 构造蛋白质关键字特征"""
    # make_features_of_keywords(kind)
    """根据可获取的特征对互作关系进行过滤，保留可获取特征的id"""
    interaction_filter(kind)
    """构造负集"""
    # make_neg(kind)
    """构造蛋白质互作网络,需要去STRING数据库下载蛋白质互作网络数据"""
    # make_network(kind)
    """构造训练DGL所需的节点信息"""
    # make_id_for_protein_and_meta(kind)
    """10. 使用dglNet生成节点嵌入特征"""
    """11. 构造用于训练模型的数据, 数据量过大，直接由函数返回给训练模型"""
    # make_train_data(kind)
    """12. 使用k_fold训练模型"""
    """13. 构造Mus数据集，使用Homo GCN 特征"""
    make_train_transfer_data(kind='Mus',source='Homo')


















