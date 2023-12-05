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
def make_features_of_keywords():
    originDF = pd.read_csv('../data/self/features/proteinFeas.csv')
    target_path = '../data/self/features/protein_kw_feas.csv'
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
def make_id_for_protein_and_meta():
    proteinDF = pd.read_csv('../data/self/origin_info/protein_infor.txt',
                            sep='\t',names=['id','gene','other','url'])
    sortDF = proteinDF.sort_values(by=['id'],ignore_index=True).reset_index()
    sortDF[['index','id']].to_csv('../data/self/net/protein_node_id.csv',index=False)
    metaDF = pd.read_csv('../data/self/origin_info/metabolite_info.txt',
                         sep='\t',
                         names=['id','name1','name2','name3','name4'])
    metaSortDF = metaDF.sort_values(by=['id'],ignore_index=True).reset_index()
    metaSortDF[['index','id']].to_csv('../data/self/net/meta_node_id.csv',
                                      index=False)
    print()
def find_one_type_protein(type_str = 'Homo sapiens'):
    df = pd.read_csv('../data/self/origin_info/organism.csv')
    df = df[df['features'] == type_str]
    df['id'].to_csv('../data/self/origin_info/human_protein.csv',header=False,index=False)
def change_id():
    string_api_url = 'https://version-11-5.string-db.org/api'
    output_format = "tsv-no-header"
    method = 'get_string_ids'
    uniprot_id_list = pd.read_csv(r'../data/self/origin_info/human_protein.csv',header=None)
    uniprot_id_list = uniprot_id_list[0].tolist()
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
    for line in results.text.strip().split("\n"):
        l = line.split("\t")
        input_identifier, string_identifier = l[0], l[2]
        resDF = resDF.append({'uniprot_id':input_identifier, 'string_id':string_identifier},ignore_index=True)
    resDF.to_csv('../data/self/origin_info/id_mapping.csv',index=False)
def make_network():
    protein_net = pd.read_csv(r'../data/self/origin_info/9606.protein.physical.links.v11.5.txt',sep=' ')
    id_mapping = pd.read_csv(r'../data/self/origin_info/id_mapping.csv')
    protein_net = pd.merge(id_mapping,protein_net,how='inner',left_on='string_id',right_on='protein1')
    protein_net = pd.merge(protein_net,id_mapping,how='inner',left_on='protein2',right_on='string_id')
    protein_net = protein_net[['uniprot_id_x','uniprot_id_y','combined_score']]
    protein_net.to_csv('/home/yuzhi/project2023/PMI_tidy/data/self/net/protein_interaction_net.csv',index=False)
def standard_numerical_and_one_hot_cols(idList,df,standard=False):
    dfFeas = df.drop(columns=idList)
    numericFeasIdx = dfFeas.dtypes[dfFeas.dtypes != 'object'].index
    if standard:
        dfFeas[numericFeasIdx] = dfFeas[numericFeasIdx].apply(lambda x: (x - x.min()) / (x.max()-x.min()))
    dfFeas[numericFeasIdx] = dfFeas[numericFeasIdx].fillna(0)
    dfFeas = pd.get_dummies(dfFeas, dummy_na=True)
    dfFeas[idList] = df[idList]
    return dfFeas
def make_train_data(interaction_csv_path='../data/self/origin_info/interaction_Mus.csv',
                                   standard=False):
    mainDF = pd.read_csv(interaction_csv_path)
    try:
        mainDF = mainDF.drop(columns=['probe'])
    except:
        print('no probe to deal')

    kwDF = pd.read_csv('../data/self/features/protein_kw_feas.csv')
    metaDF = pd.read_csv('../data/self/features/metaFeas.csv')
    metaGcnDF = pd.read_csv('../data/self/features/meta_gcn_embedding_feas.csv')
    proteinGcnDF = pd.read_csv('../data/self/features/protein_gcn_emgbedding_feas.csv')

    mainDF.rename(columns={'KEGG': 'cid'}, inplace=True)
    mainDF.rename(columns={'id': 'Entry'}, inplace=True)
    kwDF.rename(columns={'id': 'Entry'}, inplace=True)
    kwDF = standard_numerical_and_one_hot_cols(idList=['Entry'], df=kwDF, standard=standard)
    metaDF.rename(columns={'KEGG': 'cid'}, inplace=True)
    metaDF = standard_numerical_and_one_hot_cols(idList=['cid'],df=metaDF, standard=standard)
    metaGcnDF.drop(columns=['index'],inplace=True)
    metaGcnDF.rename(columns={'id': 'cid'}, inplace=True)
    metaGcnDF = standard_numerical_and_one_hot_cols(idList=['cid'], df=metaGcnDF, standard=standard)
    proteinGcnDF.drop(columns=['index'], inplace=True)
    proteinGcnDF.rename(columns={'id': 'Entry'}, inplace=True)
    proteinGcnDF = standard_numerical_and_one_hot_cols(idList=['Entry'], df=proteinGcnDF, standard=standard)


    mainDF = pd.merge(mainDF, metaDF, on='cid',how='left')
    mainDF = pd.merge(mainDF, kwDF, on='Entry', how='left')
    mainDF = pd.merge(mainDF, metaGcnDF, on='cid', how='left')
    mainDF = pd.merge(mainDF, proteinGcnDF, on='Entry', how='left')

    mainDF.drop_duplicates(inplace=True, ignore_index=True)
    dfForModel = mainDF.fillna(value=0)
    label = dfForModel[['interaction']]
    label['label'] = label[['interaction']].astype('int')
    label = label[['label']]
    id = dfForModel[['cid', 'Entry']]
    feas = dfForModel.drop(columns=['cid', 'Entry', 'interaction'])
    feas = feas.fillna(value=0)
    res = pd.concat([id,label,feas], axis=1)
    # Homo 训练数据
    # res.to_csv('../data/self/train/self_train_valid_stdd.csv', index=False)
    # Mus 训练数据
    res.to_csv('../data/self/train/self_train_valid_Mus_stdd.csv', index=False)

if __name__ == '__main__':
    glock = threading.Lock()
    x = 0
    """1. 获取蛋白质keywords信息"""
    # path = '../data/self/origin_info/protein_infor.txt'
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
    # keywordsDF.to_csv('../data/self/features/proteinFeas.csv', index=False)
    # pd.DataFrame(failedList).to_csv('../data/self/features/proteinFeas_failed.csv', index=False)
    """2. 构造蛋白质关键字特征"""
    # make_features_of_keywords()
    """3. 获取代谢物的分子描述符和指纹信息"""
    metaDF = obtain_meta_df()
    # res = obtain_from_padelpy(metaDF)
    # res.to_csv('../data/self/features/metaFeas.csv', index=False)
    """4. 对蛋白质数据进行按物种分类"""
    # path = '../data/self/origin_info/protein_infor.txt'
    # resTarPath = '../data/self/origin_info/organism.csv'
    # failTarPath = '../data/self/origin_info/organismFailed.csv'
    # threadNums = 50
    # idList = make_protein_list(path)
    # field = 'organism_name'
    # proteinIdQueue = queue.Queue()
    # failedQueue = queue.Queue()
    # failedList = []
    # threadList = []
    # for proteinId in idList:
    #     proteinIdQueue.put(proteinId)
    # for i in range(threadNums):
    #     th = multiTreadDeal(idQueue=proteinIdQueue, failedQueue=failedQueue, field=field, dealFC=dealOrganismInfo)
    #     threadList.append(th)
    #     th.start()
    # for curthread in threadList:
    #     curthread.join()
    # while not failedQueue.empty():
    #     failedList.append(failedQueue.get())
    # multiResDF.to_csv(resTarPath, index=False)
    # pd.DataFrame(failedList).to_csv(failTarPath, index=False)
    """5. 按物种划分数据集"""
    # filter_protein_type()
    """6. 提取人类蛋白质列表"""
    # find_one_type_protein()
    """7. 获取蛋白质id映射"""
    #也可以在uniprot网站直接进行线上转换
    # change_id()
    """8. 构造蛋白质互作网络"""
    make_network()
    """9. 构造训练DGL所需的节点信息"""
    # make_id_for_protein_and_meta()
    """10. 使用dglNet生成节点嵌入特征"""
    """11. 构造用于训练模型的数据"""
    make_train_data(standard=True)
    """12. 使用k_fold训练模型"""
    """13. 构造Mus数据集，使用Homo GCN 特征"""
    # make_train_data(interaction_csv_path='../data/self/origin_info/interaction_Mus.csv')


















