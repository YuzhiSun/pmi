import numpy as np
import pandas as pd
import torch as th
import dgl
from sklearn.decomposition import PCA


def pca(feasDF,dims):
    feaNP = np.array(feasDF)
    n_sample = feaNP.shape[0]
    low_feas_num = min(n_sample, dims)
    low_feas = PCA(n_components=low_feas_num)
    newFeas = low_feas.fit_transform(feaNP)
    newFeas = newFeas / newFeas.max(axis=0)
    return newFeas


def make_graph(fea_dims=20,device=None):
    """蛋白质索引"""
    protein_node_DF = pd.read_csv('../data/stitch/Homo/net/protein_node_id.csv')
    protein_node_DF.rename(columns={'index':'protein_index','Entry':'protein_id'}, inplace=True)
    protein_nums = protein_node_DF.shape[0]

    """代谢物索引"""
    meta_node_DF = pd.read_csv('../data/stitch/Homo/net/meta_node_id.csv')
    meta_node_DF.rename(columns={'index':'meta_index','cid':'meta_id'}, inplace=True)
    meta_nums = meta_node_DF.shape[0]

    """蛋白质和代谢物边"""
    edge_pm_DF = pd.read_csv('../data/stitch/Homo/origin_info/Homo_interaction_with_neg.csv')
    edge_pm_DF.rename(columns={'Entry': 'protein_id','cid':'meta_id'}, inplace=True)
    edge_pm_DF = edge_pm_DF[edge_pm_DF['label'] == 1]
    edge_pm_DF.drop_duplicates(inplace=True)

    """蛋白质间作用边"""
    edge_pp_DF = pd.read_csv(r'../data/stitch/Homo/net/protein_interaction_net.csv')

    """构造蛋白质相互作用网络"""
    p_e_p_df = pd.merge(edge_pp_DF,protein_node_DF,how='inner', left_on='Entry_x',right_on='protein_id')
    p_e_p_df.rename(columns={'protein_index':'u'},inplace=True)
    p_e_p_df = pd.merge(p_e_p_df, protein_node_DF,how='inner', left_on='Entry_y', right_on='protein_id')
    p_e_p_df.rename(columns={'protein_index':'v'},inplace=True)
    p_e_p_df.dropna(inplace=True)

    """构造蛋白质代谢物相互作用网络"""
    p_e_m_df = pd.merge(edge_pm_DF, protein_node_DF, how='left', left_on='protein_id', right_on='protein_id')
    p_e_m_df = pd.merge(p_e_m_df, meta_node_DF, how='left', left_on='meta_id', right_on='meta_id')

    p_e_m_p_node = th.tensor(p_e_m_df['protein_index'].values)
    p_e_m_m_node = th.tensor(p_e_m_df['meta_index'].values)

    p_e_p_lp_node = th.tensor(p_e_p_df['u'].values)
    p_e_p_rp_node = th.tensor(p_e_p_df['v'].values)


    protein_rnd_feas = torch.randn(protein_nums, fea_dims)
    """需要考虑是否进行pca降维"""
    meta_rnd_feas = torch.randn(meta_nums, fea_dims)


    graph_data = {
        ('protein','pminteracts','meta'):(p_e_m_p_node, p_e_m_m_node),
        ('meta','mpinteracts','protein'):(p_e_m_m_node, p_e_m_p_node),
        ('protein','ppinteracts','protein'):(p_e_p_lp_node, p_e_p_rp_node),
        ('protein','ppinteracts','protein'):(p_e_p_rp_node, p_e_p_lp_node)
    }
    node_nums = {'protein':protein_nums, 'meta':meta_nums}
    g = dgl.heterograph(graph_data,node_nums)
    g.nodes['protein'].data['rnd'] = th.tensor(protein_rnd_feas)
    g.nodes['meta'].data['rnd'] = th.tensor(meta_rnd_feas)
    g = g.to(device=device)
    print(g.ntypes)
    print(g.etypes)
    print(g.canonical_etypes)
    return g

import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
import torch

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, agg_fn='sum'):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names
        }, aggregate=agg_fn)
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names
        }, aggregate=agg_fn)
    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h是从5.1节的GNN模型中计算出的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']

def construct_negative_graph(graph, k, etype, device):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,), device=device)
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype:graph.num_nodes(ntype) for ntype in graph.ntypes},
        device=device
    )
class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names, agg_fn):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names,agg_fn)
        self.pred = HeteroDotProductPredictor()
    def forward(self, g, neg_g, x, etype):
        h = self.sage(g, x)
        return self.pred(g, h, etype), self.pred(neg_g, h, etype)
def compute_loss(pos_score, neg_score):
    """间隔损失"""
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()

def parse_node_embeddings(node_embeddings):
    meta_np = node_embeddings.get('meta').cpu().detach().numpy()
    protein_np = node_embeddings.get('protein').cpu().detach().numpy()
    meta_embedding_df = pd.DataFrame(meta_np).reset_index()
    protein_embedding_df = pd.DataFrame(protein_np).reset_index()
    meta_node_id_map = pd.read_csv('../data/stitch/Homo/net/meta_node_id.csv')
    protein_node_id_map = pd.read_csv('../data/stitch/Homo/net/protein_node_id.csv')
    meta_embedding_df = pd.merge(meta_node_id_map, meta_embedding_df, how='left', on='index')
    protein_embedding_df = pd.merge(protein_node_id_map, protein_embedding_df, how='left', on='index')
    meta_embedding_df.to_csv('../data/stitch/Homo/features/meta_snap_gcn_feas.csv', index=False)
    protein_embedding_df.to_csv('../data/stitch/Homo/features/protein_snap_gcn_feas.csv', index=False)


if __name__ == '__main__':
    fea_dims = 100
    hidden_fea_dims = 50
    out_fea_dims = 10
    k = 10
    epochs = 500
    min_loss, patient_nums, patient = 0, 20, 20

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hetero_graph = make_graph(fea_dims,device)

    agg_fn = 'mean'
    model = Model(fea_dims,hidden_fea_dims,out_fea_dims,hetero_graph.etypes, agg_fn)
    model.to(device)

    for name, param in model.named_parameters():
        print(name, '-->', param.type(), '-->', param.dtype, '-->', param.shape)

    # 随机特征
    protein_feas = torch.tensor(hetero_graph.nodes['protein'].data['rnd'], dtype=torch.float, device=device)
    meta_feas = torch.tensor(hetero_graph.nodes['meta'].data['rnd'], dtype=torch.float, device=device)
    node_features = {'protein':protein_feas, 'meta':meta_feas}
    opt = torch.optim.Adam(model.parameters())

    import time
    start = time.time()
    for epoch in range(epochs):
        negative_graph_pm = construct_negative_graph(hetero_graph, k, ('protein', 'pminteracts', 'meta'), device=device)
        negative_graph_mp = construct_negative_graph(hetero_graph, k, ('meta', 'mpinteracts', 'protein'), device=device)

        pos_score_pm, neg_score_pm = model(hetero_graph, negative_graph_pm, node_features, ('protein', 'pminteracts', 'meta'))
        pos_score_mp, neg_score_mp = model(hetero_graph, negative_graph_mp, node_features, ('meta', 'mpinteracts', 'protein'))
        pos_score = torch.cat((pos_score_pm, pos_score_mp), 0)
        neg_score = torch.cat((neg_score_pm, neg_score_mp), 0)
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_val = loss.item()
        if epoch == 0:
            min_loss = loss_val
        elif loss_val < min_loss:
            min_loss = loss_val
            patient = patient_nums
        else:
            patient -= 1
        print(f'epoch:{epoch}:loss:{round(loss.item(),4)}')
        if patient == 0: break
    end = time.time()
    print(f'total time {end - start}s')

    #保存模型
    path = '../data/stitch/Homo/model/dgl_model_homo.pt'
    torch.save(model.state_dict(), path)

    model = Model(fea_dims,hidden_fea_dims,out_fea_dims,hetero_graph.etypes, agg_fn)
    model.load_state_dict(torch.load(path))
    model.eval()
    model.to(device)
    for name, param in model.named_parameters():
        print(name, '-->', param.type(), '-->', param.dtype, '-->', param.shape)
    node_embeddings = model.sage(hetero_graph, node_features)
    parse_node_embeddings(node_embeddings)
    print()

