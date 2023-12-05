import os
import time

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize, normalize
from torch.utils import data
import torch
from torch import nn
from torch.nn import functional as F

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
class ProteinMetaDataset(data.Dataset):
    def __init__(self, x,y):
        self.Data = np.asarray(x)
        self.Label = np.asarray(y)
    def __getitem__(self, idx):
        txt = torch.tensor(self.Data[idx], dtype=torch.float)
        label = torch.tensor(self.Label[idx], dtype=torch.int64)
        return txt, label
    def __len__(self):
        return len(self.Data)
class MLP_cons(nn.Module):
    def __init__(self, drop,in_features):
        super().__init__()
        #out_features = out_features or in_features
        self.mlp = nn.Sequential(nn.Linear(in_features, 64), nn.ReLU(), nn.Linear(64, 2), nn.LogSoftmax(dim=1))

    def forward(self, x):
        return self.mlp(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0., pred=True):
        super().__init__()
        #out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.q = nn.Linear(in_features, in_features)
        self.k = nn.Linear(in_features, in_features)
        self.v = nn.Linear(in_features, in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.pred = pred
        if pred==True:
            self.fc2 = nn.Linear(hidden_features,2)
        else:
            self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x0 = x
        q = self.q(x).unsqueeze(2)
        k = self.k(x).unsqueeze(2)
        v = self.v(x).unsqueeze(2)
        attn = (q @ k.transpose(-2, -1))
        #print(attn.size())
        attn = attn.softmax(dim=-1)
        x = (attn @ v).squeeze(2)
        #print(x.size())
        x += x0
        x1 = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.pred==False:
            x += x1

        x = x.squeeze(0)

        return x


class TF(nn.Module):
    def __init__(self,  drop, in_features):
        super().__init__()
        self.features_size = 128
        self.input = nn.Linear(in_features,self.features_size)
        self.Block1 = Mlp(in_features=int(self.features_size), hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        # self.Block1_1 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        # self.Block1_2 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        # self.Block1_3 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        # self.Block1_1 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        # self.Block1_1 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        # self.Block1_1 = Mlp(in_features=in_features, hidden_features=64, act_layer=nn.GELU, drop=drop, pred=False)
        self.Block2 = Mlp(in_features=int(self.features_size), hidden_features=64, act_layer=nn.GELU, drop=drop, pred=True)
        self.out = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.input(x)
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.out(x)
        return x


class DNN(nn.Module):
    def __init__(self,drop_rate, vector_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(vector_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512,512),
            nn.Dropout(drop_rate),

            nn.Linear(512,256),
            nn.ReLU(),
            nn.BatchNorm1d(256,256),
            nn.Dropout(droprate),

            nn.Linear(256, 2),
            nn.LogSoftmax(dim=1)

        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
#弃用
class MGAT_GCN(nn.Module):
    def __init__(self,drop_rate, vector_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.linear_kw = nn.Linear(684,128)
        self.linear_meta = nn.Linear(2756,128)
        self.linear_gcn = nn.Linear(40,5)
        self.head1 = nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.head2 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.head3 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.linear_head = nn.Linear(73728,512)
        self.linear_stack_kw = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(128, 128),
            nn.Dropout(droprate),
        )
        self.linear_stack_meta = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(128, 128),
            nn.Dropout(droprate),
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512,512),
            nn.Dropout(drop_rate),

            nn.Linear(512,256),
            nn.ReLU(),
            nn.BatchNorm1d(256,256),
            nn.Dropout(droprate),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64, 64),
            nn.Dropout(droprate),

        )
        self.output = nn.Sequential(
            nn.Linear(104,2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        kw = x[:,2756:3440]
        meta = x[:,:2756]
        gcn_feas = x[:,3440:]

        kw = self.linear_kw(kw)
        meta = self.linear_meta(meta)
        kw_ = kw[:,:,None]
        meta_ = meta[:,None,:]
        att = torch.bmm(kw_,meta_)
        att = att[:,None,:,:]
        head1 = self.head1(att)
        head2 = self.head2(att)
        head3 = self.head3(att)
        att = torch.cat((head1, head2, head3),1)
        att = self.linear_head(att)
        kw = self.linear_stack_kw(kw)
        meta = self.linear_stack_meta(meta)
        all = torch.cat((kw,meta,att),1)
        all = self.linear_relu_stack(all)
        all = torch.cat((all, gcn_feas),1)
        all = self.output(all)
        return all
class MGAT_GCN_DNN(nn.Module):
    def __init__(self,drop_rate, vector_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.linear_kw = nn.Linear(741,8)
        self.linear_meta = nn.Linear(2756,8)
        self.head1 = nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.head2 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.head3 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.linear_head = nn.Sequential(nn.Linear(288,128), nn.ReLU())
        self.layer1 = nn.Sequential(nn.Linear(vector_size + 128, 1024), nn.BatchNorm1d(1024), nn.Dropout(drop_rate))
        self.layer2 = nn.Sequential(nn.Linear(1024, 300), nn.BatchNorm1d(300), nn.Dropout(drop_rate))
        self.out = nn.Sequential(nn.Linear(300, 2))
    def forward(self, x):
        kw = x[:,2756:-20]
        meta = x[:,:2756]
        kw = self.linear_kw(kw)
        meta = self.linear_meta(meta)
        kw_ = kw[:,:,None]
        meta_ = meta[:,None,:]
        att = torch.bmm(kw_,meta_)
        att = att[:,None,:,:]
        head1 = self.head1(att)
        head2 = self.head2(att)
        head3 = self.head3(att)
        att = torch.cat((head1, head2, head3),1)
        att = self.linear_head(att)
        all = torch.cat((x, att),1)
        all = F.relu(self.layer1(all))
        all = F.relu(self.layer2(all))
        all = F.log_softmax(self.out(all))
        return all
class GCN_DNN(nn.Module):
    def __init__(self,drop_rate, vector_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.linear_kw = nn.Linear(741,8)
        self.linear_meta = nn.Linear(2756,8)
        self.head1 = nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.head2 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.head3 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.linear_head = nn.Sequential(nn.Linear(288,8), nn.ReLU())
        self.layer1 = nn.Sequential(nn.Linear(3533, 1024), nn.BatchNorm1d(1024), nn.Dropout(drop_rate))
        self.layer2 = nn.Sequential(nn.Linear(1024, 300), nn.BatchNorm1d(300), nn.Dropout(drop_rate))
        self.out = nn.Sequential(nn.Linear(300, 2))
    def forward(self, x):
        kw = x[:,2756:-20]
        meta = x[:,:2756]
        kw = self.linear_kw(kw)
        meta = self.linear_meta(meta)
        all = torch.cat((x, kw, meta),1)
        all = F.relu(self.layer1(all))
        all = F.relu(self.layer2(all))
        all = F.softmax(self.out(all), dim=1)
        return all
class DNN_all(nn.Module):
    def __init__(self, drop_rate, vector_size):
        super(DNN_all, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(vector_size, 1024), nn.BatchNorm1d(1024))
        self.layer2 = nn.Sequential(nn.Linear(1024, 300), nn.BatchNorm1d(300))
        self.out = nn.Sequential(nn.Linear(300, 2))
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.out(x), dim=1)
        return x
#弃用
class MGAT(nn.Module):
    def __init__(self,drop_rate, vector_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.linear_kw = nn.Linear(741,128)
        self.linear_meta = nn.Linear(2756,128)
        self.linear_gcn = nn.Linear(20,5)
        self.head1 = nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.head2 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.head3 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.linear_head = nn.Linear(73728,512)
        self.linear_stack_kw = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(128, 128),
            nn.Dropout(droprate),
        )
        self.linear_stack_meta = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(128, 128),
            nn.Dropout(droprate),
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512,512),
            nn.Dropout(drop_rate),

            nn.Linear(512,256),
            nn.ReLU(),
            nn.BatchNorm1d(256,256),
            nn.Dropout(droprate),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64, 64),
            nn.Dropout(droprate),

        )
        self.output = nn.Sequential(
            nn.Linear(64,2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        kw = x[:,2756:-20]
        meta = x[:,:2756]

        kw = self.linear_kw(kw)
        meta = self.linear_meta(meta)
        kw_ = kw[:,:,None]
        meta_ = meta[:,None,:]
        att = torch.bmm(kw_,meta_)
        att = att[:,None,:,:]
        head1 = self.head1(att)
        head2 = self.head2(att)
        head3 = self.head3(att)
        att = torch.cat((head1, head2, head3),1)
        att = self.linear_head(att)
        kw = self.linear_stack_kw(kw)
        meta = self.linear_stack_meta(meta)
        all = torch.cat((kw,meta,att),1)
        all = self.linear_relu_stack(all)
        all = self.output(all)
        return all
def evaluate(pred_type, pred_score, y_test, event_num):
    y_one_hot = label_binarize(y_test, classes=np.arange(event_num + 1))
    y_one_hot = y_one_hot[:, [0, 1]]

    result_auc_micro = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_auc_macro = roc_auc_score(y_one_hot, pred_score, average='macro')
    return result_auc_micro, result_auc_macro

def train(dataloader, model, loss_fn, optimizer,device, print_res):
    model.train()
    train_loss = 0
    train_acc = 0
    start = time.time()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        out = model(X)
        loss = loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred == y).sum().item()
        acc = num_correct / X.shape[0]
        train_acc += acc
    end = time.time()
    cur_loss = train_loss / len(dataloader)
    cur_acc = train_acc / len(dataloader)

    print(f'train loss: {cur_loss:.7f} , train acc: {cur_acc:.4f} {(end - start):.2f} seconds')

def test(dataloader, model, loss_fn, device, print_res):
    eval_loss = 0
    eval_acc = 0
    model.eval()
    y_true, y_hat = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = loss_fn(out, y)
            eval_loss += loss.item()
            _, pred = out.max(1)
            num_correct = (pred == y).sum().item()
            acc = num_correct / X.shape[0]
            eval_acc += acc
            prob = out.cpu().numpy()
            y_hat.extend(np.argmax(prob, axis=1))
            y_true.extend(y.cpu())
    test_loss_mean = eval_loss / len(dataloader)
    correct_mean = eval_acc / len(dataloader)

    acc = accuracy_score(y_true, y_hat)
    auc_ = roc_auc_score(y_true, y_hat)
    precision, recall, thresholds = precision_recall_curve(y_true, y_hat)
    aupr = auc(recall, precision)

    print(f'test accuracy: {(100*correct_mean):.5f}%, avg loss:{test_loss_mean:.5f} , '
      f'acc:{(acc*100):.2f}%, auc: {auc_:.2f}, aupr:{aupr:.2f}')
    return test_loss_mean

def valid_auc(dataloader, model, device):
    model.eval()
    lst = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            prob = pred.cpu().numpy()
            lst.append(prob)
    res = np.vstack(lst)
    return res


def cross_validation(feature_matrix, label_matrix, model_name, event_num, seed,
                     CV, device, epochs, batch_size, early_stop, patient, min_loss,
                     drop_rate, vector_size_input, lr, model_res_fname, kind):
    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.zeros((0, event_num), dtype=float)
    label_matrix = np.array(label_matrix)
    feature_matrix = np.array(feature_matrix)
    index_all_class = get_index(label_matrix, event_num, seed, CV)

    for k in range(CV):
        lr_k = lr
        patient_k = patient
        train_index = np.where(index_all_class != k)
        test_index = np.where(index_all_class == k)
        pred = np.zeros((len(test_index[0]), event_num), dtype=float)
        y_train = label_matrix[train_index]
        x_train = feature_matrix[train_index]
        x_test = feature_matrix[test_index]
        y_test = label_matrix[test_index]
        vector_size = vector_size_input


        y_train = y_train[:,2]
        y_test = y_test[:,2]
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)


        train_dataset = ProteinMetaDataset(x_train, y_train)
        test_dataset = ProteinMetaDataset(x_test, y_test)
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
        print(f"Using {device} device")
        model = model_name(drop_rate, vector_size)
        model = model.to(device)

        # loss_fn = nn.CrossEntropyLoss()
        loss_fn = nn.NLLLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_k)
        print_res = False

        os.makedirs(f'../data/self/model/{model_res_fname}', exist_ok=True)
        model_save_path = f'../data/self/model/{model_res_fname}/{str(k)}.pt'
        for t in range(epochs):
            if t%5==0:
                optimizer.param_groups[0]['lr'] *= 0.9
            cur_lr = optimizer.param_groups[0]['lr']
            print(f'------------------------\n epoch {t + 1} lr: {cur_lr}  ')

            train(train_loader, model, loss_fn, optimizer,device,print_res)
            cur_loss = test(test_loader, model, loss_fn, device, print_res)

            if t == 0:
                min_loss = cur_loss
                print(f'save model ...')
                torch.save(model.state_dict(), model_save_path)
            elif cur_loss < min_loss:
                min_loss = cur_loss
                patient_k = early_stop
                print(f'save model ...')
                torch.save(model.state_dict(),model_save_path)
            else:
                patient_k -= 1
            if patient_k == 0 and cur_lr < 0.0001:
                break
        model = model_name(drop_rate, vector_size)
        model.load_state_dict(torch.load(model_save_path))
        model = model.to(device)
        pred += valid_auc(test_loader,model,device)
        pred_score = pred / 1
        pred_type = np.argmax(pred_score, axis=1)
        y_true = np.hstack((y_true, y_test))
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.row_stack((y_score, pred_score))
        os.makedirs(f'../data/self/predictRes/{model_res_fname}', exist_ok=True)
        pred_score_path = f'../data/self/predictRes/{model_res_fname}/{str(k)}.txt'
        wfp = open(pred_score_path, 'w')
        for i in range(len(y_test)):
            res = str(pred_score[i][0]) + ' ' + str(pred_score[i][1]) + ' ' + str(y_test[i]) + '\n'
            wfp.write(res)
        wfp.close()
        result_micro, result_macro = evaluate(pred_type, pred_score, y_test, event_num)
        print(f"idx:{k}, auc_micro:{result_micro}, auc_macro:{result_macro}")
    auc_ = roc_auc_score(y_true, y_pred)
    acc_ = accuracy_score(y_true, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    aupr = auc(recall, precision)
    result_all_micro, result_all_macro = evaluate(y_pred, y_score, y_true, event_num)
    print(f"auc_micro_all:{result_all_micro:.4f}, auc_macro_all:{result_all_macro:.4f}, "
          f"acc_all:{acc_:.4f}, auc_all:{auc_:.4f}, aupr_all:{aupr:.4f} ")

if __name__ == '__main__':
    """结果文件名称"""
    model_res_fname = 'TF_homo'
    kind = None
    """选择使用的模型"""
    model_name = TF
    event_num = 2
    droprate = 0.3
    seed = 6
    CV = 10
    batch_size = 128
    epoch = 50
    lr = 0.1
    early_stop, patient, min_loss = 20, 20, 0
    "特征文件"
    data_for_train = pd.read_csv('../data/self/train/self_train_valid_stdd.csv')

    feature_matrix, label_matrix = data_for_train.drop(columns=['cid', 'Entry', 'label']).to_numpy(), data_for_train[['cid', 'Entry', 'label']].to_numpy()
    feature_matrix[np.isinf(feature_matrix)] = 0.0
    vector_size = feature_matrix.shape[1]
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    cross_validation(feature_matrix=feature_matrix, label_matrix=label_matrix,
                     model_name=model_name, event_num=event_num, seed=seed, CV=CV,
                     batch_size=batch_size, device=device, epochs=epoch, early_stop=early_stop,
                     patient=patient, min_loss=min_loss, drop_rate=droprate, vector_size_input=vector_size,
                     lr=lr, model_res_fname=model_res_fname,kind=kind)