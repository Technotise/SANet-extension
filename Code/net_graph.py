import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class Graph:
    def __init__(self, strategy='uniform', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge()
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self):
        self.num_node = 14
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [(0, 1), (1, 3), (0, 2), (2, 4), (0, 13), (5, 13),
                         (6, 13), (5, 7), (7, 9), (6, 8), (8, 10), (12, 13), (11, 13)]
        self.edge = self_link + neighbor_link
        self.center = 13

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)
        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

class GCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, device, stride=1, partition=3, dropout=0):
        super(GCN_Block, self).__init__()
        self.device = device
        self.partition = partition
        self.skeleton_graph = Graph(strategy='spatial', max_hop=1, dilation=1)
        self.A = torch.tensor(self.skeleton_graph.A, dtype=torch.float32, requires_grad=False, device=self.device)
        self.edge_importance = nn.Parameter(torch.ones(self.A.size(), device=self.device))
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * partition, stride=1, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels * partition),
            nn.ReLU(inplace=True)
        ).to(self.device)
        self.temporal_block = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=(stride, 1), kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        ).to(self.device)
        if out_channels == in_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
                nn.BatchNorm2d(out_channels)
            ).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        res = self.residual(x)
        x = self.conv_block(x)
        b, pc, f, j = x.size()
        x = x.view(b, self.partition, pc // self.partition, f, j)
        edge_weight = self.A * self.edge_importance
        x = torch.einsum('bpcfj,pjw->bcfw', (x, edge_weight))
        x = self.temporal_block(x) + res
        return x

class GraphNet(nn.Module):
    def __init__(self, in_channels, device, num_joints=14, partition=3):
        super(GraphNet, self).__init__()
        self.device = device
        self.GCN_Block1 = GCN_Block(in_channels, 8, device, stride=1, partition=partition).to(device)
        self.GCN_Block2 = GCN_Block(8, 8, device, stride=1, partition=partition).to(device)
        self.GCN_Block3 = GCN_Block(8, 16, device, stride=2, partition=partition).to(device)
        self.GCN_Block4 = GCN_Block(16, 16, device, stride=1, partition=partition).to(device)
        self.GCN_Block5 = GCN_Block(16, 32, device, stride=2, partition=partition).to(device)
        self.GCN_Block6 = GCN_Block(32, 32, device, stride=1, partition=partition).to(device)
        self.GCN_Block7 = GCN_Block(32, 64, device, stride=2, partition=partition).to(device)
        self.fc = nn.Linear(64, 1).to(device)
        self.attention = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5, inplace=True).to(device)
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints).to(device)

    def forward(self, x):
        x = x.to(self.device)
        batch, channel, clip_length, num_joints = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(batch, num_joints * channel, clip_length)
        x = self.data_bn(x)
        x = x.view(batch, num_joints, channel, clip_length).permute(0, 2, 3, 1).contiguous()
        x = self.GCN_Block1(x)
        x = self.GCN_Block2(x)
        x = self.GCN_Block3(x)
        x = self.GCN_Block4(x)
        x = self.GCN_Block5(x)
        x = self.GCN_Block6(x)
        x = self.GCN_Block7(x)
        batch, channel, t, joints = x.size()
        x = F.max_pool2d(x, (t, joints))
        x = x.view(batch, -1)
        x = self.fc(self.dropout(x))
        att = self.attention(x)
        return x, att
