from functools import reduce
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
# Added by myself
import dgl
from dgl.nn import GATConv
import dgl.function as fn
from dgl.nn import APPNPConv


class APPNP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(APPNP, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = APPNPConv(k = 4, alpha = 0.0000)
        self.conv3 = GraphConv(hidden_dim, out_dim)

    def forward(self, g, features):
        h1 = self.conv1(g, features)
        h1 = F.dropout(h1, training = self.training)
        h2 = self.conv2(g, h1)
        h2 = F.dropout(h2, training = self.training)
        h = self.conv3(g, h2)
        #h = F.dropout(h, training = self.training)
        return h






