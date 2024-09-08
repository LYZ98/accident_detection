import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from module import DisScore, Pool, TCN, STconv_D, STconv_G

class Discriminator(torch.nn.Module):
    def __init__(self, args, node_num, device):
        super(Discriminator, self).__init__()
        self.week_onehot = F.one_hot(torch.arange(0, 7)).to(device)
        self.hour_onehot = F.one_hot(torch.arange(0, 24)).to(device)
        self.minute_onehot = F.one_hot(torch.arange(0, 12)).to(device)
        self.S_emb = nn.Embedding(node_num, args.hidden_size)
        self.encoder = nn.Sequential(nn.Linear(args.hidden_size, args.hidden_size), nn.Tanh(), nn.Linear(args.hidden_size, args.hidden_size), nn.Tanh())
        self.tanh = nn.Tanh()
        self.gconv = GCNConv(args.hidden_size, args.hidden_size)
        self.out = nn.Linear(4 * args.hidden_size, args.hidden_size)
        self.fc = nn.Linear(43, args.hidden_size)
        self.st_encoder = STconv_D(args.hidden_size, args.seq_num)
        self.pool = Pool(in_channels=args.hidden_size)
        self.tcn = TCN(args.hidden_size, args.hidden_size, [args.hidden_size]*1, 2, 0.2)
        self.BCEloss = nn.BCELoss()
        self.score = DisScore(args.hidden_size, args.hidden_size)

    def forward(self, x):
        Z = self.encoder(x)
        return Z

    def context_gen(self, graph_mask, graph_h, time_dayofweek, time_hour, time_minute, s_index, batch=None):
        """subgraph context"""
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        Z_c = self.encoder(graph_mask)
        Z_h = self.encoder(graph_h)
        hidden_c = self.tanh(self.gconv(Z_c, edge_index, edge_attr))
        hidden_h = self.st_encoder(Z_h, edge_index, edge_attr)
        week_onehot = self.week_onehot.index_select(0, time_dayofweek)
        hour_onehot = self.hour_onehot.index_select(0, time_hour)
        minute_onehot = self.minute_onehot.index_select(0, time_minute)
        s_emb = self.S_emb(s_index)
        time_emb = self.fc(torch.cat((week_onehot, hour_onehot, minute_onehot), dim=1).to(torch.float32))
        context_h = self.pool(hidden_h, edge_index, batch.batch)
        context_c = self.pool(hidden_c, edge_index, batch.batch)
        context = torch.cat((context_c, context_h, time_emb, s_emb), dim=1)
        context = self.out(context)
        return context

    def ano_score(self, hidden, context):
        out = self.score(hidden, context)
        return out

    def loss(self, score, label):
        return self.BCEloss(score, label)


class Generator(torch.nn.Module):
    def __init__(self, args, node_num, device):
        super(Generator, self).__init__()
        self.args = args
        self.device = device
        self.week_onehot = F.one_hot(torch.arange(0, 7)).to(device)
        self.hour_onehot = F.one_hot(torch.arange(0, 24)).to(device)
        self.minute_onehot = F.one_hot(torch.arange(0, 12)).to(device)
        self.S_emb = nn.Embedding(node_num, args.hidden_size)
        self.tanh = nn.Tanh()
        self.gconv = GCNConv(args.hidden_size, args.hidden_size)
        self.fc = nn.Linear(43, args.hidden_size)
        self.out = nn.Linear(5 * args.hidden_size, args.hidden_size)
        self.st_encoder = STconv_G(hidden_size=args.hidden_size, seq_len=args.seq_len)
        self.pool = Pool(in_channels=args.seq_len)
        self.tcn = TCN(args.seq_len, args.seq_len, [args.hidden_size] * 1, 2, 0.2)
        self.BCEloss = nn.BCELoss()


    def forward(self, graph_c, graph_h, time_dayofweek, time_hour, time_minute, s_index, batch, index):
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        hidden_c = self.tanh(self.gconv(graph_c, edge_index, edge_attr))
        hidden_h = []

        for i in range(self.args.seq_num):
            out = self.st_encoder(graph_h[:, i, :], edge_index, edge_attr)
            hidden_h.append(out)

        hidden_h = torch.stack(hidden_h).squeeze(dim=2).transpose(0, 1)
        out_h = hidden_h[index]

        context_h = self.pool(hidden_h, edge_index, batch.batch)
        context_c = self.pool(hidden_c, edge_index, batch.batch)
        week_onehot = self.week_onehot.index_select(0, time_dayofweek)
        hour_onehot = self.hour_onehot.index_select(0, time_hour)
        minute_onehot = self.minute_onehot.index_select(0, time_minute)
        s_emb = self.S_emb(s_index)
        time_emb = self.fc(torch.cat((week_onehot, hour_onehot, minute_onehot), dim=1).to(torch.float32))
        z = torch.rand(time_emb.size(0), self.args.hidden_size).to(self.device)

        context = torch.cat((context_c, context_h, time_emb, s_emb, z), dim=1)
        fake = self.out(self.tanh(context)) + out_h

        return fake



