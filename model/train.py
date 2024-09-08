from utils import *
import torch_geometric.utils as tgu
import torch
import torch.nn as nn
import copy
import networkx as nx
import pandas as pd
import numpy as np
import random
import argparse
from gan import Discriminator, Generator
import matplotlib.pyplot as plt


def get_parser():
    parser = argparse.ArgumentParser(description='Description: Script to run our model.')
    parser.add_argument('--dataset', help='bay', default='bay')
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--subgraph_size', type=int, help='subgraph size', default=7)
    parser.add_argument('--n_order', type=int, help='hops of neighbor nodes', default=3)
    parser.add_argument('--hidden_size', type=int, help='hidden size of encoder', default=32)
    parser.add_argument('--seq_len', type=int, help='sequence length', default=12)
    parser.add_argument('--seq_num', type=int, help='number of sequences', default=12)
    parser.add_argument('--step', type=int, help='step of sequence', default=1)
    parser.add_argument('--model_name', help='model name', default="GAN")
    parser.add_argument('--epochs', type=int, help='epoch', default=5)
    parser.add_argument('--lambda_', type=int, help='lambda', default=10)
    return parser

def data_preprocess(path):
    adj_matrix = pd.read_csv(path + f"data/adj.csv", delimiter=',', index_col=0).to_numpy()
    time_series = pd.read_csv(path + f"data/traffic_flow.csv", delimiter=',', index_col=0)

    adj_matrix, remove = process_adj(adj_matrix)
    node_num = adj_matrix.shape[0]
    Graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())
    graph = tgu.from_networkx(Graph, group_edge_attrs=all)
    subgraph = Subgraph(node_num, graph.edge_index, graph.edge_attr, path + f"subgraph/...", adj_matrix, args)
    subgraph.extract()

    date = copy.deepcopy(time_series.index)
    date = pd.to_datetime(date).to_frame()
    dayofweek, hour, minute = date[0].dt.dayofweek, date[0].dt.hour, date[0].dt.minute
    dayofweek = dayofweek.values
    hour = hour.values
    minute = minute.values / 5
    time_series = time_series.to_numpy()
    time_series = np.delete(time_series, remove, 1)
    ma, mi = np.max(time_series), np.min(time_series)
    time_series = (time_series - mi) / (ma - mi)

    data = dict()
    data['time_series'], data['dayofweek'],  data['hour'], data['minute'], data['subgraph'] = time_series, dayofweek, hour, minute, subgraph

    return data, node_num

def load_train_data(path, month, year, num_batch=500):
    train_dataloader = []
    train_data, node_num = data_preprocess(path, month, year)
    time_series, dayofweek, hour, minute, subgraph = (train_data['time_series'], train_data['dayofweek'], train_data['hour'], train_data['minute'], train_data['subgraph'])
    time_series_len = time_series.shape[0]
    s_index = torch.IntTensor(list(range(node_num))).repeat(args.batch_size)

    for i in range(num_batch):
        t_sample = random.sample(range(args.seq_len + args.step * args.seq_num, time_series_len), args.batch_size)
        time_hour, time_dayofweek, time_minute = hour[t_sample], dayofweek[t_sample], minute[t_sample]
        time_dayofweek = torch.IntTensor(time_dayofweek.repeat(node_num))
        time_hour = torch.IntTensor(time_hour.repeat(node_num))
        time_minute = torch.IntTensor(time_minute.repeat(node_num))

        x = []
        for t in t_sample:
            x.append(time_series[t - args.step * args.seq_num - args.seq_len: t, :].T)

        x = np.array(x)
        x = torch.Tensor(x)
        x = x.reshape(args.batch_size * node_num, args.seq_len + args.step * args.seq_num).to(device)
        subgraph.feature_build(x)

        batch, index = subgraph.search(range(node_num))
        graph_h, graph_c = sliding_window(batch.x, args.step, args.seq_num, args.seq_len)
        feature = graph_c[index]

        graph_mask = copy.deepcopy(graph_c)
        graph_mask[index] *= 0

        batch_data = dict()
        batch_data['graph_mask'], batch_data['graph_h'], batch_data['fea'], batch_data['graph_c'] = graph_mask, graph_h, feature, graph_c
        batch_data['time_d'], batch_data['time_h'], batch_data['time_m'] = time_dayofweek, time_hour, time_minute
        batch_data['batch'], batch_data['index'] = batch, index
        batch_data['s_index'] = s_index

        train_dataloader.append(batch_data)

    return train_dataloader, node_num


class GAN_Train:
    def __init__(self, args, node_num, device):
        self.args = args
        self.D = Discriminator(args=args, node_num=node_num, device=device).to(device)
        self.G = Generator(args=args, node_num=node_num, device=device).to(device)
        self.optim_d = torch.optim.Adam(self.D.parameters(), lr=0.001)
        self.optim_g = torch.optim.Adam(self.G.parameters(), lr=0.001)
        self.node_num = node_num
        self.cri = nn.MSELoss()

    def train_batch(self, data):
        # Model training
        self.G.train()
        self.D.train()

        feature, graph_c, graph_h, graph_mask, time_d, time_h, time_m, s_index, batch, index = (
            data['fea'].to(device), data['graph_c'].to(device), data['graph_h'].to(device), data['graph_mask'].to(device),
            data['time_d'].to(device), data['time_h'].to(device), data['time_m'].to(device),
            data['s_index'].to(device), data['batch'].to(device), data['index'].to(device))

        feature_fake = self.G(graph_c, graph_h, time_d, time_h, time_m, s_index, batch, index)
        context = self.D.context_gen(graph_mask, graph_h, time_d, time_h, time_m, s_index, batch)
        h_fake = self.D(feature_fake.detach())
        h = self.D(feature)
        score = self.D.ano_score(h, context)
        score_fake = self.D.ano_score(h_fake, context)

        real = torch.ones(score.size(0), requires_grad=False).to(device)
        fake = torch.zeros(score.size(0), requires_grad=False).to(device)

        # train D
        self.optim_d.zero_grad()
        real_loss = self.D.loss(score, real)
        fake_loss = self.D.loss(score_fake, fake)
        loss_d = (real_loss + fake_loss) / 2
        loss_d.backward(retain_graph=True)

        # train G
        self.optim_g.zero_grad()
        l1 = self.D.loss(score_fake, real)
        l2 = self.cri(feature_fake, feature)
        loss_g = l1 + args.lambda_ * l2
        loss_g.backward()

        self.optim_d.step()
        self.optim_g.step()

        return loss_d.item(), l1.item()

    def train(self, dataloader):
        print('Start training...')
        loss_list_d = []
        loss_list_g = []
        num_batch = len(dataloader)

        for epoch in range(self.args.epochs):
            d_loss, g_loss = 0, 0
            for i, data in enumerate(dataloader):
                print("epoch:", epoch)
                loss_d_, loss_g_ = self.train_batch(data)
                d_loss += loss_d_
                g_loss += loss_g_
                loss_list_g.append(loss_g_)
                loss_list_d.append(loss_d_)

            print('epoch = {}, loss_d = {}, loss_g = {}'.format(epoch, d_loss / num_batch, g_loss / num_batch))

        torch.save(self.D.state_dict(), path)
        torch.save(self.G.state_dict(), path)
        x = range(len(loss_list_d))
        plt.plot(x, loss_list_d, label="loss_D")
        plt.plot(x, loss_list_g, label="loss_G")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = "....."
    print(args)

    train_dataloader, node_num = load_train_data(path, args.train_month, args.train_year, num_batch=500)
    gan = GAN_Train(args, node_num, device)
    gan.train(train_dataloader)









