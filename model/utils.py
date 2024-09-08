import os
import torch
import numpy as np
from cytoolz import curry
from torch_geometric.data import Data, Batch
import networkx as nx


class Search:
    def __init__(self, adj_mat, maxsize, hops):
        self.hops = hops
        self.maxsize = maxsize
        self.adj_mat = adj_mat

    def search(self, seed):
        n = k_hop_neighbor(self.adj_mat, seed, self.hops)
        neighbor = list(set(n))
        neighbor.sort(key=n.index)
        neighbor.insert(0, seed)
        # print(list(neighbor)[:self.maxsize])

        return list(neighbor)[:self.maxsize]

    @curry
    def process(self, path, seed):
        subgraph_path = os.path.join(path, 'subgraph{}'.format(seed))
        if not os.path.isfile(subgraph_path) or os.stat(subgraph_path).st_size == 0:
            neighbor = self.search(seed)
            torch.save(neighbor, subgraph_path)
        else:
            print('File of node {} exists.'.format(seed))

    def search_all(self, node_num, path):
        neighbor = {}
        if os.path.isfile(path + '_neighbor') and os.stat(path + '_neighbor').st_size != 0:
            print("Exists neighbor file")
            neighbor = torch.load(path + '_neighbor')
        else:
            print("Extracting subgraphs")
            os.system('mkdir {}'.format(path))


            for i in range(node_num):
                self.process(path, i)
            print("Finish Extracting")

            for i in range(node_num):
                neighbor[i] = torch.load(os.path.join(path, 'subgraph{}'.format(i)))
            torch.save(neighbor, path + '_neighbor')
            os.system('rm -r {}'.format(path))
            print("Finish Writing")
        return neighbor


class Subgraph:
    # Class for subgraph extraction
    def __init__(self, node_num, edge_index, edge_weight, path, adj, args):
        self.batch_size = args.batch_size
        self.path = path
        self.adj = adj
        self.edge_weight = np.array(edge_weight)
        self.edge_index = np.array(edge_index)
        self.edge_num = edge_index[0].size(0)
        self.node_num = node_num
        self.maxsize = args.subgraph_size
        self.neighbor_search = Search(self.adj, maxsize=self.maxsize, hops=args.n_order)

        self.neighbor = {}
        self.adj_list = {}
        self.weight_list = {}
        self.subgraph_data = {}
        self.subgraph = {}
        self.subgraph_edge_attr = {}


    def process_adj_list(self):
        for i in range(self.node_num):
            self.adj_list[i] = set()
            self.weight_list[i] = {}
        for i in range(self.edge_num):
            u, v = self.edge_index[0][i], self.edge_index[1][i]
            weight = self.edge_weight[i, 0]
            self.adj_list[u].add(v)
            self.adj_list[v].add(u)
            self.weight_list[u].update({"%d" % v: weight})
            self.weight_list[v].update({"%d" % u: weight})

    def adjust_edge(self, idx):
        dic = {}
        for i in range(len(idx)):
            dic[idx[i]] = i

        new_index = [[], []]
        new_att = []
        nodes = set(idx)
        for i in idx:
            edge = list(self.adj_list[i] & nodes)
            w = [self.weight_list[i]["%d" % _] for _ in edge]
            edge = [dic[_] for _ in edge]
            new_index[0] += len(edge) * [dic[i]]
            new_index[1] += edge
            new_att += w
        new_att = np.expand_dims(np.array(new_att), axis=1)
        return torch.LongTensor(new_index), torch.Tensor(new_att)

    def adjust_x(self, t, idx):
        x = self.x[list(t * self.node_num + np.array(idx))]
        return x

    def extract(self):
        if os.path.isfile(self.path + '_subgraph') and os.stat(self.path + '_subgraph').st_size != 0:
            print("Exists subgraph file")
            self.subgraph = torch.load(self.path + '_subgraph')
            return

        self.neighbor = self.neighbor_search.search_all(self.node_num, self.path)
        self.process_adj_list()
        for i in range(self.node_num):
            nodes = self.neighbor[i][:self.maxsize]
            edge, attr = self.adjust_edge(nodes)
            self.subgraph[i] = edge
            self.subgraph_edge_attr[i] = attr
        # torch.save(self.subgraph, self.path+'_subgraph')

    def feature_build(self, x):
        self.batch_size = int(x.shape[0] / self.node_num)
        self.x = x

        for t in range(self.batch_size):
            for i in range(self.node_num):
                nodes = self.neighbor[i][:self.maxsize]
                feature = self.adjust_x(t, nodes)
                self.subgraph_data[t, i] = Data(feature, self.subgraph[i], self.subgraph_edge_attr[i])

    def search(self, node_list):
        batch, index = [], []
        size = 0
        for t in range(self.batch_size):
            for node in node_list:
                batch.append(self.subgraph_data[t, node])
                index.append(size)
                size += self.subgraph_data[t, node].x.size(0)

        index = torch.tensor(index)
        batch = Batch().from_data_list(batch)
        return batch, index


def sliding_window(x, step, seq_num, seq_len):
    le = step * seq_num + seq_len
    feature = x[:, - seq_len:]
    neighbor_list = x[:, :le-step]
    window = neighbor_list.unfold(1, seq_len, step)
    return window, feature


def k_hop_neighbor(adj, seed, k):
    if k == 0:
        return seed
    elif k == 1:
        G = nx.from_numpy_array(adj)
        k_hop_neighbor1 = nx.single_source_shortest_path_length(G, seed, cutoff=k)
        return list(k_hop_neighbor1)
    elif k > 0:
        G = nx.from_numpy_array(adj)
        neighbor = []
        for i in range(k):
            k_hop_neighbor1 = nx.single_source_shortest_path_length(G, seed, cutoff=i + 1)
            k_hop_neighbor2 = nx.single_source_shortest_path_length(G, seed, cutoff=i)
            k_hop_neighbor = k_hop_neighbor1.keys() - k_hop_neighbor2.keys()
            neighbor += k_hop_neighbor
        return list(neighbor)


def process_adj(adj):
    adj = np.array(adj)
    node_num = adj.shape[0]
    adj = adj - np.identity(node_num)
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph())
    remove = list(nx.isolates(G))
    G.remove_nodes_from(remove)
    A = np.array(nx.adjacency_matrix(G).todense())
    return A, remove
