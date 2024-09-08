import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool, SAGPooling

### tcn module
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        y = self.tcn(inputs)
        out = self.linear(y[:, :, -1])
        return out

class Pool(nn.Module):
    def __init__(self, in_channels, ratio=1.0):
        super(Pool, self).__init__()
        self.sag_pool = SAGPooling(in_channels, ratio)

    def forward(self, x, edge, batch, type='mean_pool'):
        if type == 'mean_pool':
            return global_mean_pool(x, batch)
        elif type == 'max_pool':
            return global_max_pool(x, batch)
        elif type == 'sum_pool':
            return global_add_pool(x, batch)
        elif type == 'sag_pool':
            x1, _, _, batch, _, _ = self.sag_pool(x, edge, batch=batch)
            return global_mean_pool(x1, batch)

class DisScore(nn.Module):
    def __init__(self, hidden_size, context_len):
        super(DisScore, self).__init__()
        self.bilinear = nn.Bilinear(hidden_size, context_len, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input1, input2):
        output = self.bilinear(input1, input2)
        output = output.squeeze(dim=1)
        output = self.sigmoid(output)
        return output

class STconv_D(nn.Module):
    def __init__(self, hidden_size, neighbor_num):
        super(STconv_D, self).__init__()
        self.neighbor_num = neighbor_num
        self.hidden_size = hidden_size
        self.tanh = nn.Tanh()
        self.tcn = TCN(hidden_size, hidden_size, [hidden_size] * 1, 2, 0.2)
        self.gconv = GCNConv(hidden_size, hidden_size)

    def forward(self, x, edge_index, edge_attr):
        hidden = []
        for i in range(self.neighbor_num):
            hidden.append(self.tanh(self.gconv(x[:, i, :], edge_index, edge_attr)))
        hidden = torch.stack(hidden)
        hidden = self.tanh(self.tcn(hidden.permute(1, 2, 0)))
        return hidden

class STconv_G(nn.Module):
    def __init__(self, seq_len, hidden_size, input_size=1):
        super(STconv_G, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.tcn = TCN(hidden_size, 1, [hidden_size] * 1, 2, 0.2)
        self.fc = nn.Linear(input_size, hidden_size)
        self.gconv = GCNConv(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, x, edge_index, edge_attr):
        x = self.fc(x.unsqueeze(-1))
        x = self.tanh(x)

        hidden = []
        for step in range(self.seq_len):
            g_h = self.gconv(x[:, step, :], edge_index, edge_attr)
            hidden.append(g_h)
        hidden = torch.stack(hidden)
        hidden = self.tcn(hidden.permute(1, 2, 0))
        return hidden




