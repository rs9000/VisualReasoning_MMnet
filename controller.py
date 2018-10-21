import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(1)


class LstmEncoder(nn.Module):
    """ Question Encoder used by SAN models """
    def __init__(self, token_to_idx, rnn_dim, wordvec_dim=300,
                 rnn_num_layers=2, rnn_dropout=0):
        super(LstmEncoder, self).__init__()
        self.token_to_idx = token_to_idx
        self.NULL = token_to_idx['<NULL>']
        self.START = token_to_idx['<START>']
        self.END = token_to_idx['<END>']

        self.embed = nn.Embedding(len(token_to_idx), wordvec_dim)
        self.rnn = nn.LSTM(wordvec_dim, rnn_dim, rnn_num_layers,
                           dropout=rnn_dropout, batch_first=True)

    def forward(self, x):
        N, T = x.size()
        idx = torch.LongTensor(N).fill_(T - 1)

        # Find the last non-null element in each sequence
        x_cpu = x.data.cpu()
        for i in range(N):
            for t in range(T - 1):
                if x_cpu[i, t] != self.NULL and x_cpu[i, t + 1] == self.NULL:
                    idx[i] = t
                    break
        idx = idx.type_as(x.data).long()
        idx.requires_grad = False

        hs, _ = self.rnn(self.embed(x))
        return hs, idx


class StackedAttention(nn.Module):
    """ Stack attention """
    def __init__(self, input_dim, hidden_dim):
        super(StackedAttention, self).__init__()
        self.Wv = nn.Conv2d(input_dim, hidden_dim, kernel_size=1, padding=0)
        self.Wu = nn.Linear(input_dim, hidden_dim)
        self.Wp = nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0)
        self.hidden_dim = hidden_dim
        self.attention_maps = None

    def getMap(self):
        return torch.squeeze(self.attention_maps[0], 1)

    def forward(self, v, u):
        """
        Input:
        - v: N x D x H x W
        - u: N x D
        Returns:
        - next_u: N x D
        """
        N, K = v.size(0), self.hidden_dim
        D, H, W = v.size(1), v.size(2), v.size(3)
        v_proj = self.Wv(v)  # N x K x H x W
        u_proj = self.Wu(u)  # N x K
        u_proj_expand = u_proj.view(N, K, 1, 1).expand(N, K, H, W)
        h = F.tanh(v_proj + u_proj_expand)
        p = F.softmax(self.Wp(h).view(N, H * W)).view(N, 1, H, W)
        self.attention_maps = p.data.clone()

        v_tilde = (p.expand_as(v) * v).sum(3).sum(2).view(N, D)
        return v_tilde


class Unary_module(nn.Module):
    """ Resblock used as unary module """
    def __init__(self, num_input, num_output):
        super(Unary_module, self).__init__()

        self.c1 = nn.Conv2d(num_input, num_output, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(num_output, num_output, kernel_size=3, padding=1)
        self.saved_map = None

    def get_map(self):
        return self.saved_map

    def forward(self, x):
        out = self.c2(F.relu(self.c1(x)))
        out = F.relu(torch.add(x, out))
        self.saved_map = out[0]
        return out


class Binary_module(nn.Module):
    """ Resblock used as binary module """
    def __init__(self, num_input, num_output):
        super(Binary_module, self).__init__()

        self.c1 = nn.Conv2d(num_input, num_output, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(num_output, num_output, kernel_size=3, padding=1)
        self.c3 = nn.Conv2d(num_output, num_output, kernel_size=3, padding=1)
        self.saved_map = None

    def get_map(self):
        return self.saved_map

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        out1 = F.relu(self.c1(x))
        out = F.relu(self.c2(out1))
        out = self.c3(out)
        out = F.relu(torch.add(out, out1))
        self.saved_map = out[0]
        return out


class Exec_unary_module(nn.Module):
    """ A without-parameters function to execute unary module"""
    def __init__(self):
        super(Exec_unary_module, self).__init__()
        self.saved_map = None

    def get_map(self):
        return self.saved_map

    def forward(self, x, w1, w2):
        out = F.relu(F.conv2d(x, w1, padding=1))
        out = F.conv2d(out, w2, padding=1)
        out = F.relu(torch.add(x, out))
        self.saved_map = out[0]
        return out


class Exec_binary_module(nn.Module):
    """ A without-parameters function to execute binary module"""
    def __init__(self):
        super(Exec_binary_module, self).__init__()

        self.saved_map = None

    def get_map(self):
        return self.saved_map

    def forward(self, x1, x2, w1, w2, w3):

        if x1 is None or x2 is None:
            print("Error types")
            return x1

        x = torch.cat((x1, x2), 1)
        out1 = F.relu(F.conv2d(x, w1, padding=1))
        out = F.relu(F.conv2d(out1, w2, padding=1))
        out = F.conv2d(out, w3, padding=1)
        out = F.relu(torch.add(out, out1))
        self.saved_map = out[0]
        return out