import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn import Module,ModuleList
from torch.nn import Dropout, LayerNorm
from torch.nn import MultiheadAttention
import copy
import math
#
#
# class PositionalEmbedding(nn.Module):
#
#     def __init__(self, d_model, max_len=512):
#         super().__init__()
#
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model).float()
#         pe.require_grad = False
#
#         position = torch.arange(0, max_len).float().unsqueeze(1)
#         div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
#
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         return self.pe[:, :x.size(1)]
#
# def _get_clones(module, N):
#     return ModuleList([copy.deepcopy(module) for i in range(N)])
#
# def get_padding_mask(x, x_lens):
#     """
#     :param x: (seq_len, batch_size, feature_dim)
#     :param x_lens: sequence lengths within a batch with size (batch_size,)
#     :return: padding_mask with size (batch_size, seq_len)
#     """
#     if not type(x_lens) == list:
#         x_lens = x_lens.tolist()
#     seq_len, batch_size, _ = x.size()
#     mask = torch.ones(batch_size, seq_len, device=x.device)
#     for seq, seq_len in enumerate(x_lens):
#         mask[seq, :seq_len] = 0
#     #mask = mask.bool()
#     return mask
#
# class AttentionLayer(Module):
#     def __init__(self, d_model, n_heads, dropout=0.0):
#         super(AttentionLayer, self).__init__()
#         self.multihead_attn = MultiheadAttention(d_model, n_heads, dropout=dropout)
#
#         self.norm1 = LayerNorm(d_model)
#         self.dropout1 = Dropout(dropout)
#
#         self.feedforward = PositionwiseFeedForward(d_model=64, d_ff=64, dropout=0.1)
#
#
#     def forward(self, query, key, src_key_padding_mask=None):
#
#         src2, att = self.multihead_attn(query, key, key,key_padding_mask=src_key_padding_mask)
#         src = query + self.dropout1(src2)
#         src = self.norm1(src)
#         src = self.feedforward(src)
#         return src
#
# class Attention(Module):
#     def __init__(self, n_layers, d_model, n_heads, dropout=0.0):
#         super(Attention, self).__init__()
#         self.n_layers = n_layers
#         self.layers = _get_clones(AttentionLayer(d_model, n_heads, dropout), n_layers)
#
#     def forward(self, query, key, x_padding_mask):
#         for layer in self.layers:
#             query = layer(query,key, src_key_padding_mask=x_padding_mask)
#         return query
#
# class RNN(nn.Module):
#     def __init__(self, d_in, d_out, n_layers=1, bi=True, dropout=0.2):
#         super(RNN, self).__init__()
#         self.rnn = nn.GRU(input_size=d_in, hidden_size=d_out, bidirectional=bi, num_layers=n_layers, dropout=dropout, batch_first=True)
#
#     def forward(self, x, x_len):
#         x_packed = pack_padded_sequence(x, x_len, batch_first=True)
#         x_out = self.rnn(x_packed)[0]
#         x_padded = pad_packed_sequence(x_out, batch_first=True)[0]
#         return x_padded
#
# class GELU(nn.Module):
#     """
#     Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
#     """
#
#     def forward(self, x):
#         return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
#
# class PositionwiseFeedForward(nn.Module):
#     "Implements FFN equation."
#
#     def __init__(self, d_model, d_ff, dropout=0.1):
#         super(PositionwiseFeedForward, self).__init__()
#         self.w_1 = nn.Linear(d_model, d_ff)
#         self.w_2 = nn.Linear(d_ff, d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = GELU()
#
#     def forward(self, x):
#         return self.w_2(self.dropout(self.activation(self.w_1(x))))
#
#
#
# class Model(nn.Module):
#     def __init__(self, param):
#         super(Model, self).__init__()
#         self.inp = nn.Linear(param.fea_dim, 64, bias=False)
#         self.rnn = RNN(64, 64, n_layers=4, bi=False,dropout=0.2)
#         self.att = Attention(2, 64, 4)
#         self.mlp_head = nn.Linear(64, 8,bias=True)
#         self.adaptive_max_pool = nn.AdaptiveAvgPool1d(1)
#
#     def forward(self, x, length):
#
#         x = self.inp(x)
#
#         x = self.rnn(x, length)
#
#         x = x.transpose(0, 1)
#         mask = get_padding_mask(x, length)
#         x = self.att(x, x, mask)
#         x = x.transpose(0, 1)
#
#         x = self.adaptive_max_pool(x.permute(0, 2, 1)).permute(0, 2, 1).squeeze(1)
#         y = self.mlp_head(x)
#         return y



class Model(nn.Module):
    def __init__(self,args):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(args.fea_dim,1024)
        self.linear2 = nn.Linear(1024, 8)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.adaptive_max_pool = nn.AdaptiveMaxPool1d(1)


    def forward(self, x, length):
        y = self.adaptive_max_pool(x.permute(0, 2, 1)).permute(0, 2, 1).squeeze(1)
        y = self.drop(self.relu(self.linear1(y)))
        y = self.relu(self.linear2(y))
        return y