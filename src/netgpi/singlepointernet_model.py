""" This module contains a Torch NN model. The architecture is based on 
    https://arxiv.org/abs/1506.03134. The architecture in the paper
    expects to output multiple 'pointers', we however only want a 
    single pointer. For that reason we do not need a 'Decoder' and apply
    the modified attention on the LSTM output instead. """
# coding: utf-8

from utility import *
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM
from torch.nn.functional import log_softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SinglePointerNet(nn.Module):
    """  """
    def __init__(self, hidden_size,  num_layers, dropout, attention_dimension, apply_mask, embedding_dim, num_embeddings):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidir_mod = 2
        self.apply_mask = apply_mask
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim, padding_idx=21)
        self.rnn_e = LSTM(input_size=embedding_dim,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=dropout,
                          bidirectional=True,
                          batch_first=True)
        
        self.attention_dimension = attention_dimension
        
        self.V = nn.Parameter(torch.nn.init.xavier_normal_(torch.Tensor(1, self.attention_dimension)))
        self.W = nn.Parameter(torch.nn.init.xavier_normal_(torch.Tensor(self.attention_dimension, self.hidden_size * self.bidir_mod)))
        self.b = nn.Parameter(torch.randn(self.hidden_size * self.bidir_mod ))
        
    def forward(self, x, hidden, cell, lengths):
        bsize = x.shape[0]
        feature_dim = x.shape[1]
        x = self.embeddings(x)
        x = pack_padded_sequence(x, lengths, batch_first=True)
        x, hidden = self.rnn_e(x, (hidden,cell))
        x, _ = pad_packed_sequence(x, batch_first=True,total_length=feature_dim,padding_value=-9999999)
        a = x.reshape(-1,self.hidden_size * self.bidir_mod)
        a = F.linear(a, self.W,)
        z = torch.tanh(a)
        a = F.linear(self.V, z)
        a = a.reshape(bsize, -1)
        out = a#.view(bsize,-1)
        if self.apply_mask:
            for i, o in enumerate(out):
                out[i][lengths[i]:] = -99999
        out = log_softmax(out, dim=1)
        return out
    
    
    
    def init_hidden(self, batch_size):
#         init = torch.zeros(batch_size, self.hidden_size, device=device)
        init = torch.zeros(self.num_layers * self.bidir_mod, batch_size, self.hidden_size, device=DEVICE)
        return init
    