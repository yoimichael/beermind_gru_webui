import torch.nn as nn
import torch
import numpy as np


class baselineGRU(nn.Module):
    def __init__(self):
        super(baselineGRU, self).__init__()
        self.input_dim = 208
        self.hidden_dim = 100
        self.output_dim = 98
        self.batch_size = 1
        self.num_layers = 2
        self.dropout = 0.2
        self.bidirect = False
        self.cuda = False

        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.num_layers,
                          bias=True, dropout=self.dropout,
                          bidirectional=self.bidirect, batch_first=True)

        self.out = nn.Linear(self.hidden_dim, self.output_dim)

        # load model cache
        self.load_state_dict(torch.load("./model_cache", map_location='cpu'))

        self.zero_grad()
        self.init_hidden(self.batch_size)

    def init_hidden(self, batch_size=1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim)

    def forward(self, sequence):
        '''
        Takes in the sequence of the form
        (batch_size x sequence_length x input_dim) and
        returns the output of form
        (batch_size x sequence_length x output_dim)
        '''

        output, self.hidden = self.gru(sequence, self.hidden)
        return self.out(output)
