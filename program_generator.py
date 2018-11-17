# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license available at
# https://github.com/facebookresearch/clevr-iep/blob/master/LICENSE
#
# Modifications by David Mascharka to update the code for compatibility with PyTorch >0.1 lead to:
# DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.
#
# This material is based upon work supported by the Assistant Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8721-05-C-0002 and/or FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the author(s) and
# do not necessarily reflect the views of the Assistant Secretary of Defense for Research and
# Engineering.
#
# © 2017 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or
# 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are
# defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than
# as specifically authorized by the U.S. Government may violate any copyrights that exist in this
# work.

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import h5py
from pathlib import Path
from torch.distributions import Categorical


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder_vocab_size=93,
                 decoder_vocab_size=44,
                 wordvec_dim=300,
                 hidden_dim=256,
                 rnn_num_layers=2,
                 rnn_dropout=0,
                 null_token=0,
                 start_token=1,
                 end_token=2):
        super().__init__()
        self.encoder_embed = nn.Embedding(encoder_vocab_size, wordvec_dim)
        self.encoder_rnn = nn.LSTM(wordvec_dim, hidden_dim, rnn_num_layers,
                                   dropout=rnn_dropout, batch_first=True)
        self.decoder_embed = nn.Embedding(decoder_vocab_size, wordvec_dim)
        self.decoder_rnn = nn.LSTM(wordvec_dim + hidden_dim, hidden_dim, rnn_num_layers,
                                   dropout=rnn_dropout, batch_first=True)
        self.decoder_linear = nn.Linear(hidden_dim, decoder_vocab_size)
        self.NULL = null_token
        self.START = start_token
        self.END = end_token
        self.policy_history = []
        self.reset_param()

    def reset_param(self):
        for name, param in self.encoder_rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0)

        for name, param in self.decoder_rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0)

        nn.init.xavier_normal_(self.decoder_linear.weight)
        nn.init.constant_(self.decoder_linear.bias, 0)

    def get_dims(self, x=None, y=None):
        V_in = self.encoder_embed.num_embeddings
        V_out = self.decoder_embed.num_embeddings
        D = self.encoder_embed.embedding_dim
        H = self.encoder_rnn.hidden_size
        L = self.encoder_rnn.num_layers

        N = x.size(0) if x is not None else None
        N = y.size(0) if N is None and y is not None else N
        T_in = x.size(1) if x is not None else None
        T_out = y.size(1) if y is not None else None
        return V_in, V_out, D, H, L, N, T_in, T_out

    def before_rnn(self, x, replace=0):
        N, T = x.size()
        idx = torch.LongTensor(N).fill_(T - 1)

        # Find the last non-null element in each sequence.
        x_cpu = x.cpu()
        for i in range(N):
            for t in range(T - 1):
                if x_cpu[i, t] != self.NULL and x_cpu[i, t + 1] == self.NULL:
                    idx[i] = t
                    break
        idx = idx.type_as(x)
        return x, idx

    def encoder(self, x):
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(x=x)
        x, idx = self.before_rnn(x)
        embed = self.encoder_embed(x)
        h0 = torch.zeros(L, N, H).type_as(embed)
        c0 = torch.zeros(L, N, H).type_as(embed)
        out, _ = self.encoder_rnn(embed, (h0, c0))

        # Pull out the hidden state for the last non-null value in each input
        idx = idx.view(N, 1, 1).expand(N, 1, H)
        return out.gather(1, idx).view(N, H)

    def decoder(self, encoded, y, h0=None, c0=None):
        V_in, V_out, D, H, L, N, T_in, T_out = self.get_dims(y=y)

        if T_out > 1:
            y, _ = self.before_rnn(y)
        y_embed = self.decoder_embed(y)
        encoded_repeat = encoded.view(N, 1, H)
        encoded_repeat = encoded_repeat.expand(N, T_out, H)
        rnn_input = torch.cat([encoded_repeat, y_embed], 2)
        if h0 is None:
            h0 = torch.zeros(L, N, H).type_as(encoded)
        if c0 is None:
            c0 = torch.zeros(L, N, H).type_as(encoded)
        rnn_output, (ht, ct) = self.decoder_rnn(rnn_input, (h0, c0))

        rnn_output_2d = rnn_output.contiguous().view(N * T_out, H)
        output_logprobs = self.decoder_linear(rnn_output_2d).view(N, T_out, V_out)

        return output_logprobs, ht, ct

    def forward(self, x, max_length=30, mode='soft'):
        N, T = x.size(0), max_length
        encoded = self.encoder(x)
        cur_input = torch.LongTensor(N, 1).fill_(self.START).cuda()
        y_probs = []
        policy_hist = []
        h, c = None, None
        for t in range(T):
            # Compute logprobs
            logprobs, h, c = self.decoder(encoded, cur_input, h0=h, c0=c)
            probs = F.softmax(logprobs.view(N, -1), dim=-1)
            if mode == 'soft':
                y_probs.append(probs)
                _, cur_input = probs.max(1)
            elif mode == 'gumbel':
                y_probs.append(F.gumbel_softmax(logprobs, hard=True))
                _, cur_input = probs.max(1)
            elif mode == 'hard':
                # Distrubution probability
                m = Categorical(probs)
                # Sampling
                action = m.sample()
                # Create output vector (one-hot)
                y = torch.zeros(1, probs.size(1)).cuda()
                y[0, action.item()] = 1
                y_probs.append(y)
                # Update policy history
                policy_hist.append(torch.squeeze(torch.matmul(y, probs.view(-1, 1))))

            _, cur_input = probs.max(1)
            cur_input = torch.unsqueeze(cur_input, 0)
            # Stop if generated token is an end_token
            if (cur_input[0].data == self.END and t >= 2) or t >= max_length:
                break

        if mode == 'hard':
            self.policy_history.append(torch.stack(policy_hist))
        y_probs = torch.squeeze(torch.stack(y_probs, 1))
        return y_probs

    def reinforce_backward(self, reward):
        policy_loss = []
        for sample, rew in zip(self.policy_history, reward):
            policy_loss.append(torch.sum(torch.mul(torch.log(sample), rew).mul(-1), -1))
        policy_loss = torch.stack(policy_loss).sum() / len(policy_loss)
        policy_loss.backward()
        self.policy_history = []
