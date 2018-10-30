import torch
from torch import nn
from controller import Exec_unary_module, Exec_binary_module
import numpy as np

torch.manual_seed(1)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class PG(nn.Module):
    def __init__(self, vocab, question_size, stem_dim, n_channel, n_answers, batch_size):
        super(PG, self).__init__()

        print("----------- Build Neural Network -----------")

        # Useful variables declaration
        self.question_size = question_size+1
        self.stem_dim = stem_dim
        self.n_answers = n_answers+1
        self.batch_size = batch_size
        self.saved_output = None
        self.program_tokens = vocab['program_token_to_idx']
        self.program_idx = vocab['program_idx_to_token']
        self.conv_dim = self.stem_dim * self.stem_dim * 3 * 3
        self.addressing_u = []
        self.addressing_b = []

        # Memory
        self.memory_unary = None
        self.memory_binary = None
        self.initalize_state()

        # Executor
        self.exec_unary_module = Exec_unary_module()
        self.exec_binary_module = Exec_binary_module()

        # Layers
        self.stem = nn.Sequential(nn.Conv2d(n_channel, self.stem_dim, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(self.stem_dim, self.stem_dim, kernel_size=3, padding=1),
                                  nn.ReLU()
                                  )

        self.classifier = nn.Sequential(nn.Conv2d(self.stem_dim, 512, kernel_size=1),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2),
                                        Flatten(),
                                        nn.Linear(512*7*7, 1024),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(1024, self.n_answers)  # note no softmax here
                                        )

    def forward(self, feats, programs):

        final_module_outputs = []

        # Visual embedding
        feats = self.stem(feats)

        # Loop on batch
        for b in range(self.batch_size):
            self.addressing_u = []
            self.addressing_b = []

            feat_input = feats[b, :, :]
            feat_input = torch.unsqueeze(feat_input, 0)
            output = feat_input

            # Loop on programs
            for i in reversed(programs.data[b].cpu().numpy()):
                module_type = self.program_idx[i]

                # NOP modules
                if module_type in {'<NULL>', '<START>', '<END>', '<UNK>'}:
                    continue

                # Scene module
                if module_type == 'scene':
                    self.saved_output = output
                    _, idx = self.is_binary(i)
                    w1, w2 = self.load_unary_module(idx)
                    output = self.exec_unary_module(feat_input, w1, w2)
                    continue

                isbinary, idx = self.is_binary(i)

                # Binary modules
                if isbinary:
                    w1, w2, w3 = self.load_binary_module(idx)
                    output = self.exec_binary_module(output, self.saved_output, w1, w2, w3)

                # Unary modules
                else:
                    w1, w2 = self.load_unary_module(idx)
                    output = self.exec_unary_module(output, w1, w2)

            final_module_outputs.append(output)

        # Classifier
        out = torch.cat(final_module_outputs, 0)
        out = self.classifier(out)
        return out

    def read_memory(self, memory, weights=None, idx=None):
        """ Read from memory (w X M)"""
        if weights is None:
            weights = torch.zeros(1, memory.size(0)).cuda()
            weights[0, idx] = 1

        if memory.size(0) == self.memory_unary.size(0):
            self.addressing_u.append(weights[0].data)
        else:
            self.addressing_b.append(weights[0].data)

        read = torch.mm(weights, memory)
        return read

    def load_unary_module(self, idx):
        """ Read the unary module from memory """
        read = self.read_memory(self.memory_unary, idx=idx)
        w1, w2 = torch.split(read, [self.conv_dim, self.conv_dim], dim=-1)
        w1 = w1.view(self.stem_dim, self.stem_dim, 3, 3)
        w2 = w2.view(self.stem_dim, self.stem_dim, 3, 3)
        return w1, w2

    def load_binary_module(self, idx):
        """ Read the binary module from memory """
        read = self.read_memory(self.memory_binary, idx=idx)
        w1, w2, w3 = torch.split(read, [2*self.conv_dim, self.conv_dim, self.conv_dim], dim=-1)
        w1 = w1.view(self.stem_dim, self.stem_dim*2, 3, 3)
        w2 = w2.view(self.stem_dim, self.stem_dim, 3, 3)
        w3 = w3.view(self.stem_dim, self.stem_dim, 3, 3)
        return w1, w2, w3

    def initalize_state(self):
        # Initialize stuff
        stdev = 1 / (np.sqrt(self.conv_dim))
        self.memory_unary = nn.Parameter(nn.init.uniform_(torch.Tensor(31, self.conv_dim*2).cuda(), -stdev, stdev))
        self.memory_binary = nn.Parameter(nn.init.uniform_(torch.Tensor(9, self.conv_dim*4).cuda(), -stdev, stdev))

    def is_binary(self, idx):
        """ Check if selected module is binary or not and map in memory"""
        binary = [5, 6, 7, 8, 9, 26, 27, 28, 42]
        unary = [4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 29, 30, 31, 32, 33, 34, 35, 36, 37,
                 38, 39, 40, 41, 43]

        if idx in unary:
            return False, unary.index(idx)

        if idx in binary:
            return True, binary.index(idx)

        return 0  # Error module not exist

    def getData(self):
        return self.addressing_u, self.addressing_b

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params
