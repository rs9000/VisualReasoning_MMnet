import torch
from torch import nn
from controller import Exec_unary_module, Exec_binary_module
from memory import ReadHead
import numpy as np

torch.manual_seed(1)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class PG(nn.Module):
    def __init__(self, vocab, question_size, stem_dim, n_answers, batch_size):
        super(PG, self).__init__()

        print("----------- Build Neural Turing machine -----------")

        # Useful variables declaration
        self.question_size = question_size+1
        self.stem_dim = stem_dim
        self.n_answers = n_answers+1
        self.batch_size = batch_size
        self.saved_output = None

        # Layers
        self.program_tokens = vocab['program_token_to_idx']
        self.program_idx = vocab['program_idx_to_token']
        self.conv_dim = self.stem_dim*self.stem_dim*3*3

        # Memory
        self.memory = torch.nn.Parameter(torch.Tensor(45, self.conv_dim*4))
        self.read_head = ReadHead(45, self.conv_dim*4)

        # Executor
        self.exec_unary_module = Exec_unary_module()
        self.exec_binary_module = Exec_binary_module()

        self.stem = nn.Sequential(nn.Conv2d(1024, self.stem_dim, kernel_size=3, padding=1),
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
        self.initalize_state()

    def forward(self, feats, programs):

        final_module_outputs = []

        # Visual embedding
        v = self.stem(feats)

        for b in range(self.batch_size):
            self.read_head.clean_addressing()

            feat_input = v[b, :, :]
            feat_input = torch.unsqueeze(feat_input, 0)
            output = feat_input

            for i in reversed(programs.data[b].cpu().numpy()):
                module_type = self.program_idx[i]

                # NOP modules
                if module_type in {'<NULL>', '<START>', '<END>', '<UNK>'}:
                    continue

                # Scene module
                if module_type == 'scene':
                    self.saved_output = output
                    w1, w2 = self.load_unary_module(i)
                    output = self.exec_unary_module(feat_input, w1, w2)
                    continue

                # Binary modules
                if self.isBinary(module_type):
                    w1, w2, w3 = self.load_binary_module(i)
                    output = self.exec_binary_module(output, self.saved_output, w1, w2, w3)

                # Unary modules
                else:
                    w1, w2 = self.load_unary_module(i)
                    output = self.exec_unary_module(output, w1, w2)

            final_module_outputs.append(output)

        # Classifier
        out = torch.cat(final_module_outputs, 0)
        out = self.classifier(out)
        return out

    def load_unary_module(self, idx):
        """ Read the unary module from memory """
        mem_read = self.read_head(self.memory, idx=idx)
        w1, w2, _ = torch.split(mem_read, [self.conv_dim, self.conv_dim, 2 * self.conv_dim], dim=-1)
        w1 = w1.view(self.stem_dim, self.stem_dim, 3, 3)
        w2 = w2.view(self.stem_dim, self.stem_dim, 3, 3)
        return w1, w2

    def load_binary_module(self, idx):
        """ Read the binary module from memory """
        mem_read = self.read_head(self.memory, idx=idx)
        w1, w2, w3 = torch.split(mem_read, [2 * self.conv_dim, self.conv_dim, self.conv_dim], dim=-1)
        w1 = w1.view(self.stem_dim, self.stem_dim * 2, 3, 3)
        w2 = w2.view(self.stem_dim, self.stem_dim, 3, 3)
        w3 = w3.view(self.stem_dim, self.stem_dim, 3, 3)
        return w1, w2, w3

    def initalize_state(self):
        # Initialize stuff
        stdev = 1 / (np.sqrt(self.conv_dim*2))
        self.memory = nn.Parameter(nn.init.uniform_((torch.Tensor(45, self.conv_dim*4).cuda()), -stdev, stdev))
        self.read_head.clean_addressing()

    def isBinary(self, module_type):
        """ Check if selected module is binary or not """
        rtn = False
        if 'equal' in module_type or module_type in {'intersect', 'union', 'less_than',
                                                     'greater_than'}:
            rtn = True
        return rtn

    def getData(self):
        return self.memory, self.read_head.get_weights()

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params
