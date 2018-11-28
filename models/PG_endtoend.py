import torch
from torch import nn
from controller import Exec_unary_module, Exec_binary_module
import numpy as np
from program_generator import Seq2Seq


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class PG(nn.Module):
    def __init__(self, vocab, question_size, stem_dim, n_channel, n_answers, batch_size, decoder_mode, use_curriculum):
        super(PG, self).__init__()

        print("----------- Build Neural Network -----------")

        # Useful variables declaration
        self.question_size = question_size+1
        self.stem_dim = stem_dim
        self.n_answers = n_answers+1
        self.batch_size = batch_size
        self.decoder_mode = decoder_mode
        self.saved_output = None
        self.program_tokens = vocab['program_token_to_idx']
        self.program_idx = vocab['program_idx_to_token']
        self.conv_dim = self.stem_dim * self.stem_dim * 3 * 3
        self.addressing = []
        self.use_curriculum = use_curriculum
        self.curriculum_step = 0
        self.curriculum_len = 3

        # Program generator
        self.program_generator = Seq2Seq()

        # Memory
        self.memory = None
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

    def forward(self, feats, question):

        final_module_outputs = []
        self.saved_output = None
        self.program_generator.init_policy()

        # Curriculum learning program generator
        if self.use_curriculum:
            self.curriculum_step += 1
            if self.curriculum_step % 1000 and self.curriculum_len < 30 == 0:
                self.curriculum_len += 1
                self.curriculum_step = 0

        # Visual embedding
        v = self.stem(feats)

        # Loop on batch
        for b in range(self.batch_size):
            self.addressing = []
            # Features
            feat_input = v[b, :, :]
            feat_input = torch.unsqueeze(feat_input, 0)
            output = feat_input

            # Program
            question_input = torch.unsqueeze(question[b, :], 0)
            prog_var = self.program_generator(question_input, mode=self.decoder_mode, max_length=self.curriculum_len).data

            # Loop on programs
            for i in reversed(range(prog_var.size(0))):
                # Check most probably program type
                progr_var_input = prog_var[i, :]
                _, i_max = torch.max(progr_var_input.data, 0)
                i_max = int(i_max.data)
                module_type = self.program_idx[i_max]

                # NOP modules
                if module_type in {'<NULL>', '<START>', '<END>', '<UNK>'}:
                    continue

                # Scene module
                if module_type == 'scene':
                    self.saved_output = output
                    w1, w2 = self.load_unary_module(progr_var_input)
                    output = self.exec_unary_module(feat_input, w1, w2)
                    continue

                # Binary Modules
                if self.isBinary(module_type):
                    if self.saved_output is None: continue
                    w1, w2, w3 = self.load_binary_module(progr_var_input)
                    output = self.exec_binary_module(output, self.saved_output, w1, w2, w3)

                # Unary Modules
                else:
                    w1, w2 = self.load_unary_module(progr_var_input)
                    output = self.exec_unary_module(output, w1, w2)

            final_module_outputs.append(output)

        # Classifier
        out = torch.cat(final_module_outputs, 0)
        out = self.classifier(out)
        return out

    def read_memory(self, weights=None, idx=None):
        """ Read from memory (w X M)"""
        if weights is None:
            weights = torch.zeros(1, self.memory.size(0)).cuda()
            weights[0, idx] = 1

        self.addressing.append(weights[0].data)
        read = torch.mm(weights, self.memory)
        return read

    def load_unary_module(self, progr_var_input):
        """ Read the unary module from memory """
        w = torch.unsqueeze(progr_var_input, 0)
        mem_read = self.read_memory(weights=w)
        w1, w2, _ = torch.split(mem_read, [self.conv_dim, self.conv_dim, 2 * self.conv_dim], dim=-1)
        w1 = w1.view(self.stem_dim, self.stem_dim, 3, 3)
        w2 = w2.view(self.stem_dim, self.stem_dim, 3, 3)
        return w1, w2

    def load_binary_module(self, progr_var_input):
        """ Read the binary module from memory """
        w = torch.unsqueeze(progr_var_input, 0)
        mem_read = self.read_memory(weights=w)
        w1, w2, w3 = torch.split(mem_read, [2 * self.conv_dim, self.conv_dim, self.conv_dim], dim=-1)
        w1 = w1.view(self.stem_dim, self.stem_dim * 2, 3, 3)
        w2 = w2.view(self.stem_dim, self.stem_dim, 3, 3)
        w3 = w3.view(self.stem_dim, self.stem_dim, 3, 3)
        return w1, w2, w3

    def initalize_state(self):
        # Initialize stuff
        stdev = 1 / (np.sqrt(self.conv_dim))
        self.memory = nn.Parameter(nn.init.uniform_((torch.Tensor(44, self.conv_dim*4).cuda()), -stdev, stdev))
        self.curriculum_len = 5 if self.use_curriculum else 30

    def isBinary(self, module_type):
        # Check program is unary or binary
        rtn = False
        if 'equal' in module_type or module_type in {'intersect', 'union', 'less_than',
                                                     'greater_than'}:
            rtn = True
        return rtn

    def getData(self):
        return self.addressing, None

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params
