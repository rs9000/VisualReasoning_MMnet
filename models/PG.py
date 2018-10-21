import torch
from torch import nn
from controller import Unary_module, Binary_module

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
        self.function_modules = {}  # holds our modules

        # Init modules
        for module_name in self.program_tokens:
            if module_name in ['<NULL>', '<START>', '<END>', '<UNK>']:
                continue  # we don't need modules for the placeholders
            # figure out which module we want we use
            if 'equal' in module_name or module_name in {'union', 'intersect', 'less_than', 'greater_than'}:
                module = Binary_module(2*self.stem_dim, self.stem_dim)
            else:
                module = Unary_module(self.stem_dim, self.stem_dim)

            # add the module to our dictionary and register its parameters so it can learn
            self.function_modules[module_name] = module
            self.add_module(module_name, module)

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

    def forward(self, feats, programs):

        final_module_outputs = []

        # Visual embedding
        v = self.stem(feats)

        for b in range(self.batch_size):
            feat_input = v[b, :, :]
            feat_input = torch.unsqueeze(feat_input, 0)
            output = feat_input

            for i in reversed(programs.data[b].cpu().numpy()):
                module_type = self.program_idx[i]

                # NOP modules
                if module_type in {'<NULL>', '<START>', '<END>', '<UNK>'}:
                    continue

                # Load module
                module = self.function_modules[module_type]

                # Scene module
                if module_type == 'scene':
                    # store the previous output; it will be needed later
                    self.saved_output = output
                    output = module(feat_input)
                    continue

                # Binary Modules
                if 'equal' in module_type or module_type in {'intersect', 'union', 'less_than',
                                                             'greater_than'}:
                    output = module(output, self.saved_output)

                # Unary Modules
                else:
                    output = module(output)

            final_module_outputs.append(output)

        # Classifier
        out = torch.cat(final_module_outputs, 0)
        out = self.classifier(out)
        return out

    def getData(self):
        return self.function_modules['union'].get_map()

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params
