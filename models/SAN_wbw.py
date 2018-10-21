import torch
from torch import nn
from controller import LstmEncoder, StackedAttention

torch.manual_seed(1)


class SAN(nn.Module):
    def __init__(self, vocab, question_size, stem_dim, n_answers, batch_size):
        super(SAN, self).__init__()

        print("----------- Build Neural Turing machine -----------")

        # Useful variables declaration
        self.question_size = question_size+1
        self.stem_dim = stem_dim
        self.n_answers = n_answers+1
        self.batch_size = batch_size
        self.attention_list = []

        # Layers
        question_tokens = vocab['question_token_to_idx']
        self.rnn = LstmEncoder(question_tokens, self.stem_dim)
        self.attention = StackedAttention(self.stem_dim, 512)

        self.stem = nn.Sequential(nn.Conv2d(1024, self.stem_dim, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(self.stem_dim, self.stem_dim, kernel_size=3, padding=1),
                                  nn.ReLU()
                                  )

        self.classifier = nn.Sequential(nn.Linear(self.stem_dim, 1024),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(1024, self.n_answers)  # note no softmax here
                                        )

    def forward(self, feats, question):

        self.init()

        # Visual embedding
        v = self.stem(feats)
        # Question embedding
        q, q_len = self.rnn(question)

        for b in range(self.batch_size):
            v1 = v[b, :, :]
            v1 = torch.unsqueeze(v1, 0)
            q1 = q[b, 0, :]
            for i in range(q_len[b]):
                if i != 0:
                    q1 = q1 + q[b, i, :]
                q1 = self.attention(v1, q1.view(1, 1, -1))
            self.attention_list.append(q1)

        out = torch.stack(self.attention_list)
        out = torch.squeeze(out, 1)
        out = self.classifier(out)
        return out

    def init(self):
        self.attention_list = []

    def getData(self):
        return self.attention.getMap()

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params
