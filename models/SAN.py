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

        # Visual embedding
        v = self.stem(feats)

        # Question embedding
        q, q_len = self.rnn(question)
        q_len = q_len.view(self.batch_size, 1, 1).expand(self.batch_size, 1, self.stem_dim)
        q = q.gather(1, q_len).view(self.batch_size, self.stem_dim)

        for i in range(3):
            u = self.attention(v, q)
            q = u + q

        out = self.classifier(q)
        return out

    def getData(self):
        return self.attention.getMap()

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params
