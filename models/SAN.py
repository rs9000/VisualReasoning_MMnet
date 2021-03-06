import torch
from torch import nn
from controller import LstmEncoder, StackedAttention

torch.manual_seed(1)


class SAN(nn.Module):
    def __init__(self, vocab, question_size, stem_dim, n_channel, n_answers, batch_size):
        super(SAN, self).__init__()

        print("----------- Build Neural Network -----------")

        # Useful variables declaration
        self.question_size = question_size+1
        self.stem_dim = stem_dim
        self.n_answers = n_answers+1
        self.batch_size = batch_size
        question_tokens = vocab['question_token_to_idx']

        # Layers
        self.rnn = LstmEncoder(question_tokens, self.stem_dim)
        self.attention = StackedAttention(self.stem_dim, 512)

        self.stem = nn.Sequential(nn.Conv2d(n_channel, self.stem_dim, kernel_size=3, padding=1),
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
        # Trunk each question sequence at t = question_len
        q = q.gather(1, q_len).view(self.batch_size, self.stem_dim)

        # Attention
        for i in range(3):
            u = self.attention(v, q)
            q = u + q

        # Classifier
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
