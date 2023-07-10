from torch import nn


class AL_net(nn.Module):
    def __init__(self, embedding_dim, batch_size, vocab_size, token_len, hidden_size, dropout=0):
        super(AL_net, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.model1 = nn.Sequential(
            # nn.BatchNorm1d(102),
            nn.Conv1d(102, 256, kernel_size=5, padding='valid'),
            nn.MaxPool1d(3),
            nn.LeakyReLU(),
            nn.Conv1d(256, 64, kernel_size=3, padding='valid'),
            nn.MaxPool1d(2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(1216, 128),
            nn.Linear(128, 32),
            nn.Linear(32, 2),
        )

    def forward(self, inputs):
        inputs = self.embedding(inputs)
        outputs = self.model1(inputs)
        return outputs

