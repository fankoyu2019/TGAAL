from torch import nn


class Discriminator(nn.Module):
    def __init__(self, embedding_dim, batch_size, vocab_size, token_len, hidden_size, dropout=0):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.sequential = nn.Sequential(
            nn.Linear(embedding_dim * token_len, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, inputs):
        # inputs = self.embedding(inputs)
        inputs = inputs.reshape(inputs.shape[0], -1)
        outputs = self.sequential(inputs)
        return outputs

