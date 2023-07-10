from math import inf
import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PositionalEncoding(nn.Module):
    """位置编码
    Defined in :numref:`sec_self-attention-and-positional-encoding`"""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class MyTransformer(nn.Module):
    def __init__(self, embedding_dim, batch_size, vocab_size, token_len, hidden_size, dropout=0):
        super(MyTransformer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.transformer = nn.Transformer(d_model=embedding_dim, num_encoder_layers=3, num_decoder_layers=3,
                                          dim_feedforward=512)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout=dropout)
        self.predictor = nn.Linear(embedding_dim, vocab_size)

    def forward(self, src, tgt):
        # 生成mask
        # tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.transformer, sz=tgt.shape[-1]).bool().to(device)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask( sz=tgt.shape[-1]).bool().to(device)
        src_key_padding_mask = MyTransformer.get_key_padding_mask(src).to(device)
        tgt_key_padding_mask = MyTransformer.get_key_padding_mask(tgt).to(device)
        # 对src和tgt进行编码
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src).permute(1, 0, 2)
        tgt = self.positional_encoding(tgt).permute(1, 0, 2)

        # 将准备好的数据送给transformer
        output = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask)

        """
                这里直接返回transformer的结果。因为训练和推理时的行为不一样，
                所以在该模型外再进行线性层的预测。
        """
        return output


    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 1] = -inf
        return key_padding_mask
