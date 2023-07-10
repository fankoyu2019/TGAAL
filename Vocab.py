import torch
from torch.utils import data
import collections


class VocabClass:
    def __init__(self, words_list=None, min_freq=0, reserved_words=None):
        if words_list is None:
            words_list = []
        if reserved_words is None:
            reserved_words = []
        counter = count_corpus(words_list)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_words
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def getWordsList(path):
    print("loading dataset... : " + path)
    words_list = []
    with open(path, 'r') as f:
        for line in f:
            words = []
            line = line[:-1]
            for i in range(int(len(line)) // 3):
                word = line[i * 3:(i + 1) * 3]
                words.append(word)
            words_list.append(words)

    return words_list


# @save
def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


# @save
def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))  # padding


# @save
def build_array_nmt(lines, vocab, num_steps):
    tokens = [vocab[l] for l in lines]
    tokens = [l + [vocab['<eos>']] for l in tokens]
    array = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in tokens])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


def load_array(data_arrays, batch_size, is_train=True):
    # construct dataset iter
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

