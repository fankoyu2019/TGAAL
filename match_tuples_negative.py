import torch
import Vocab

from utils.sorf2features import *
from DistanceCalculation import *
from Vocab import VocabClass
from utils.vocab_utils import vocab


def getSeqList(path):
    print("loading dataset : ", path)
    seq_list = []
    with open(path, 'r') as f:
        for line in f:
            line = line[:-1]
            seq_list.append(line)
    return seq_list


def get_Features(seq):
    kmer1 = get_kmer_frequency(k=1, seq=seq, kmer_list=kmer_1_list_name)
    kmer2 = get_kmer_frequency(k=2, seq=seq, kmer_list=kmer_2_list_name)
    kmer3 = get_kmer_frequency(k=3, seq=seq, kmer_list=kmer_3_list_name)

    features = np.concatenate([kmer1, kmer2, kmer3], axis=0)
    features = np.exp(features) / np.sum(np.exp(features))
    return features


# generate n-p pairs
source_words_data = Vocab.getWordsList("./data/train_negative_sorf.fa")
target_words_data = Vocab.getWordsList("./data/train_positive_sorf.fa")
src_tokens_array, src_valid_len = Vocab.build_array_nmt(source_words_data, vocab, num_steps=102)
tgt_tokens_array, tgt_valid_len = Vocab.build_array_nmt(target_words_data, vocab, num_steps=102)

ps_seq_list = getSeqList("./data/train_positive_sorf.fa")
ng_seq_list = getSeqList("./data/train_negative_sorf.fa")

ps_features_list, ng_features_list = [], []
for seq in ps_seq_list:
    features = get_Features(seq)
    ps_features_list.append(features)
for seq in ng_seq_list:
    features = get_Features(seq)
    ng_features_list.append(features)

ps_features_list, ng_features_list = np.array(ps_features_list), np.array(ng_features_list)
idx_list = Distance(ng_features_list, ps_features_list, k=3)
_, count_array = np.unique(idx_list, return_counts=True)

src, tgt = torch.IntTensor(), torch.IntTensor()
for index, idx in enumerate(idx_list):
    tran_token_array = tgt_tokens_array[idx]
    context = src_tokens_array[index].repeat(tran_token_array.shape[0], 1)
    src = torch.cat((src, context), 0)
    tgt = torch.cat((tgt, tran_token_array), 0)
torch.save(src, "./data/n_p_pair_src.pth")
torch.save(tgt, "./data/n_p_pair_tgt.pth")
