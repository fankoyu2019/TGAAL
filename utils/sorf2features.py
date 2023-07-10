import csv
import math
import re
from itertools import product
from collections import defaultdict

import numpy as np
import pandas as pd


# return list = [lens / 303]
def get_lens(seq):
    return [len(seq) / 303]


# gc_content with gc_ratio
def gc_content_with_gc_ratio(seq):
    g, c = 0, 0
    lens = len(seq)
    g = seq.count('G')
    c = seq.count('C')
    # print(g, c)
    gc_content = (g + c) / lens
    if c == 0:
        gc_ratio = 0
    else:
        gc_ratio = g / c
    gc = [gc_content, gc_ratio]
    return gc


t_arr = np.zeros((6, 4))
f_arr = np.zeros((6, 4))
nuc_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def total_nucleic_bia():
    datasets = pd.read_csv('./sorf_datasets_600.csv')
    cal_total_bia_list = []
    for i in range(len(datasets)):
        sorf_coding = int(datasets.loc[i, 'sorf_coding'])
        bia_forward = datasets.loc[i, 'bia_forward']
        bia_backward = datasets.loc[i, 'bia_backward']
        cal_total_bia_list.append((sorf_coding, bia_forward, bia_backward))

    for k in cal_total_bia_list:
        sorf_coding, bia_forward, bia_backward = k
        if sorf_coding:
            for i in range(3):
                if isinstance(bia_forward, str):
                    t_arr[i, nuc_dict[bia_forward[i]]] += 1
                t_arr[i + 3, nuc_dict[bia_backward[i]]] += 1
        else:
            for i in range(3):
                if isinstance(bia_forward, str):
                    f_arr[i, nuc_dict[bia_forward[i]]] += 1
                f_arr[i + 3, nuc_dict[bia_backward[i]]] += 1

    for i in range(6):
        t_arr[i] /= np.sum(t_arr[i])
        f_arr[i] /= np.sum(f_arr[i])


def nucleic_bia(bia_forward, bia_backward):
    bia = 0.0
    for i in range(3):
        if isinstance(bia_forward, str):
            bia += math.log(f_arr[i, nuc_dict[bia_forward[i]]] / t_arr[i, nuc_dict[bia_forward[i]]])
        bia += math.log(f_arr[i + 3, nuc_dict[bia_backward[i]]] / t_arr[i + 3, nuc_dict[bia_backward[i]]])
    return [bia]


# k-mer
def nucleotide_type(k):
    z = []
    for i in product('ACGT', repeat=k):  # 笛卡尔积（有放回抽样排列）
        z.append(''.join(i))  # 把('A,A,A')转变成（AAA）形式
    return z


kmer_1_list_name = nucleotide_type(k=1)
kmer_2_list_name = nucleotide_type(k=2)
kmer_3_list_name = nucleotide_type(k=3)
kmer_4_list_name = nucleotide_type(k=4)


def get_kmer_frequency(k, seq, kmer_list):
    frequencies = []
    for i in kmer_list:
        number = 0
        for j in range(0, len(seq) - k + 1):
            if seq[j:j + k] == i:
                number += 1
        char_frequency = number / (len(seq) - k + 1)
        frequencies.append(char_frequency)
    return frequencies


# g-gap g=1,2,3
def get_ggap(g, seq):
    gap_dict = {}
    gap_frequencies = []
    nucleobase = ['A', 'T', 'G', 'C']
    for nucl1 in nucleobase:
        for nucl2 in nucleobase:
            gap_dict[nucl1 + '*' * g + nucl2] = 0
    for i in range(0, len(seq) - (g + 2) + 1):
        str = seq[i] + '*' * g + seq[i + g + 1]
        gap_dict[str] += 1
    # print(gap_dict)
    for i_value in gap_dict.values():
        gap_frequency = i_value / (len(seq) - (g + 2) + 1)
        gap_frequencies.append(gap_frequency)
    return gap_frequencies


# input g, return g_bigap_name_list
def g_bigap_name_list(g):
    g_name_list = []
    nucleobase = ['A', 'T', 'G', 'C']
    for nucl1 in nucleobase:
        for nucl2 in nucleobase:
            for nucl3 in nucleobase:
                for nucl4 in nucleobase:
                    g_name_list.append(nucl1 + nucl2 + '*' * g + nucl3 + nucl4)
    return g_name_list


g_bigap_2_list_name = g_bigap_name_list(2)
g_bigap_3_list_name = g_bigap_name_list(3)


# g-bigap
def get_gbigap(g, seq):
    bigap_dict = {}
    bigap_frequencies = []
    nucleobase = ['A', 'T', 'G', 'C']
    for nucl1 in nucleobase:
        for nucl2 in nucleobase:
            for nucl3 in nucleobase:
                for nucl4 in nucleobase:
                    bigap_dict[nucl1 + nucl2 + '*' * g + nucl3 + nucl4] = 0
    for i in range(0, len(seq) - (g + 4) + 1):
        str = seq[i] + seq[i + 1] + '*' * g + seq[i + 1 + g + 1] + seq[i + 1 + g + 2]
        bigap_dict[str] += 1
    for i_value in bigap_dict.values():
        bigap_frequency = 0
        if (len(seq) - (g + 4) + 1) > 0:
            bigap_frequency = i_value / (len(seq) - (g + 4) + 1)
        bigap_frequencies.append(bigap_frequency)
    # print(bigap_dict)
    return bigap_frequencies


def get_Features(seq, bia_forward, bia_backward):
    lens = get_lens(seq)
    gc = gc_content_with_gc_ratio(seq)
    bia = nucleic_bia(bia_forward, bia_backward)
    kmer1 = get_kmer_frequency(k=1, seq=seq, kmer_list=kmer_1_list_name)
    kmer2 = get_kmer_frequency(k=2, seq=seq, kmer_list=kmer_2_list_name)
    kmer3 = get_kmer_frequency(k=3, seq=seq, kmer_list=kmer_3_list_name)
    bigap2 = get_gbigap(2, seq)
    bigap3 = get_gbigap(3, seq)

    sorf_features = np.concatenate([lens, gc, bia, kmer1, kmer2, kmer3, bigap2, bigap3],
                                   axis=0)
    return sorf_features

